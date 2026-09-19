"""The model zoo: one content-addressed store for every model and synthetic
dataset the project builds.

The pipeline produces exactly two kinds of thing, over and over, in every
experiment:

  fit     a generator fitted to some data
  sample  a synthetic dataset drawn from a fit

and the data a fit was trained on is itself either a slice of a real cohort or
another artifact's sample.  So the whole project is one provenance DAG rooted in
real samples, and every node in it is reusable: a generator fitted to split 3's
training half is the same object whether an experiment is calling it a target,
a base shadow, or a probe.

*Roles are not properties of artifacts.*  They are how one experiment chooses to
use them, and they live in `mia.zoo.roles`.  Keeping the two apart is what makes
reuse safe -- the store never has to know, and an experiment can take another
experiment's shadow and make it a target without anything being copied.

Layout under ARTIFACTS/zoo:

    index.jsonl          one line per artifact, append-only
    fit/<ab>/<id>/       model.pt, spec.json, closure.txt
    sample/<ab>/<id>/    data.npz, spec.json, closure.txt

`closure.txt` holds the real sample ids underlying the artifact, one per line.
It is the single most useful field in the store: membership labels for *any*
artifact, in *any* experiment, are recomputed from it rather than remembered,
so a reused shadow can never carry stale labels.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np

from .. import paths
from . import ids as I

ZOO = paths.ARTIFACTS / "zoo"
INDEX = ZOO / "index.jsonl"

KINDS = ("fit", "sample")


@dataclass
class Artifact:
    id: str
    kind: str                  # fit | sample
    dataset: str
    generator: str             # generator family (a sample inherits its fit's)
    params: dict               # generator hyperparameters, or sampling params
    source: str                # data reference this was produced from
    seed: int
    closure: str               # hash of the real sample ids underneath it
    n_closure: int
    n: Optional[int] = None    # rows, for a sample
    created: str = ""
    note: str = ""

    @property
    def dir(self) -> Path:
        return ZOO / self.kind / self.id[:2] / self.id

    @property
    def ref(self) -> str:
        return I.artifact_ref(self.id)

    def closure_ids(self) -> list:
        p = self.dir / "closure.txt"
        return p.read_text().split("\n") if p.exists() else []

    def membership(self, sample_ids: Iterable) -> np.ndarray:
        """Membership labels for arbitrary real samples under this artifact.

        Recomputed from the stored closure every time, which is why an artifact
        can be reused as a shadow in an experiment it was not built for.
        """
        members = set(self.closure_ids())
        return np.array([1 if str(s) in members else 0 for s in sample_ids], dtype=np.int64)


# ── Index ────────────────────────────────────────────────────────────────────

def _read_index() -> list[Artifact]:
    if not INDEX.exists():
        return []
    out = []
    for line in INDEX.read_text().splitlines():
        line = line.strip()
        if line:
            out.append(Artifact(**json.loads(line)))
    return out


def _append_index(a: Artifact) -> None:
    INDEX.parent.mkdir(parents=True, exist_ok=True)
    # Append-only with a single write call, so concurrent workers interleave
    # whole lines rather than corrupting each other's.
    with open(INDEX, "a") as f:
        f.write(json.dumps(asdict(a)) + "\n")


def all_artifacts() -> list[Artifact]:
    """Every artifact, de-duplicated by id (last record wins)."""
    seen = {}
    for a in _read_index():
        seen[a.id] = a
    return list(seen.values())


def get(artifact_id: str) -> Optional[Artifact]:
    for a in all_artifacts():
        if a.id == artifact_id:
            return a
    return None


def find(**query) -> list[Artifact]:
    """Artifacts matching every supplied field, newest first."""
    out = [a for a in all_artifacts()
           if all(getattr(a, k, None) == v for k, v in query.items())]
    return sorted(out, key=lambda a: a.created, reverse=True)


# ── Identity ─────────────────────────────────────────────────────────────────

def fit_id(dataset: str, generator: str, params: dict, source: str, seed: int) -> str:
    return I.digest({"kind": "fit", "dataset": dataset, "generator": generator,
                     "params": params, "source": source, "seed": seed})


def sample_id(source_fit: str, n: int, seed: int) -> str:
    return I.digest({"kind": "sample", "source": source_fit, "n": n, "seed": seed})


# ── Locking ──────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def claim(path: Path):
    """Claim an artifact directory for building, or yield None if someone else has.

    Parallel workers across GPUs share one store; whoever creates the lock file
    builds, everyone else moves on and picks the result up later.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.parent / f"{path.name}.lock"
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        yield None
        return
    try:
        os.write(fd, f"{os.getpid()}\n".encode())
        os.close(fd)
        yield path
    finally:
        lock.unlink(missing_ok=True)


def await_artifact(artifact_id: str, kind: str, timeout_s: float = 7200.0,
                   poll_s: float = 15.0) -> bool:
    """Block until another worker finishes building this artifact."""
    marker = ZOO / kind / artifact_id[:2] / artifact_id / "spec.json"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if marker.exists():
            return True
        time.sleep(poll_s)
    return False


# ── Building ─────────────────────────────────────────────────────────────────

def _write_common(d: Path, spec: dict, closure_ids: list) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / "closure.txt").write_text("\n".join(str(s) for s in sorted(closure_ids)))
    tmp = d / "spec.json.tmp"
    tmp.write_text(json.dumps(spec, indent=2, default=I._default))
    tmp.replace(d / "spec.json")        # spec.json last: it marks completion


def get_or_create_fit(
    dataset: str,
    generator: str,
    params: dict,
    source: str,
    seed: int,
    closure_ids: Iterable,
    build: Callable[[Path], None],
    note: str = "",
) -> Artifact:
    """Return the fitted generator for this spec, building it only if absent.

    `build` receives the artifact directory and must write the model into it.
    `closure_ids` are the real sample ids underlying the training data -- for a
    fit on synthetic data, the closure of the artifact that produced it.
    """
    closure_ids = list(closure_ids)
    aid = fit_id(dataset, generator, params, source, seed)
    existing = get(aid)
    if existing is not None and (existing.dir / "spec.json").exists():
        return existing

    d = ZOO / "fit" / aid[:2] / aid
    with claim(d) as claimed:
        if claimed is None:
            await_artifact(aid, "fit")
            found = get(aid)
            if found is not None:
                return found
            # Index line not visible yet; reconstruct the record locally.
            return _record(aid, "fit", dataset, generator, params, source, seed,
                           closure_ids, None, note, index=False)
        d.mkdir(parents=True, exist_ok=True)
        build(d)
        spec = {"id": aid, "kind": "fit", "dataset": dataset,
                "generator": generator, "params": params, "source": source,
                "seed": seed, "note": note}
        _write_common(d, spec, closure_ids)
    return _record(aid, "fit", dataset, generator, params, source, seed,
                   closure_ids, None, note)


def get_or_create_sample(
    fit: Artifact,
    n: int,
    seed: int,
    build: Callable[[Path], None],
    note: str = "",
) -> Artifact:
    """Return the synthetic dataset drawn from `fit`, building it only if absent."""
    aid = sample_id(fit.id, n, seed)
    existing = get(aid)
    if existing is not None and (existing.dir / "spec.json").exists():
        return existing

    d = ZOO / "sample" / aid[:2] / aid
    closure_ids = fit.closure_ids()
    with claim(d) as claimed:
        if claimed is None:
            await_artifact(aid, "sample")
            found = get(aid)
            if found is not None:
                return found
            return _record(aid, "sample", fit.dataset, fit.generator,
                           {"n": n}, fit.ref, seed, closure_ids, n, note,
                           index=False)
        d.mkdir(parents=True, exist_ok=True)
        build(d)
        spec = {"id": aid, "kind": "sample", "dataset": fit.dataset,
                "generator": fit.generator, "params": {"n": n},
                "source": fit.ref, "seed": seed, "note": note}
        _write_common(d, spec, closure_ids)
    return _record(aid, "sample", fit.dataset, fit.generator, {"n": n},
                   fit.ref, seed, closure_ids, n, note)


def _record(aid, kind, dataset, generator, params, source, seed, closure_ids,
            n, note, index: bool = True) -> Artifact:
    a = Artifact(id=aid, kind=kind, dataset=dataset, generator=generator,
                 params=params, source=source, seed=seed,
                 closure=I.closure_hash(closure_ids), n_closure=len(closure_ids),
                 n=n, created=time.strftime("%Y-%m-%dT%H:%M:%S"), note=note)
    if index:
        _append_index(a)
    return a


# ── Provenance ───────────────────────────────────────────────────────────────

def ancestors(artifact_id: str) -> list[str]:
    """Every artifact id upstream of this one, nearest first."""
    out, seen = [], set()
    cur = get(artifact_id)
    while cur is not None:
        kind, rest = I.parse_ref(cur.source)
        if kind != "artifact" or rest in seen:
            break
        seen.add(rest)
        out.append(rest)
        cur = get(rest)
    return out


def lineage(artifact_id: str) -> str:
    """Human-readable provenance chain back to real data."""
    parts, cur = [], get(artifact_id)
    while cur is not None:
        parts.append(f"{cur.kind}:{cur.generator}[{cur.id[:8]}]")
        kind, rest = I.parse_ref(cur.source)
        if kind != "artifact":
            parts.append(cur.source)
            break
        cur = get(rest)
    return " <- ".join(parts)
