"""Building and loading the target synthetic datasets.

One *target* is: a generator, trained on the member half of one canonical
split, plus the synthetic dataset it emitted.  The grid needs
4 generators x 5 splits = 20 targets per cohort, and every attack is scored
against the same 20, so a target is built once and cached.

The NoisyDiffusion column is a special case.  The blue team published synthetic
data generated from exactly these five splits, so by default we reuse it rather
than retraining: that keeps the ND column identical to what the CAMDA abstract
reported.  Pass `retrain=True` to train our own instead (needed for the
white-box comparison, where the target weights must be ours).
"""

from __future__ import annotations

import hashlib
import json
import zlib
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import datasets as D
from . import generators as G
from . import paths

GENERATORS = ("mvn", "cvae", "nd", "pgm")


# ─────────────────────────────────────────────────────────────────────────────
# Target identity
# ─────────────────────────────────────────────────────────────────────────────
#
# A target is named `<generator>` (canonical parameters, `default_params`) or
# `<generator>@<k>=<v>,<k>=<v>` (a variant).  The variant string *is* the
# parameter override, so a directory name says exactly what produced it, and
# the name flows unchanged through every attack: run ids, the index's
# `generator` column and MeLoMIA's proxy cache all key on it.  Filtering the
# index on `generator == "pgm"` therefore still selects only canonical targets.

def split_name(generator: str) -> tuple[str, dict]:
    """`"pgm@epsilon=0.1,n_bins=8"` -> `("pgm", {"epsilon": 0.1, "n_bins": 8})`."""
    import yaml
    base, _, variant = generator.partition("@")
    overrides = {}
    for item in filter(None, variant.split(",")):
        k, _, v = item.partition("=")
        overrides[k] = yaml.safe_load(v)
    return base, overrides


def variant_name(generator: str, dataset: str, overrides: dict | None) -> str:
    """Canonical target name for `generator` with `overrides` applied.

    Overrides equal to the default are dropped, so the canonical target keeps
    its plain name (`pgm@epsilon=10` is `pgm`) and is reused, not rebuilt.
    """
    defaults = default_params(generator, dataset)
    diff = {k: v for k, v in (overrides or {}).items() if defaults.get(k) != v}
    if not diff:
        return generator
    def fmt(v):
        # lists/tuples as a-b-c: no spaces or brackets in directory names
        return "-".join(str(x) for x in v) if isinstance(v, (list, tuple)) else v
    return generator + "@" + ",".join(f"{k}={fmt(diff[k])}" for k in sorted(diff))


# ─────────────────────────────────────────────────────────────────────────────
# Known-broken targets
# ─────────────────────────────────────────────────────────────────────────────
#
# DP-PGM targets built with binning=dp_quantile / dp_uniform BEFORE generator
# commit cd5d1d5 (2026-09-23) read their DP bin edges with a biased estimator
# (negative noisy cells clipped, empty cells keep their positive noise), so the
# per-gene bounds sat near the ends of (0, 24) even at eps=1000 -- median 2.0 /
# 22.0 against a true 9.5 / 13.5.  The DP accounting of those targets is correct,
# but the data is effectively 4 equal bins over (2, 22) and useless as a DP-PGM
# release.  They are recognisable by name: `binning=dp_*` without
# `edge_estimator=threshold` (the generator default is still "clip", for
# reproducibility).  See results/BROKEN.md and FINDINGS 10i.

BROKEN_DP_EDGES = "BROKEN_DP_EDGES"


class BrokenTargetError(RuntimeError):
    pass


def broken_reason(generator: str) -> str | None:
    """Why `generator` (a target name) must not be used, or None if it is fine."""
    base, over = split_name(str(generator))
    if (base == "pgm" and str(over.get("binning", "")).startswith("dp_")
            and over.get("edge_estimator", "clip") != "threshold"):
        return (f"{BROKEN_DP_EDGES}: dp_* bin edges read with the biased 'clip' "
                "estimator (bounds ~2/22 instead of ~9.5/13.5); see results/BROKEN.md")
    return None


def fingerprint(dataset: str, generator: str, split: int) -> str:
    """Content hash of a target's released data.

    Anything cached from a target must be keyed on this, not on the name: the
    DP-PGM targets were rebuilt in place on 2026-09-20 under the same name, and
    a name-keyed cache would go on serving features of the old generator.
    """
    f = _files(dataset, generator, split)
    h = hashlib.sha1()
    for key in ("X", "y"):
        h.update(f[key].read_bytes())
    return h.hexdigest()[:16]


def target_record(dataset: str, generator: str, split: int) -> dict:
    """What a run should record about the target it attacked."""
    meta = json.loads(_files(dataset, generator, split)["meta"].read_text())
    return {"fingerprint": fingerprint(dataset, generator, split),
            "params": meta.get("resolved_params", meta.get("params")),
            "seed": meta.get("seed"), "source": meta.get("source")}


def _files(dataset: str, generator: str, split: int) -> dict:
    d = paths.target_dir(dataset, generator, split)
    return {
        "dir": d,
        "X": d / "synthetic_data.csv",
        "y": d / "synthetic_labels.csv",
        "meta": d / "meta.json",
        "model": d / "generator.pt",
    }


def exists(dataset: str, generator: str, split: int) -> bool:
    f = _files(dataset, generator, split)
    return f["X"].exists() and f["y"].exists() and f["meta"].exists()


# ─────────────────────────────────────────────────────────────────────────────
# Building
# ─────────────────────────────────────────────────────────────────────────────

def default_params(generator: str, dataset: str) -> dict:
    """Hyperparameters that reproduce the challenge's released synthetic data."""
    if generator == "mvn":
        return {"noise_level": 0.7}
    if generator == "cvae":
        return {"preprocess": "standard", "optimizer": "adam", "scheduler": "none",
                "lr": 1e-3, "batch_size": 64, "num_iters": 10000, "z_dim": 128,
                "beta": 0.001, "condition_type": "embedding", "early_stopping": False}
    if generator == "nd":
        return {"epochs": 200, "dp_noise_multiplier": 1e-5, "smote_upsample_to": 3000}
    if generator == "pgm":
        return {"epsilon": 10.0, "n_bins": 4, "n_1way": 978, "n_2way": 0,
                "joint_mode": True, "pgm_iters": 1000}
    raise KeyError(generator)


def build_target(
    dataset: str,
    generator: str,
    split: int,
    params: dict | None = None,
    *,
    device: str = "cuda",
    force: bool = False,
    save_model: bool = True,
    retrain_nd: bool = False,
    allow_broken: bool = False,
) -> Path:
    """Train one target generator and write its synthetic dataset.  Idempotent."""
    reason = broken_reason(generator)
    if reason and not allow_broken:
        raise BrokenTargetError(f"refusing to build or reuse {dataset}/{generator}/"
                                f"split_{split}: {reason}. Add "
                                "edge_estimator=threshold.")
    f = _files(dataset, generator, split)
    if exists(dataset, generator, split) and not force:
        print(f"  [target] {dataset}/{generator}/split_{split} cached", flush=True)
        return f["dir"]

    if generator == "nd" and not retrain_nd:
        return _import_published_nd(dataset, split)
    if generator.startswith("nd@") and not retrain_nd:
        raise ValueError(f"{generator}: an ND variant has no published data; "
                         "pass retrain_nd=True")

    base, overrides = split_name(generator)
    params = {**default_params(base, dataset), **(params or {}), **overrides}
    X_train, y_train, member_ids = D.training_subset(dataset, split)
    n_classes = D.n_classes(dataset)

    # Seeded from the base generator, so every variant of one split shares its
    # randomness and a sweep's curve is not also a curve over seeds.  (This
    # used Python's `hash()`, which is salted per process: seeds of targets
    # built before 2026-09-21 are recorded in their meta.json but were not
    # reproducible from the code.)
    seed = 1000 * split + zlib.crc32(base.encode()) % 1000
    gen = G.build(base, seed=seed, device=device, **params)

    print(f"  [target] fitting {generator} on {dataset} split {split} "
          f"({X_train.shape[0]} members)", flush=True)
    gen.fit(X_train, y_train, n_classes)
    X_syn, y_syn = gen.sample(len(X_train))

    f["dir"].mkdir(parents=True, exist_ok=True)
    vocab = D.class_names(dataset)
    pd.DataFrame(X_syn, columns=D.gene_names(dataset)).to_csv(f["X"], index=False)
    pd.DataFrame({"label": [vocab[i] for i in y_syn]}).to_csv(f["y"], index=False)

    if save_model:
        try:
            gen.save(f["model"])
        except NotImplementedError:
            pass

    # `params` records only what the caller overrode, so a target built on a
    # default that later changes is indistinguishable from one built before.
    # That bit us with the DP-PGM accounting fixes, where the same config file
    # produced materially different generators either side of 2026-09-20, so
    # record the generator's full resolved state alongside it.
    resolved = {k: v for k, v in asdict(gen).items()
                if not k.startswith("_") and isinstance(v, (int, float, str, bool,
                                                            tuple, list, type(None)))}
    f["meta"].write_text(json.dumps({
        "dataset": dataset, "generator": base, "target": generator, "split": split,
        "params": params, "resolved_params": resolved,
        "seed": seed, "source": "trained",
        "n_synthetic": int(len(X_syn)), "n_train": int(len(X_train)),
    }, indent=2, default=str))
    print(f"  [target] wrote {f['X']}  shape={X_syn.shape}", flush=True)
    return f["dir"]


def _import_published_nd(dataset: str, split: int) -> Path:
    """Copy the blue team's published ND synthetic data into the target tree."""
    ds = D.spec(dataset)
    src_X = ds.nd_synthetic_dir / f"synthetic_data_split_{split}.csv"
    src_y = ds.nd_synthetic_dir / f"synthetic_labels_split_{split}.csv"
    if not src_X.exists():
        raise FileNotFoundError(
            f"{src_X} missing. Either clone CAMDA25_NoisyDiffusion or build the "
            "ND target with retrain_nd=True."
        )

    f = _files(dataset, "nd", split)
    f["dir"].mkdir(parents=True, exist_ok=True)

    X = pd.read_csv(src_X)
    X.columns = D.gene_names(dataset)      # published file numbers its columns
    X.to_csv(f["X"], index=False)
    y = pd.read_csv(src_y)
    pd.DataFrame({"label": y.iloc[:, 0].astype(str).values}).to_csv(f["y"], index=False)

    f["meta"].write_text(json.dumps({
        "dataset": dataset, "generator": "nd", "split": split,
        "params": default_params("nd", dataset), "seed": None,
        "source": "published-blue-team", "origin": str(src_X),
        "n_synthetic": int(len(X)),
    }, indent=2))
    print(f"  [target] imported published ND split {split}  shape={X.shape}", flush=True)
    return f["dir"]


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_target(dataset: str, generator: str, split: int,
                allow_broken: bool = False) -> dict:
    """Return {X (n, 978) float32, y_str, y_int, meta} for one target.

    Refuses known-broken targets (see `broken_reason`) unless `allow_broken`.
    """
    reason = broken_reason(generator)
    if reason and not allow_broken:
        raise BrokenTargetError(f"{dataset}/{generator}/split_{split}: {reason}")
    f = _files(dataset, generator, split)
    if not exists(dataset, generator, split):
        raise FileNotFoundError(
            f"Target {dataset}/{generator}/split_{split} not built. "
            "Run scripts/build_targets.py first."
        )
    X = pd.read_csv(f["X"]).values.astype(np.float32)
    y_str = pd.read_csv(f["y"]).iloc[:, 0].astype(str).values
    meta = json.loads(f["meta"].read_text())

    vocab = {c: i for i, c in enumerate(D.class_names(dataset))}
    y_int = np.array([vocab.get(l, -1) for l in y_str], dtype=np.int64)
    return {"X": X, "y_str": y_str, "y_int": y_int, "meta": meta, "paths": f}
