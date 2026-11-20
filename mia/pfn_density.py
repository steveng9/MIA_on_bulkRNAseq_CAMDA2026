"""TabPFN conditional densities and truncated TabPFN releases, for MeLoMIA-TabPFN.

Runs in the `sota` environment (TabPFN needs Python >= 3.12), started by
`mia/attacks/melomia_tabpfn.py` as

    <sota python> -m mia.pfn_density <jobs.json>

and must therefore import nothing from the main stack except the TabPFN
generator itself.  A job file lists *units*; the worker claims each with an
O_EXCL lock file, skips units whose output already exists, and writes outputs
atomically, so several workers (on one or two GPUs) can share one cache.

Two unit kinds:

  terms   per-gene log p_TabPFN(x_c | label, genes < c) for every query row,
          with a given context table, for one chunk of genes.  This is the
          chain rule the TabPFN generator samples from (column 0 = label, then
          genes in their natural order, each regressed on everything before it;
          see tabpfn_extensions' `outliers_single_permutation_`), read as a
          density instead of sampled.
  shadow  a synthetic release from the real TabPFN generator, at the target's
          settings, fitted to a shadow context and truncated to the first G
          genes.  The generator fills columns in order and each column only
          conditions on earlier ones, so this is exactly the first G columns of
          a full release, at a fraction of the cost.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np


# ── Lock-claimed units ───────────────────────────────────────────────────────

def _claim(out: Path) -> bool:
    if out.exists():
        return False
    lock = out.with_name(out.name + ".lock")
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, f"{os.getpid()} {os.uname().nodename}\n".encode())
    os.close(fd)
    if out.exists():                       # finished between the check and the lock
        lock.unlink()
        return False
    return True


def _release(out: Path) -> None:
    lock = out.with_name(out.name + ".lock")
    if lock.exists():
        lock.unlink()


def _save_atomic(out: Path, **arrays) -> None:
    tmp = out.with_name(out.name + f".tmp{os.getpid()}.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, out)


# ── TabPFN ───────────────────────────────────────────────────────────────────

def _token() -> None:
    os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
    token_file = Path(os.environ.get("CAMDA_TABPFN_TOKEN_FILE",
                                     "~/.config/camda/tabpfn_token")).expanduser()
    if "TABPFN_TOKEN" not in os.environ and token_file.exists():
        os.environ["TABPFN_TOKEN"] = token_file.read_text().strip()


def _regressor(n_estimators, seed: int, device: str):
    _token()
    from tabpfn import TabPFNRegressor
    kw = {"device": device, "random_state": int(seed), "ignore_pretraining_limits": True}
    if n_estimators is not None:
        kw["n_estimators"] = int(n_estimators)
    return TabPFNRegressor(**kw)


def gene_logdens(reg, ctx_X, ctx_y, q_X, q_y, genes) -> np.ndarray:
    """(n_query, len(genes)) log p(x_c | label, x_<c) with the context as prompt."""
    import torch
    ctx_lab = ctx_y.astype(np.float32)[:, None]
    q_lab = q_y.astype(np.float32)[:, None]
    out = np.empty((len(q_X), len(genes)), dtype=np.float32)
    for j, c in enumerate(genes):
        f_ctx = np.hstack([ctx_lab, ctx_X[:, :c]])
        f_q = np.hstack([q_lab, q_X[:, :c]])
        reg.fit(f_ctx, ctx_X[:, c])
        pred = reg.predict(f_q, output_type="full")
        logits = torch.as_tensor(pred["logits"])
        crit = pred["criterion"].to(logits.device)
        # The bar distribution keeps float64 borders; match them.
        dt = crit.borders.dtype
        logits = logits.to(dt)
        target = torch.as_tensor(q_X[:, c], dtype=dt, device=logits.device)
        with torch.no_grad():
            out[:, j] = (-crit.forward(logits, target)).float().cpu().numpy()
    return out


def _arrays(path: str, cache: dict) -> dict:
    if path not in cache:
        with np.load(path) as z:
            cache[path] = {"X": z["X"].astype(np.float32), "y": z["y"].astype(np.int64)}
    return cache[path]


def run_terms(unit: dict, cache: dict) -> None:
    out = Path(unit["out"])
    ctx, q = _arrays(unit["ctx"], cache), _arrays(unit["query"], cache)
    if "reg" not in cache:
        cache["reg"] = _regressor(unit["n_estimators"], unit["seed"], unit["device"])
    genes = list(range(unit["g0"], unit["g1"]))
    t0 = time.time()
    lp = gene_logdens(cache["reg"], ctx["X"], ctx["y"], q["X"], q["y"], genes)
    _save_atomic(out, logdens=lp, genes=np.array(genes))
    print(f"[pfn] {out.parent.name}/{out.name}: {len(genes)} genes x {len(q['X'])} "
          f"queries, ctx {len(ctx['X'])}, {time.time() - t0:.0f}s", flush=True)


def run_shadow(unit: dict, cache: dict) -> None:
    import random
    import torch
    from mia.generators.tabpfn import TabPFNGenerator
    out = Path(unit["out"])
    d = _arrays(unit["ctx"], cache)
    G = int(unit["n_genes"])
    seed = int(unit["seed"])
    random.seed(seed)               # the extension draws feature permutations from `random`
    t0 = time.time()
    gen = TabPFNGenerator(seed=seed, device=unit["device"], verbose=False,
                          **unit.get("generator_params", {}))
    gen.fit(d["X"][:, :G], d["y"], int(unit["n_classes"]))
    X, y = gen.sample(int(unit["n"]))
    _save_atomic(out, X=np.asarray(X, dtype=np.float32), y=np.asarray(y, dtype=np.int64))
    del gen
    torch.cuda.empty_cache()
    print(f"[pfn] shadow {out.name}: {len(X)} rows x {G} genes, "
          f"{time.time() - t0:.0f}s", flush=True)


def _is_oom(e: BaseException) -> bool:
    return "out of memory" in str(e).lower()


def _run_unit(unit: dict, cache: dict, retries: int = 30, wait: int = 120) -> None:
    """Run one unit, retrying on CUDA out-of-memory: the GPU is shared, so an
    OOM usually means another process holds memory for a while."""
    for attempt in range(retries + 1):
        try:
            if unit["kind"] == "terms":
                return run_terms(unit, cache)
            if unit["kind"] == "shadow":
                return run_shadow(unit, cache)
            raise ValueError(unit["kind"])
        except Exception as e:  # noqa: BLE001 -- re-raised unless OOM
            if not _is_oom(e) or attempt == retries:
                raise
            import torch
            cache.pop("reg", None)
            torch.cuda.empty_cache()
            print(f"[pfn] OOM on {Path(unit['out']).name} (attempt {attempt + 1}); "
                  f"retrying in {wait}s", flush=True)
            time.sleep(wait)


def main(job_file: str) -> None:
    job = json.loads(Path(job_file).read_text())
    cache: dict = {}
    for unit in job["units"]:
        out = Path(unit["out"])
        # A terms unit may read a shadow release that another worker is still
        # generating; leave it for a later pass rather than wait here.
        if unit["kind"] == "terms" and not Path(unit["ctx"]).exists():
            continue
        if not _claim(out):
            continue
        try:
            _run_unit(unit, cache)
        finally:
            _release(out)
        # Keep the data cache from growing across releases.
        keep = {"reg", unit.get("ctx"), unit.get("query")}
        for k in [k for k in cache if k not in keep]:
            cache.pop(k)


if __name__ == "__main__":
    main(sys.argv[1])
