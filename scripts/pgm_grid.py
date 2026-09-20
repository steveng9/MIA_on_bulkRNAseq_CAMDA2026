#!/usr/bin/env python
"""Parallel DP-PGM fidelity grid: configs x splits x composition.

`mbi.FactoredInference` is single-threaded -- a 978-gene fit pins one core at
100% and leaves the other 23 idle -- so the way to use this machine is many
concurrent fits, not more threads per fit.  Each fit holds about 0.75 GB, so
memory is not the binding constraint either.

Resumable: a job whose key is already in the output CSV is skipped, so an
interrupted grid can be relaunched with the same command.  Workers append
through `mia.csvlock`, which holds an exclusive lock across the read-modify-
write -- without it, parallel writers corrupt the file exactly as they once
corrupted results/index.csv.
"""

import os

# Must precede numpy: each worker gets a small thread budget so N workers do not
# oversubscribe.  The fit itself is single-threaded; this is for the BLAS calls
# in the fidelity metrics.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import argparse  # noqa: E402
import itertools  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mia import csvlock  # noqa: E402
from mia import datasets as D  # noqa: E402
from mia import fidelity as F  # noqa: E402
from mia import generators as G  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "results" / "pgm_sweep.csv"

KEY = ["dataset", "split", "epsilon", "composition",
       "n_1way", "n_2way", "n_bins", "joint_mode"]

GRID = [
    dict(n_1way=978, n_2way=0,  n_bins=4, joint_mode=True),   # as submitted
    dict(n_1way=978, n_2way=0,  n_bins=8, joint_mode=True),
    dict(n_1way=978, n_2way=0,  n_bins=2, joint_mode=True),
    dict(n_1way=400, n_2way=0,  n_bins=4, joint_mode=True),
    dict(n_1way=200, n_2way=0,  n_bins=4, joint_mode=True),
    dict(n_1way=100, n_2way=0,  n_bins=4, joint_mode=True),
    dict(n_1way=50,  n_2way=0,  n_bins=4, joint_mode=True),
    dict(n_1way=200, n_2way=0,  n_bins=8, joint_mode=True),
    dict(n_1way=200, n_2way=0,  n_bins=2, joint_mode=True),
    dict(n_1way=200, n_2way=50, n_bins=4, joint_mode=False),  # stratified
    dict(n_1way=100, n_2way=50, n_bins=4, joint_mode=False),
]


def label(job):
    return (f"{job['dataset']}/s{job['split']} {job['composition']:5s} "
            f"n1={job['n_1way']:<4d} n2={job['n_2way']:<3d} k={job['n_bins']} "
            f"{'joint' if job['joint_mode'] else 'strat'}")


def run_one(job):
    """Fit one configuration and score it.  Returns (label, row, error)."""
    t0 = time.time()
    try:
        X = D.load_expression(job["dataset"]).values.astype(np.float64)
        y = D.encode_subtypes(job["dataset"],
                              D.load_subtypes(job["dataset"]).values)
        m = D.membership_labels(job["dataset"], job["split"]).astype(bool)
        n_classes = D.n_classes(job["dataset"])

        cfg = {k: job[k] for k in ("n_1way", "n_2way", "n_bins", "joint_mode")}
        gen = G.build("pgm", epsilon=job["epsilon"], pgm_iters=job["pgm_iters"],
                      seed=job["seed"], composition=job["composition"], **cfg)
        gen.fit(X[m], y[m], n_classes)
        X_syn, y_syn = gen.sample(int(m.sum()))

        row = F.evaluate(X_syn, y_syn, X[~m], y[~m], X[m], y[m])
        row.update(job, seconds=round(time.time() - t0, 1))
        row.pop("pgm_iters", None)
        return label(job), row, None
    except Exception:
        return label(job), None, traceback.format_exc(limit=3)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="BRCA")
    p.add_argument("--splits", default="1", help="comma-separated, e.g. 1,2,3,4,5")
    p.add_argument("--epsilon", type=float, default=10.0)
    p.add_argument("--composition", default="zcdp",
                   help="zcdp, basic, or both")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--pgm-iters", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true",
                   help="re-run configurations already present in the CSV")
    args = p.parse_args()

    splits = [int(s) for s in args.splits.split(",")]
    comps = (["basic", "zcdp"] if args.composition == "both"
             else [args.composition])

    jobs = [dict(cfg, dataset=args.dataset, split=s, epsilon=args.epsilon,
                 composition=c, seed=args.seed, pgm_iters=args.pgm_iters)
            for cfg, s, c in itertools.product(GRID, splits, comps)]

    if not args.force and OUT.exists() and OUT.stat().st_size > 0:
        done = pd.read_csv(OUT)
        if set(KEY).issubset(done.columns):
            seen = {tuple(r) for r in done[KEY].itertuples(index=False)}
            before = len(jobs)
            jobs = [j for j in jobs if tuple(j[k] for k in KEY) not in seen]
            print(f"skipping {before - len(jobs)} already-recorded configurations")

    if not jobs:
        print("nothing to do")
        return

    print(f"{len(jobs)} jobs on {args.workers} workers "
          f"({os.environ['OMP_NUM_THREADS']} threads each)\n", flush=True)

    t0, ok, failed = time.time(), 0, 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, j): j for j in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            name, row, err = fut.result()
            if err is not None:
                failed += 1
                print(f"[{i}/{len(jobs)}] {name}  FAILED\n{err}", flush=True)
                continue
            csvlock.append_row(OUT, row, key=KEY)
            ok += 1
            print(f"[{i}/{len(jobs)}] {name}  "
                  f"TSTR-F1={row['tstr_macro_f1']:.3f} "
                  f"ratio={row['utility_ratio']:.3f} "
                  f"W1={row['wasserstein_mean']:.3f} "
                  f"corr-MAE={row['corr_mae']:.3f} "
                  f"disc={row['discriminator_auc']:.3f} "
                  f"({row['seconds']:.0f}s)", flush=True)

    print(f"\n{ok} ok, {failed} failed, {time.time() - t0:.0f}s wall -> {OUT}")


if __name__ == "__main__":
    main()
