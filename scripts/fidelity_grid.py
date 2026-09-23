#!/usr/bin/env python
"""Fidelity and utility for the non-DP generators, in the eps sweep's metrics.

    python scripts/fidelity_grid.py

One row per (cohort, generator, split) -> results/fidelity_grid.csv, computed
with the same `fidelity.evaluate` call as scripts/pgm_eps_sweep.py, so these
rows sit directly beside the DP-PGM rows in results/pgm_eps_sweep.csv.
"""
import os
import sys
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from mia import datasets as D  # noqa: E402
from mia import fidelity as F  # noqa: E402
from mia import targets as T  # noqa: E402

OUT = ROOT / "results" / "fidelity_grid.csv"


def one(job):
    ds, gen, split = job
    tg = T.load_target(ds, gen, split)
    X = D.load_expression(ds).values.astype(np.float64)
    y = D.encode_subtypes(ds, D.load_subtypes(ds).values)
    m = D.membership_labels(ds, split).astype(bool)
    row = F.evaluate(tg["X"].astype(np.float64), tg["y_int"], X[~m], y[~m], X[m], y[m])
    return {**row, "dataset": ds, "generator": gen, "split": split,
            "n_syn": len(tg["X"]),
            "fingerprint": T.target_record(ds, gen, split)["fingerprint"]}


if __name__ == "__main__":
    jobs = [(ds, g, s) for ds in ("BRCA", "COMBINED") for g in ("mvn", "cvae", "nd")
            for s in (1, 2, 3, 4, 5) if T.exists(ds, g, s)]
    with ProcessPoolExecutor(10) as ex:
        rows = list(ex.map(one, jobs))
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"{len(rows)} rows -> {OUT}")
