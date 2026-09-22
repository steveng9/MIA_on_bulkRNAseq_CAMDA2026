#!/usr/bin/env python
"""How the attack's own binning affects what it sees: three edge choices.

    python scripts/pgm_attack_binning.py --workers 12

For every DP-PGM target in results/pgm_eps_sweep.csv, this re-scores the nine
MAMA-MIA aggregation arms three ways and writes one row per (target, edges) to
results/pgm_attack_binning.csv:

  * `aux`      -- equal-frequency edges over the candidate pool (the shipped
                  attack).  Black box.
  * `uniform`  -- equal width over the public range (0, 24), bin count assumed
                  known.  Black box.
  * `generator`-- the target's own cells.  WHITE BOX, a diagnostic only: it is
                  what a model-aware adversary would gain, and for legacy
                  `quantile` targets those edges are the private percentiles,
                  so that row is an oracle, not an attack.

Each row also carries how far the attack's cells sit from the generator's:
`edge_mae` and `edge_max` in expression units, and `cell_agreement`, the share
of candidate values the two binnings put in the same bin.  That is the axis for
"how much does binning mismatch cost the attack", which the black-box rows
answer and the white-box row bounds.
"""
import argparse
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
sys.path.insert(0, str(ROOT / "scripts"))
from mia import datasets as D  # noqa: E402
from mia import metrics as M  # noqa: E402
from mamamia_aggregation import (arms_for_split, attack_edges,  # noqa: E402
                                 edge_divergence, generator_edges)
from pgm_eps_sweep import slug  # noqa: E402

EDGES = ["aux", "uniform", "generator"]
OUT = ROOT / "results" / "pgm_attack_binning.csv"


def one(job):
    r, kind = job
    ds, split, k = r["dataset"], int(r["split"]), int(r["n_bins"])
    X = D.load_expression(ds).values.astype(np.float64)
    ge = generator_edges(ds, r["target"], split)
    ae = ge if kind == "generator" else attack_edges(X, k, kind)
    yr, arms = arms_for_split(ds, split, k, float(r["epsilon"]), 1e-5,
                              generator=r["target"], edges=ae)
    ym = D.membership_labels(ds, split).astype(int)
    return {**{c: r[c] for c in ("dataset", "target", "binning", "n_bins",
                                 "epsilon", "split", "fingerprint")},
            "attack_edges": kind, "white_box": kind == "generator",
            **edge_divergence(ae, ge, X, k),
            **{f"auc_{slug(a)}": M.evaluate(ym, s)["auc"] for a, s in arms.items()}}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--edges", nargs="+", default=EDGES, choices=EDGES)
    args = p.parse_args()
    d = pd.read_csv(ROOT / "results" / "pgm_eps_sweep.csv")
    d["binning"] = d["binning"].fillna("quantile")
    d["n_bins"] = d["n_bins"].fillna(4).astype(int)
    jobs = [(r, k) for r in d.to_dict("records") for k in args.edges]
    with ProcessPoolExecutor(args.workers) as ex:
        rows = list(ex.map(one, jobs))
    out = pd.DataFrame(rows)
    if OUT.exists():        # keep rows for targets this call did not cover
        old = pd.read_csv(OUT)
        key = ["fingerprint", "attack_edges"]
        out = pd.concat([old.merge(out[key], on=key, how="left", indicator=True)
                         .query('_merge == "left_only"').drop(columns="_merge"), out])
    out.to_csv(OUT, index=False)
    print(f"{len(rows)} rows -> {OUT}")
    print("ATTACK BINNING COMPLETE")
