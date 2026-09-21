#!/usr/bin/env python
"""Re-score every eps-sweep target with the generator's own bin cells.

The sweep's aggregation arms bin at quantiles of the auxiliary pool.  Against
`uniform` or `dp_quantile` targets those cells do not line up with the
generator's, so a low AUC there could be misalignment rather than privacy.
This re-scores the same targets with `edges="generator"` and writes
results/pgm_aligned_edges.csv, one row per (target, split), columns
auc_<arm>.  For legacy `quantile` targets the generator's edges are the private
percentiles, so those rows are an oracle, not an attack.
"""
import os, sys
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "2")
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from mia import metrics as M  # noqa: E402
from mia import datasets as D  # noqa: E402
from mamamia_aggregation import arms_for_split  # noqa: E402
from pgm_eps_sweep import slug  # noqa: E402


def one(r):
    yr, arms = arms_for_split(r["dataset"], int(r["split"]), int(r["n_bins"]),
                              float(r["epsilon"]), 1e-5, generator=r["target"],
                              edges="generator")
    ym = D.membership_labels(r["dataset"], int(r["split"])).astype(int)
    return {**{k: r[k] for k in ("dataset", "target", "binning", "n_bins",
                                 "epsilon", "split", "fingerprint")},
            **{f"auc_{slug(a)}": M.evaluate(ym, s)["auc"] for a, s in arms.items()}}


if __name__ == "__main__":
    d = pd.read_csv(ROOT / "results" / "pgm_eps_sweep.csv")
    d["binning"] = d["binning"].fillna("quantile")
    d["n_bins"] = d["n_bins"].fillna(4).astype(int)
    with ProcessPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 10) as ex:
        rows = list(ex.map(one, d.to_dict("records")))
    pd.DataFrame(rows).to_csv(ROOT / "results" / "pgm_aligned_edges.csv", index=False)
    print("ALIGNED COMPLETE")
