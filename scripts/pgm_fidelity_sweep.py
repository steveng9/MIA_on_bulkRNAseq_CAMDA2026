#!/usr/bin/env python
"""Where does DP-PGM's budget actually go?

Each configuration is fitted on split 1's member half and scored with
`mia.fidelity`, writing to results/pgm_sweep.csv.

The axis that matters is the marginal count.  The upstream fitter splits epsilon
linearly across marginals, so per-marginal sigma grows linearly in their number:
at n_1way=978 in joint mode it measures 1956 marginals and lands at sigma=1436
on a dataset of 871 rows, which is noise with no signal under it.  Fewer genes
means a smaller sigma per gene but fewer genes released at all, and the
unmodelled ones are filled with a constant.  Somewhere in between is the best
release this generator can make at eps=10.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mia import datasets as D  # noqa: E402
from mia import fidelity as F  # noqa: E402
from mia import generators as G  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "results" / "pgm_sweep.csv"

GRID = [
    dict(n_1way=978, n_2way=0, n_bins=4, joint_mode=True),    # as submitted
    dict(n_1way=400, n_2way=0, n_bins=4, joint_mode=True),
    dict(n_1way=200, n_2way=0, n_bins=4, joint_mode=True),
    dict(n_1way=100, n_2way=0, n_bins=4, joint_mode=True),
    dict(n_1way=50,  n_2way=0, n_bins=4, joint_mode=True),
    dict(n_1way=200, n_2way=0, n_bins=8, joint_mode=True),
    dict(n_1way=200, n_2way=0, n_bins=2, joint_mode=True),
    dict(n_1way=200, n_2way=50, n_bins=4, joint_mode=False),  # stratified
    dict(n_1way=100, n_2way=50, n_bins=4, joint_mode=False),
]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="BRCA")
    p.add_argument("--split", type=int, default=1)
    p.add_argument("--epsilon", type=float, default=10.0)
    p.add_argument("--composition", default="zcdp", choices=["zcdp", "basic"])
    args = p.parse_args()

    X = D.load_expression(args.dataset).values.astype(np.float64)
    y = D.encode_subtypes(args.dataset, D.load_subtypes(args.dataset).values)
    m = D.membership_labels(args.dataset, args.split).astype(bool)
    n_classes = D.n_classes(args.dataset)

    rows = []
    for cfg in GRID:
        label = (f"n1={cfg['n_1way']} n2={cfg['n_2way']} k={cfg['n_bins']} "
                 f"{'joint' if cfg['joint_mode'] else 'strat'}")
        t0 = time.time()
        try:
            gen = G.build("pgm", epsilon=args.epsilon, pgm_iters=1000, seed=42,
                          composition=args.composition, **cfg)
            gen.fit(X[m], y[m], n_classes)
            X_syn, y_syn = gen.sample(int(m.sum()))
        except Exception as exc:
            print(f"{label:42s} FAILED {type(exc).__name__}: {exc}", flush=True)
            continue
        row = F.evaluate(X_syn, y_syn, X[~m], y[~m], X[m], y[m])
        row.update(cfg, dataset=args.dataset, split=args.split,
                   epsilon=args.epsilon, composition=args.composition,
                   seconds=round(time.time() - t0, 1))
        rows.append(row)
        print(f"{label:42s} TSTR-F1={row['tstr_macro_f1']:.3f} "
              f"ratio={row['utility_ratio']:.3f} "
              f"W1={row['wasserstein_mean']:.3f} "
              f"corr-MAE={row['corr_mae']:.3f} "
              f"disc={row['discriminator_auc']:.3f} "
              f"({row['seconds']:.0f}s)", flush=True)

        df = pd.DataFrame(rows)
        if OUT.exists():
            df = pd.concat([pd.read_csv(OUT), df], ignore_index=True)
        key = ["dataset", "split", "epsilon", "composition",
               "n_1way", "n_2way", "n_bins", "joint_mode"]
        df.drop_duplicates(subset=key, keep="last").to_csv(OUT, index=False)

    print(f"\n{len(rows)} configs -> {OUT}")


if __name__ == "__main__":
    main()
