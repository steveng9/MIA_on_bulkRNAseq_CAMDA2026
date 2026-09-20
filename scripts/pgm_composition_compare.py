#!/usr/bin/env python
"""Basic against zCDP composition, at identical (epsilon, delta).

Drives the generator's own `composition` flag -- nothing is monkeypatched -- so
what this measures is exactly what the generator ships.

Both arms satisfy (eps, delta)-DP for the measurements.  They differ only in
how the budget is divided over marginals: linearly under `basic`, as sqrt(k)
under `zcdp`.  Accounting details and the caveat about unbudgeted marginal
selection and discretisation are in docs/PGM_ATTACK_SURFACE.md.
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

OUT = Path(__file__).resolve().parent.parent / "results" / "pgm_composition.csv"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="BRCA")
    p.add_argument("--split", type=int, default=1)
    p.add_argument("--epsilon", type=float, default=10.0)
    p.add_argument("--n_1way", type=int, default=978)
    p.add_argument("--n_bins", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    cfg = dict(n_1way=args.n_1way, n_2way=0, n_bins=args.n_bins, joint_mode=True)

    X = D.load_expression(args.dataset).values.astype(np.float64)
    y = D.encode_subtypes(args.dataset, D.load_subtypes(args.dataset).values)
    m = D.membership_labels(args.dataset, args.split).astype(bool)
    n_classes = D.n_classes(args.dataset)

    rows = []
    for composition in ("basic", "zcdp"):
        t0 = time.time()
        gen = G.build("pgm", epsilon=args.epsilon, pgm_iters=1000, seed=args.seed,
                      composition=composition, **cfg)
        gen.fit(X[m], y[m], n_classes)
        X_syn, y_syn = gen.sample(int(m.sum()))
        row = F.evaluate(X_syn, y_syn, X[~m], y[~m], X[m], y[m])
        row.update(cfg, composition=composition, epsilon=args.epsilon,
                   dataset=args.dataset, split=args.split, seed=args.seed,
                   seconds=round(time.time() - t0, 1))
        rows.append(row)
        print(f"{composition:6s} TSTR-F1={row['tstr_macro_f1']:.3f} "
              f"ratio={row['utility_ratio']:.3f} W1={row['wasserstein_mean']:.3f} "
              f"corr-MAE={row['corr_mae']:.3f} disc={row['discriminator_auc']:.3f} "
              f"({row['seconds']:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    if OUT.exists():
        df = pd.concat([pd.read_csv(OUT), df], ignore_index=True)
    key = ["dataset", "split", "epsilon", "composition", "n_1way", "n_bins", "seed"]
    df.drop_duplicates(subset=key, keep="last").to_csv(OUT, index=False)
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
