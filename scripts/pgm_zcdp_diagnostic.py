#!/usr/bin/env python
"""How much fidelity is the DP-PGM budget accounting costing us?

DIAGNOSTIC ONLY.  This script does not change the shipped generator or its
privacy accounting; it monkeypatches the noise calibration inside one process
so the cost of the accounting can be measured.  Nothing it produces is a
privacy claim about the submitted DP-PGM data.  See docs/PGM_ATTACK_SURFACE.md
question 7.

The upstream fitter (steveng9/PrivateRNAseqGen, src/pgm_fitter.py) splits the
budget linearly over marginals:

    eps_per = frac * epsilon / len(cliques)
    sigma   = sqrt(2 ln(1.25/delta)) / eps_per

so sigma grows in proportion to the number of measurements -- basic sequential
composition.  Gaussian mechanisms compose additively in zero-concentrated DP
instead, where a sensitivity-1 release costs rho = 1/(2 sigma^2) and k of them
cost k/(2 sigma^2), giving sigma = sqrt(k / (2 rho)): growth in sqrt(k), not k.

At epsilon=10, delta=1e-5, k=1956 that is sigma 1436 against sigma 25 -- a 57x
noise reduction at an identical (epsilon, delta).  This runs both and reports
the fidelity gap.
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mia import datasets as D  # noqa: E402
from mia import fidelity as F  # noqa: E402
from mia import generators as G  # noqa: E402
from mia import paths  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "results" / "pgm_zcdp.csv"


def rho_from_eps_delta(epsilon: float, delta: float) -> float:
    """Largest rho whose zCDP-to-(eps, delta) conversion still lands at epsilon.

    The standard conversion is eps = rho + 2 sqrt(rho ln(1/delta)); solving the
    quadratic in sqrt(rho) inverts it exactly.
    """
    L = math.log(1.0 / delta)
    t = -math.sqrt(L) + math.sqrt(L + epsilon)   # t = sqrt(rho)
    return t * t


def patch_zcdp(fitter_cls):
    """Replace _build_measurements with a zCDP-calibrated version.

    Identical to upstream except for how sigma is derived: the per-order rho is
    budget_weights[order] * rho_total, and the order's k measurements share it,
    so sigma_order = sqrt(k_order / (2 * rho_order)).
    """
    original = fitter_cls._build_measurements

    def _build_measurements_zcdp(self, dataset, cliques_by_order):
        rho_total = rho_from_eps_delta(self.epsilon, self.delta)
        measurements = []
        for order_idx, cliques in enumerate(cliques_by_order):
            frac = self.budget_weights[order_idx]
            if not cliques or frac == 0.0:
                continue
            rho_order = frac * rho_total
            sigma = math.sqrt(len(cliques) / (2.0 * rho_order))
            print(f"  [zcdp] {order_idx + 1}-way: {len(cliques)} marginals, "
                  f"rho={rho_order:.4f}, sigma={sigma:.4f}", flush=True)
            for clique in cliques:
                true_marginal = dataset.project(clique).datavector()
                y = true_marginal + np.random.normal(0, sigma, true_marginal.shape)
                measurements.append((None, y, sigma, clique))
        return measurements

    fitter_cls._build_measurements = _build_measurements_zcdp
    return original


def run(accounting, X, y, m, n_classes, cfg, epsilon, seed=42):
    t0 = time.time()
    gen = G.build("pgm", epsilon=epsilon, pgm_iters=1000, seed=seed, **cfg)
    gen.fit(X[m], y[m], n_classes)
    X_syn, y_syn = gen.sample(int(m.sum()))
    row = F.evaluate(X_syn, y_syn, X[~m], y[~m], X[m], y[m])
    row.update(cfg, accounting=accounting, epsilon=epsilon,
               seconds=round(time.time() - t0, 1))
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="BRCA")
    p.add_argument("--split", type=int, default=1)
    p.add_argument("--epsilon", type=float, default=10.0)
    p.add_argument("--n_1way", type=int, default=978)
    p.add_argument("--n_bins", type=int, default=4)
    p.add_argument("--delta", type=float, default=1e-5)
    args = p.parse_args()

    cfg = dict(n_1way=args.n_1way, n_2way=0, n_bins=args.n_bins, joint_mode=True)

    X = D.load_expression(args.dataset).values.astype(np.float64)
    y = D.encode_subtypes(args.dataset, D.load_subtypes(args.dataset).values)
    m = D.membership_labels(args.dataset, args.split).astype(bool)
    n_classes = D.n_classes(args.dataset)

    rho = rho_from_eps_delta(args.epsilon, args.delta)
    print(f"epsilon={args.epsilon} delta={args.delta} -> rho={rho:.4f}\n", flush=True)

    rows = [run("basic (as shipped)", X, y, m, n_classes, cfg, args.epsilon)]

    sys.path.insert(0, str(paths.PGM_REPO / "src"))
    from pgm_fitter import PrivatePGMFitter  # noqa: E402
    patch_zcdp(PrivatePGMFitter)
    rows.append(run("zcdp (diagnostic)", X, y, m, n_classes, cfg, args.epsilon))

    print()
    for r in rows:
        print(f"{r['accounting']:20s} TSTR-F1={r['tstr_macro_f1']:.3f} "
              f"ratio={r['utility_ratio']:.3f} W1={r['wasserstein_mean']:.3f} "
              f"corr-MAE={r['corr_mae']:.3f} disc={r['discriminator_auc']:.3f} "
              f"({r['seconds']:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    df["dataset"], df["split"] = args.dataset, args.split
    if OUT.exists():
        df = pd.concat([pd.read_csv(OUT), df], ignore_index=True)
    key = ["dataset", "split", "epsilon", "accounting", "n_1way", "n_bins"]
    df.drop_duplicates(subset=key, keep="last").to_csv(OUT, index=False)
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
