#!/usr/bin/env python
"""How MAMA-MIA's evidence should be combined, against the corrected DP-PGM.

    python scripts/mamamia_aggregation.py --datasets BRCA COMBINED

Answers Steven's Q11 (ratio space vs log space) and the two questions that grew
out of it: whether the two marginal families should be weighted by inverse
noise variance, and whether scores should be centred within subtype.

Focal points need no selection here.  The cohorts carry exactly 978 genes and
the release sets n_1way=978, so "top 978 by variance" is every gene; with
n_2way=0 and label marginals on, the clique set is 978 one-way plus 978
gene x label, fixed by public configuration.  So every arm below targets the
same, known marginals and differs only in aggregation.

Writes results/mamamia_aggregation.csv.
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mia import csvlock, datasets as D, metrics as M, targets as T  # noqa: E402
from mia.attacks.mamamia import (EPS, _oneway_probs, _twoway_probs,  # noqa: E402
                                 digitize, quantile_bin_edges)

OUT = Path(__file__).resolve().parent.parent / "results" / "mamamia_aggregation.csv"


def center_by_class(s: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Subtract each subtype's mean score.

    Subtype is known to the adversary (it ships with the challenge data) and is
    strongly predictive of the score but not of membership, so leaving it in
    adds a within-class constant that dilutes the membership signal.
    """
    out = s.astype(float).copy()
    for c in np.unique(y):
        k = y == c
        out[k] -= out[k].mean()
    return out


def family_sigma(n_cliques: int, weight_idx: int, eps: float, delta: float,
                 weights=(0.33, 0.67)) -> float:
    """Reconstruct a marginal family's sigma from published parameters alone."""
    try:
        from snsynth.utils import cdp_rho
        rho = float(cdp_rho(eps, delta))
    except Exception:
        L = math.log(1.0 / delta)
        rho = (-math.sqrt(L) + math.sqrt(L + eps)) ** 2
    w = weights[weight_idx] / sum(weights)
    return math.sqrt(n_cliques / (2.0 * w * rho))


def generator_edges(dataset: str, generator: str, split: int) -> np.ndarray:
    """The target's own interior bin edges, in the cohort's gene order.

    Shape (n_genes, n_bins-1), the layout `digitize` takes.  Under `uniform`
    binning these are a public configuration and under `dp_quantile` a DP
    release, so an attacker may use them.  Under the legacy `quantile` binning
    they are the private percentiles -- an oracle, not an attack.
    """
    from mia import paths
    from mia.generators.pgm import PGMGenerator
    gen = PGMGenerator().load(paths.target_dir(dataset, generator, split)
                              / "generator.pt")._gen
    names = [f"gene_{i}" for i in range(len(D.gene_names(dataset)))]
    pos = {g: j for j, g in enumerate(names)}
    out = np.empty((len(names), gen._discretizer.n_bins - 1))
    for j, gene in enumerate(gen.selected_gene_names):
        out[pos[gene]] = gen._discretizer._edges[j][1:-1]
    return out


def arms_for_split(dataset: str, split: int, n_bins: int, eps: float, delta: float,
                   generator: str = "pgm", edges: str = "aux"):
    """Every aggregation arm's score vector for one split.

    `edges="aux"` bins at quantiles of the auxiliary pool (the shipped attack);
    `edges="generator"` bins with the target's own cells (`generator_edges`).
    """
    ncl = D.n_classes(dataset)
    Xr = D.load_expression(dataset).values.astype(np.float64)
    yr = D.encode_subtypes(dataset, D.load_subtypes(dataset).values)
    tg = T.load_target(dataset, generator, split)
    Xs, ys = tg["X"].astype(np.float64), tg["y_int"]

    if edges == "generator":
        edges = generator_edges(dataset, generator, split)
    else:   # auxiliary pool only -- never the target's training half
        edges = quantile_bin_edges(Xr, n_bins)
    br, bs = digitize(Xr, edges, n_bins), digitize(Xs, edges, n_bins)
    g = np.arange(br.shape[1])
    lab = yr[:, None]

    r1 = (np.maximum(_oneway_probs(bs, n_bins)[g, br], EPS)
          / np.maximum(_oneway_probs(br, n_bins)[g, br], EPS))
    r2 = (np.maximum(_twoway_probs(bs, ys, n_bins, ncl)[g, br, lab], EPS)
          / np.maximum(_twoway_probs(br, yr, n_bins, ncl)[g, br, lab], EPS))
    L1, L2 = np.log(r1), np.log(r2)
    n_genes, n_syn = br.shape[1], len(bs)

    p1 = 1.0 / n_bins
    pc = np.bincount(yr, minlength=ncl) / len(yr)
    p2 = (pc / n_bins).mean()
    w1 = 1.0 / (family_sigma(n_genes, 0, eps, delta) ** 2 + n_syn * p1 * (1 - p1))
    w2 = 1.0 / (family_sigma(n_genes, 1, eps, delta) ** 2 + n_syn * p2 * (1 - p2))

    s1, s2 = L1.mean(1), L2.mean(1)
    both = np.concatenate([L1, L2], axis=1).mean(1)
    return yr, {
        "ratio (shipped)": np.concatenate([r1, r2], axis=1).mean(1),
        "ratio, 1-way only": r1.mean(1),
        "log": both,
        "log, 1-way only": s1,
        "log, 2-way only": s2,
        "log, inverse-variance": (w1 * s1 + w2 * s2) / (w1 + w2),
        "log, class-centred": center_by_class(both, yr),
        "log, 1-way only, class-centred": center_by_class(s1, yr),
        "log, 2-way only, class-centred": center_by_class(s2, yr),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", default=["BRCA", "COMBINED"])
    p.add_argument("--splits", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    p.add_argument("--n_bins", type=int, default=4)
    p.add_argument("--epsilon", type=float, default=10.0)
    p.add_argument("--delta", type=float, default=1e-5)
    args = p.parse_args()

    for ds in args.datasets:
        acc: dict[str, list[dict]] = {}
        for split in args.splits:
            yr, arms = arms_for_split(ds, split, args.n_bins, args.epsilon, args.delta)
            m = D.membership_labels(ds, split).astype(int)
            for name, sc in arms.items():
                acc.setdefault(name, []).append(M.evaluate(m, sc))
        print(f"\n=== {ds} ===")
        ranked = sorted(acc.items(),
                        key=lambda kv: -np.mean([r["auc"] for r in kv[1]]))
        for name, per in ranked:
            auc = np.array([r["auc"] for r in per])
            t1 = np.mean([r["tpr_at_fpr_0.01"] for r in per])
            t10 = np.mean([r["tpr_at_fpr_0.1"] for r in per])
            print(f"  {name:34s} AUC={auc.mean():.4f}+-{auc.std(ddof=1):.4f} "
                  f"T@1={t1:.4f} T@10={t10:.4f}")
            csvlock.append_row(OUT, dict(
                dataset=ds, arm=name, auc=auc.mean(), auc_sd=auc.std(ddof=1),
                tpr_at_1=t1, tpr_at_10=t10, n_splits=len(per),
                n_bins=args.n_bins, epsilon=args.epsilon),
                key=["dataset", "arm", "n_bins", "epsilon"])


if __name__ == "__main__":
    main()
