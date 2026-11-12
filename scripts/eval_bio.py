#!/usr/bin/env python
"""Biological plausibility of the released synthetic datasets.

    python scripts/eval_bio.py --workers 8
    python scripts/eval_bio.py --datasets BRCA --generators mvn cvae --splits 1

The two measures of the CAMDA 2025 Health Privacy Challenge benchmark
(Health-Privacy-Challenge/src/evaluation/bio/{diffexpression,coexpression}.R),
re-implemented in Python because scran and hCoCena are not installed here.

Differential expression.  For every ordered pair of classes (a, b), a gene is
"up" in a if a one-sided Wilcoxon rank-sum test of a against b gives p < 0.05
(unadjusted, as in the benchmark).  With `lfc` > 0 the test asks for a shift
larger than `lfc` (class a's values are lowered by `lfc` first), which is what
scran::pairwiseWilcox(lfc=) does.  The synthetic data's DE genes are compared
with the real training data's, pair by pair: TPR (share of the real DE genes
recovered), FPR (share of the non-DE genes called DE) and precision, averaged
over pairs.  "Down" in (a, b) is "up" in (b, a), so ordered pairs cover both
directions.

Co-expression.  An edge is a pair of genes whose Pearson correlation over all
samples exceeds `cutoff`.  The synthetic data's edges are compared with the
real training data's: recall (share of real edges recovered) and precision
(share of synthetic edges that are real), plus the number of edges on each
side.  The benchmark builds the same edge lists inside hCoCena.

Reference row.  `real_heldout` scores the real held-out donors (the 20% of a
split) against the same training data.  It is a real dataset a quarter of the
size, so it shows what sampling alone costs, not a ceiling.

Writes one row per (dataset, generator, split, lfc, cutoff) to results/bio.csv.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import mannwhitneyu  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mia import datasets as D  # noqa: E402
from mia import targets as T  # noqa: E402

OUT = ROOT / "results" / "bio.csv"
TABSYN = ("tabsyn@class_freq=train,diff_schedule=steps,latent_scale=std,"
          "preprocess=clip:0.001:0.999+standard")
PGM = "pgm@binning=dp_quantile,edge_estimator=threshold,n_bins=16"
GENERATORS = ["mvn", "cvae", "nd", TABSYN, "tabpfn", "pgg", PGM]
P_VALUE = 0.05
MIN_CLASS = 5          # a class needs this many samples on both sides to be tested


def de_sets(X: np.ndarray, y: np.ndarray, classes, lfc: float) -> dict:
    """{(a, b): boolean mask over genes, True where the gene is up in a against b}."""
    out = {}
    for a, b in itertools.permutations(classes, 2):
        Xa, Xb = X[y == a], X[y == b]
        if len(Xa) < MIN_CLASS or len(Xb) < MIN_CLASS:
            continue
        p = mannwhitneyu(Xa - lfc, Xb, alternative="greater", axis=0).pvalue
        out[(a, b)] = p < P_VALUE
    return out


def de_scores(real: dict, syn: dict) -> dict:
    tpr, fpr, prec = [], [], []
    for pair, r in real.items():
        if pair not in syn:
            continue
        s = syn[pair]
        if r.sum():
            tpr.append((s & r).sum() / r.sum())
        if (~r).sum():
            fpr.append((s & ~r).sum() / (~r).sum())
        if s.sum():
            prec.append((s & r).sum() / s.sum())
    mean = lambda v: float(np.mean(v)) if v else float("nan")  # noqa: E731
    return {"de_tpr": mean(tpr), "de_fpr": mean(fpr), "de_precision": mean(prec),
            "de_pairs": len(tpr), "de_pairs_real": len(real),
            "de_real_share": float(np.mean([r.mean() for r in real.values()]))}


def edges(X: np.ndarray, cutoff: float) -> np.ndarray:
    C = np.corrcoef(X, rowvar=False)
    iu = np.triu_indices_from(C, k=1)
    return np.nan_to_num(C[iu]) > cutoff


def coex_scores(real: np.ndarray, syn: np.ndarray) -> dict:
    both = (real & syn).sum()
    return {"coex_recall": float(both / real.sum()) if real.sum() else float("nan"),
            "coex_precision": float(both / syn.sum()) if syn.sum() else float("nan"),
            "coex_edges_real": int(real.sum()), "coex_edges_syn": int(syn.sum())}


def one(job) -> list:
    ds, split, gens, lfcs, cutoffs = job
    X = D.load_expression(ds).values.astype(np.float64)
    y = np.asarray(D.encode_subtypes(ds, D.load_subtypes(ds).values))
    m = D.membership_labels(ds, split).astype(bool)
    classes = np.unique(y)
    sides = {"real_heldout": (X[~m], y[~m])}
    for g in gens:
        if T.exists(ds, g, split):
            t = T.load_target(ds, g, split)
            sides[g] = (t["X"].astype(np.float64), np.asarray(t["y_int"]))
    real_de = {lfc: de_sets(X[m], y[m], classes, lfc) for lfc in lfcs}
    real_edges = {c: edges(X[m], c) for c in cutoffs}
    rows = []
    for name, (Xs, ys) in sides.items():
        de = {lfc: de_scores(real_de[lfc], de_sets(Xs, ys, classes, lfc)) for lfc in lfcs}
        co = {c: coex_scores(real_edges[c], edges(Xs, c)) for c in cutoffs}
        for lfc, c in itertools.product(lfcs, cutoffs):
            rows.append(dict(dataset=ds, generator=name, split=split, n=len(Xs),
                             lfc=lfc, cutoff=c, **de[lfc], **co[c]))
    print(f"  {ds} split {split}: {len(sides)} datasets scored", flush=True)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", default=["BRCA", "COMBINED", "GTEX"])
    p.add_argument("--generators", nargs="+", default=GENERATORS)
    p.add_argument("--splits", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    p.add_argument("--lfc", nargs="+", type=float, default=[0.0, 1.0])
    p.add_argument("--cutoff", nargs="+", type=float, default=[0.5, 0.7])
    p.add_argument("--workers", type=int, default=5)
    args = p.parse_args()

    jobs = [(ds, s, args.generators, args.lfc, args.cutoff)
            for ds in args.datasets for s in args.splits]
    rows = []
    with ProcessPoolExecutor(args.workers) as pool:
        for r in pool.map(one, jobs):
            rows += r
    new = pd.DataFrame(rows)
    key = ["dataset", "generator", "split", "lfc", "cutoff"]
    if OUT.exists():
        old = pd.read_csv(OUT)
        new = pd.concat([old, new]).drop_duplicates(key, keep="last")
    new.sort_values(key).to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(new)} rows)")


if __name__ == "__main__":
    main()
