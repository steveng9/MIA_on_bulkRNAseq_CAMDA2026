#!/usr/bin/env python
"""Evidence for the per-gene quantile-support leak in released synthetic data.

    python scripts/analyse_quantile_leak.py --dataset BRCA
    python scripts/analyse_quantile_leak.py --dataset BRCA --generators nd mvn

Why this exists: MAMA-MIA at very fine bin resolution reaches AUC ~1.0 against
NoisyDiffusion while staying at chance against every other generator, which is
surprising enough that it needs a mechanism rather than a number.

The mechanism is the generator's output pipeline, not its model.  NoisyDiffusion
normalises with a `QuantileTransformer` fitted on the training split and inverts
that transform to produce its released samples.  With `n_quantiles` at least the
training size, the learned quantiles *are* the sorted training values, so every
synthetic value is an interpolation between two adjacent values that a training
member actually had.  The released data therefore carries the training set's
per-gene empirical support almost exactly, and a candidate can be tested against
it one gene at a time.

This prints four pieces of evidence:

  1. how much closer a member's value sits to the nearest synthetic value than a
     non-member's, per gene and aggregated;
  2. how often the *extremes* of the synthetic range coincide exactly with the
     member extremes, versus the non-member ones;
  3. how often a real value falls within floating-point distance of a synthetic
     value, split by membership;
  4. the negative controls, so the aggregate is not mistaken for a bug.

Any generator that inverts an empirical quantile map fitted on its training data
inherits this, independently of how private the model itself is.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mia import datasets as D  # noqa: E402
from mia import metrics as M  # noqa: E402
from mia import targets as T  # noqa: E402


def nearest_value_gaps(X: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Per-gene distance from each real value to the nearest synthetic value."""
    gaps = np.empty_like(X)
    for g in range(X.shape[1]):
        col = np.sort(S[:, g])
        idx = np.clip(np.searchsorted(col, X[:, g]), 1, len(col) - 1)
        gaps[:, g] = np.minimum(np.abs(X[:, g] - col[idx - 1]),
                                np.abs(X[:, g] - col[idx]))
    return gaps


def aggregate_score(gaps: np.ndarray) -> np.ndarray:
    """Standardise each gene, then average -- weak per-gene evidence, pooled."""
    z = (gaps - gaps.mean(0)) / (gaps.std(0) + 1e-12)
    return -z.mean(axis=1)


def report(dataset: str, generator: str, split: int, atol: float = 1e-5) -> None:
    X = D.load_expression(dataset).values.astype(np.float64)
    y = D.membership_labels(dataset, split)
    try:
        S = T.load_target(dataset, generator, split)["X"].astype(np.float64)
    except FileNotFoundError:
        print(f"\n{generator}: target for split {split} not built, skipping")
        return

    n_genes = X.shape[1]
    mem, non = X[y == 1], X[y == 0]
    gaps = nearest_value_gaps(X, S)
    score = aggregate_score(gaps)

    single = np.array([M.evaluate(y, -gaps[:, g])["auc"]
                       for g in range(0, n_genes, 25)])
    exact = {}
    for label, sub in (("member", mem), ("non-member", non)):
        hits = []
        for g in range(0, n_genes, 20):
            col = np.unique(S[:, g])
            idx = np.clip(np.searchsorted(col, sub[:, g]), 1, len(col) - 1)
            gap = np.minimum(np.abs(sub[:, g] - col[idx - 1]),
                             np.abs(sub[:, g] - col[idx]))
            hits.append((gap < atol).mean())
        exact[label] = float(np.mean(hits))

    min_hits_mem = sum(abs(S[:, g].min() - mem[:, g].min()) < 1e-4
                       for g in range(n_genes))
    min_hits_non = sum(abs(S[:, g].min() - non[:, g].min()) < 1e-4
                       for g in range(n_genes))

    rng = np.random.default_rng(0)
    shuffled = float(np.mean([M.evaluate(rng.permutation(y), score)["auc"]
                              for _ in range(20)]))
    other = next(s for s in D.load_target_splits(dataset) if s != split)
    cross = M.evaluate(D.membership_labels(dataset, other), score)["auc"]

    print(f"\n── {generator} (split {split}) " + "─" * 46)
    print(f"  single-gene AUC, mean over every 25th gene : {single.mean():.4f} "
          f"(max {single.max():.4f})")
    print(f"  aggregated over all {n_genes} genes             : "
          f"{M.evaluate(y, score)['auc']:.4f}")
    print(f"  min(synthetic) == min(member) for           : "
          f"{min_hits_mem}/{n_genes} genes")
    print(f"  min(synthetic) == min(non-member) for       : "
          f"{min_hits_non}/{n_genes} genes")
    print(f"  values within {atol:g} of a synthetic value    : "
          f"members {exact['member']:.4f}  non-members {exact['non-member']:.4f}")
    print(f"  [control] permuted labels                   : {shuffled:.4f}")
    print(f"  [control] split {other} labels                    : {cross:.4f}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="BRCA")
    p.add_argument("--generators", nargs="+", default=list(T.GENERATORS))
    p.add_argument("--split", type=int, default=1)
    args = p.parse_args()

    print(f"Per-gene quantile-support leak — {args.dataset}, split {args.split}")
    print("Higher single-gene AUC means the released values sit measurably "
          "closer to members.")
    for gen in args.generators:
        report(args.dataset, gen, args.split)


if __name__ == "__main__":
    main()
