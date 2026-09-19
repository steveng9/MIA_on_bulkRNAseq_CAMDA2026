#!/usr/bin/env python
"""Is the MVN generator's exposure a p-approximately-n effect?

MahalaMIA with a ridge-conditioned covariance reaches AUC 1.000 against the MVN
generator on BRCA (871 training samples, 978 genes) and gains nothing at all on
COMBINED (3,458 training samples, same 978 genes).  Two explanations fit:
the cohorts differ in sample size, or they differ in something else -- number of
classes, tissue heterogeneity, how well a Gaussian fits at all.

This holds the cohort fixed and varies only n.  One cohort, one generator, one
attack; the training set is resampled to each size and everything else is
identical, so any trend is attributable to n/p alone.  If the p-approximately-n
story is right, the AUC should climb as n falls through 978 and the ridge
variant should separate from the pseudo-inverse only below that point.

    python scripts/cohort_size_sweep.py --dataset COMBINED
    python scripts/cohort_size_sweep.py --sizes 500 1000 2000 --trials 5

Results land in results/index.csv under experiment "cohort_size", so they are
comparable to every other run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mia import datasets as D            # noqa: E402
from mia import metrics as M             # noqa: E402
from mia import runs as R                # noqa: E402
from mia.attacks.mahalamia import (mahalanobis, precision,  # noqa: E402
                                   sigmoid_calibrate)
from mia.generators.mvn import MVNGenerator  # noqa: E402

DEFAULT_SIZES = (500, 700, 871, 1100, 1500, 2000, 2800, 3458)
# Both ends of the alpha range: on BRCA (n < p) the small alpha wins and the
# large one loses, and on COMBINED (n > p) that ordering inverts.  Carrying
# both through the sweep shows where the crossover actually is.
COVARIANCES = (("pinv", 0.0), ("ridge", 1e-6), ("ridge", 1e-2))


def stratified_sample(y: np.ndarray, n: int, rng: np.random.RandomState) -> np.ndarray:
    """n indices keeping class proportions, with at least 2 per present class.

    A per-class Gaussian needs two samples to have a covariance at all, so a
    class that would round down to one is given two; the surplus comes off the
    largest classes.  Without this the smallest subtypes drop out entirely at
    small n and the sweep would be confounding class count with sample count.
    """
    classes, counts = np.unique(y, return_counts=True)
    quota = np.maximum(2, np.round(counts / counts.sum() * n).astype(int))
    quota = np.minimum(quota, counts)

    # Reconcile to exactly n by trimming the largest classes first.
    while quota.sum() > n:
        order = np.argsort(-quota)
        for c in order:
            if quota[c] > 2:
                quota[c] -= 1
                if quota.sum() == n:
                    break
        else:
            break                        # every class already at its floor

    picked = []
    for c, q in zip(classes, quota):
        idx = np.where(y == c)[0]
        picked.append(rng.choice(idx, size=min(q, len(idx)), replace=False))
    return np.concatenate(picked)


def score(X_real, X_syn, X_aux, covariance, ridge_alpha, aux_cache):
    """MahalaMIA's score, composed from the attack module's own helpers."""
    d_syn = mahalanobis(X_real, X_syn.mean(axis=0),
                        precision(X_syn, covariance, ridge_alpha))
    if X_aux is None:
        raw = 1.0 / (d_syn + 1e-10)
    else:
        key = (covariance, ridge_alpha)
        if key not in aux_cache:         # the auxiliary set never changes
            aux_cache[key] = (X_aux.mean(axis=0),
                              precision(X_aux, covariance, ridge_alpha))
        mu_aux, prec_aux = aux_cache[key]
        d_aux = mahalanobis(X_real, mu_aux, prec_aux)
        raw = d_aux / (d_syn + d_aux + 1e-10)
    raw = np.nan_to_num(raw, nan=float(np.nanmedian(raw)))
    return sigmoid_calibrate(raw)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="COMBINED")
    p.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES))
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--noise-level", type=float, default=0.7)
    p.add_argument("--no-save", action="store_true")
    args = p.parse_args()

    df = D.load_expression(args.dataset)
    ids = np.array([str(i) for i in df.index], dtype=object)
    X_all = df.values.astype(np.float64)
    y_all = D.encode_subtypes(args.dataset, D.load_subtypes(args.dataset))
    n_classes = D.n_classes(args.dataset)
    ref = D.load_reference(args.dataset)
    X_aux = ref.values.astype(np.float64) if ref is not None else None

    p_genes = X_all.shape[1]
    print(f"{args.dataset}: {len(X_all)} samples x {p_genes} genes, "
          f"{n_classes} classes, aux={'none' if X_aux is None else len(X_aux)}")
    print(f"p = {p_genes}; sizes crossing it: "
          f"{[n for n in args.sizes if n < p_genes]} below, "
          f"{[n for n in args.sizes if n >= p_genes]} at or above\n")

    aux_cache: dict = {}
    print(f"{'n_train':>8} {'n/p':>6} {'covariance':>12} {'AUC':>7} {'T@10':>7}")
    for n in args.sizes:
        if n > len(X_all):
            print(f"  [skip] n={n} exceeds the cohort ({len(X_all)})")
            continue

        # Fit once per (n, trial) and score every covariance setting off the same
        # synthetic data.  The generator fit is 12 per-class eigendecompositions
        # of a 978x978 matrix and dominates the cost; the covariance variants
        # differ only in how the attack inverts what it is given, so refitting
        # per variant would treble the work and also add noise between the rows
        # we most want to compare.
        results: dict = {}
        for trial in range(1, args.trials + 1):
            rng = np.random.RandomState(10_000 * trial + n)
            train_idx = stratified_sample(y_all, n, rng)
            y_member = np.zeros(len(X_all), dtype=int)
            y_member[train_idx] = 1

            gen = MVNGenerator(noise_level=args.noise_level,
                               seed=int(rng.randint(1 << 30)),
                               device="cpu", verbose=False)
            gen.fit(X_all[train_idx].astype(np.float32), y_all[train_idx], n_classes)
            X_syn = gen.sample(len(train_idx))[0].astype(np.float64)

            for cov, alpha in COVARIANCES:
                sc = score(X_all, X_syn, X_aux, cov, alpha, aux_cache)
                met = M.evaluate(y_member, sc)
                results.setdefault((cov, alpha), []).append(met)

                if not args.no_save:
                    tag = f"n{n}_{cov}" + (f"{alpha:g}" if cov == "ridge" else "")
                    R.save_run(
                        dataset=args.dataset, attack="mahalamia", generator="mvn",
                        split=trial, tag=tag, experiment="cohort_size",
                        variant=tag,
                        notes="cohort-size sweep at fixed gene count",
                        params={"attack": "mahalamia", "covariance": cov,
                                "ridge_alpha": alpha, "use_reference": True,
                                "n_train": n, "n_genes": p_genes,
                                "noise_level": args.noise_level},
                        metrics=met, sample_ids=ids, scores=sc,
                        y_member=y_member,
                    )

        for (cov, alpha), mets in results.items():
            label = cov if cov == "pinv" else f"{cov} {alpha:g}"
            print(f"{n:>8} {n / p_genes:>6.2f} {label:>12} "
                  f"{np.mean([m['auc'] for m in mets]):>7.4f} "
                  f"{np.mean([m['tpr_at_fpr_0.1'] for m in mets]):>7.4f}", flush=True)
        print(flush=True)


if __name__ == "__main__":
    main()
