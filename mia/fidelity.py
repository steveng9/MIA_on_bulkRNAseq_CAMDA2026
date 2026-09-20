"""How good is a synthetic dataset, as data?

The attack side of this repo asks whether synthetic data leaks.  This module
asks the complementary question -- whether it is worth releasing at all -- and
exists because tuning a generator "for the best quality we can get" is
meaningless without a number to tune against.

Four families, deliberately cheap enough to sit inside a hyperparameter sweep:

  utility        train a classifier on synthetic, test on real (TSTR).  The
                 headline number, because subtype prediction is what the
                 challenge's downstream task actually is.  Reported against a
                 train-on-real ceiling so a low score can be read as "the task
                 is hard" or "the synthetic data is bad", not both at once.

  marginals      mean 1-Wasserstein distance per gene, on z-scored genes so the
                 average is not dominated by whichever gene has the widest
                 dynamic range.  This is the part a marginal-based generator
                 like DP-PGM is directly optimising, so it flatters PGM and
                 should be read alongside the next one.

  dependence     mean absolute difference between the real and synthetic
                 gene-gene Spearman matrices, over a variance-ranked gene
                 subset.  This is the part a 1-way-marginal-only PGM cannot
                 represent even in principle, and it is where the fidelity cost
                 of `n_2way=0` shows up.

  distinguish    AUC of a gradient-boosted classifier asked to tell real rows
                 from synthetic ones.  0.5 means indistinguishable; 1.0 means
                 the two sets do not overlap.  One number, no distributional
                 assumptions, and it catches failure modes the other three miss
                 -- notably support violations, which is how the NoisyDiffusion
                 quantile leak was first visible.

The utility and distinguishability metrics both hold out real data the
generator never saw, because a generator that has memorised its training split
would otherwise score perfectly on both.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold


def _subset(n_genes: int, X_real: np.ndarray, max_genes: int) -> np.ndarray:
    """Variance-ranked gene indices, so the O(g^2) metrics stay affordable."""
    if n_genes <= max_genes:
        return np.arange(n_genes)
    return np.argsort(X_real.var(axis=0))[::-1][:max_genes]


def utility(X_syn, y_syn, X_test, y_test, X_train=None, y_train=None,
            seed: int = 0) -> dict:
    """Train-on-synthetic / test-on-real, against a train-on-real ceiling.

    `X_test` must be data the generator never saw -- the non-member half of the
    split -- or a generator that memorised its training set would score
    perfectly here as well as on the membership attacks.

    The ceiling trains on `X_train`, the same real rows the generator was fitted
    to, which is the fair comparison: it asks what the synthetic data costs
    relative to just releasing the original.  With no `X_train` the ceiling is
    cross-validated on the test half instead, which answers the easier question
    of how hard the task is at all.
    """
    def fit_score(Xtr, ytr, Xte, yte):
        if len(np.unique(ytr)) < 2:
            return {"accuracy": float("nan"), "macro_f1": float("nan")}
        clf = HistGradientBoostingClassifier(max_iter=150, random_state=seed)
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)
        return {"accuracy": float(accuracy_score(yte, pred)),
                "macro_f1": float(f1_score(yte, pred, average="macro"))}

    tstr = fit_score(X_syn, y_syn, X_test, y_test)
    if X_train is not None:
        real = fit_score(X_train, y_train, X_test, y_test)
    else:
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        rows = [fit_score(X_test[tr], y_test[tr], X_test[te], y_test[te])
                for tr, te in skf.split(X_test, y_test)]
        real = {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}

    out = {f"tstr_{k}": v for k, v in tstr.items()}
    out.update({f"real_{k}": v for k, v in real.items()})
    out["utility_ratio"] = (out["tstr_macro_f1"] / out["real_macro_f1"]
                            if out["real_macro_f1"] > 0 else float("nan"))
    return out


def marginals(X_syn: np.ndarray, X_real: np.ndarray) -> dict:
    """Mean per-gene Wasserstein distance, in units of the real gene's SD."""
    sd = X_real.std(axis=0)
    sd[sd < 1e-12] = 1.0
    d = [stats.wasserstein_distance(X_real[:, j] / sd[j], X_syn[:, j] / sd[j])
         for j in range(X_real.shape[1])]
    return {"wasserstein_mean": float(np.mean(d)),
            "wasserstein_p90": float(np.percentile(d, 90))}


def dependence(X_syn: np.ndarray, X_real: np.ndarray, max_genes: int = 300) -> dict:
    """Mean |Spearman_real - Spearman_syn| over a variance-ranked subset."""
    idx = _subset(X_real.shape[1], X_real, max_genes)
    Cr = stats.spearmanr(X_real[:, idx]).correlation
    Cs = stats.spearmanr(X_syn[:, idx]).correlation
    Cr, Cs = np.nan_to_num(Cr), np.nan_to_num(Cs)
    iu = np.triu_indices_from(Cr, k=1)
    diff = np.abs(Cr[iu] - Cs[iu])
    return {"corr_mae": float(diff.mean()),
            "corr_frobenius": float(np.linalg.norm(Cr - Cs) / Cr.shape[0]),
            "corr_real_mean_abs": float(np.abs(Cr[iu]).mean()),
            "corr_syn_mean_abs": float(np.abs(Cs[iu]).mean())}


def distinguishability(X_syn: np.ndarray, X_real: np.ndarray,
                       seed: int = 0, max_genes: int = 300) -> dict:
    """AUC of a real-vs-synthetic discriminator.  0.5 is perfect synthesis."""
    idx = _subset(X_real.shape[1], X_real, max_genes)
    X = np.vstack([X_real[:, idx], X_syn[:, idx]])
    y = np.r_[np.zeros(len(X_real)), np.ones(len(X_syn))]
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in skf.split(X, y):
        clf = HistGradientBoostingClassifier(max_iter=150, random_state=seed)
        clf.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    return {"discriminator_auc": float(np.mean(aucs))}


def evaluate(X_syn, y_syn, X_test, y_test, X_train=None, y_train=None, *,
             seed: int = 0, max_genes: int = 300) -> dict:
    """Every metric, as one flat dict ready for a results row.

    The distributional metrics compare against `X_train` when it is given --
    that is the distribution the generator was actually asked to reproduce --
    and fall back to the held-out half otherwise.
    """
    X_syn = np.asarray(X_syn, dtype=np.float64)
    X_test = np.asarray(X_test, dtype=np.float64)
    X_real = np.asarray(X_train, dtype=np.float64) if X_train is not None else X_test
    out = {}
    out.update(utility(X_syn, y_syn, X_test, y_test, X_train, y_train, seed=seed))
    out.update(marginals(X_syn, X_real))
    out.update(dependence(X_syn, X_real, max_genes=max_genes))
    out.update(distinguishability(X_syn, X_real, seed=seed, max_genes=max_genes))
    return out
