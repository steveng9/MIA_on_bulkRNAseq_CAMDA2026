"""The challenge's SDG-agnostic baseline attacks, ported for the internal grid.

    generic(method="mc" | "gan_leaks" | "conf_lr" | "conf_rf"
                   | "logan_d1" | "gan_leaks_cal" | "domias_kde")

These are the "MC / GAN-leaks / Conf-LR / Conf-RF / LOGAN / GAN-leaks-cal /
DOMIAS-KDE" rows of the CAMDA-26 abstract's Table 1.  The code follows the
submitted red-team repo (`src/mia/models/baseline.py`, itself adapted from
DOMIAS) line for line, including its preprocessing: the synthetic data, the
candidates and the reference set are each standardised with their OWN
StandardScaler (`MIADataLoader`), and the reference-based methods reduce to
150 PCA components fitted separately per set.  Porting them lets the abstract's
baseline rows be filled in for targets the challenge never released (the fixed
DP-PGM), and checks that our retrained targets behave like the team's.

The three reference-based methods need an auxiliary non-member set, which only
TCGA-COMBINED ships; on BRCA they raise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .. import datasets as D
from .. import targets as T
from .base import Attack, register

METHODS = ("mc", "gan_leaks", "conf_lr", "conf_rf", "logan_d1", "gan_leaks_cal",
           "domias_kde")
NEEDS_REFERENCE = ("logan_d1", "gan_leaks_cal", "domias_kde")


def _sq_dists(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances, (len(X), len(Y)), in row blocks."""
    yy = (Y ** 2).sum(1)
    out = np.empty((len(X), len(Y)))
    for i in range(0, len(X), 512):
        x = X[i:i + 512]
        out[i:i + 512] = np.maximum((x ** 2).sum(1)[:, None] + yy[None, :] - 2 * x @ Y.T, 0)
    return out


def mc(X_test: np.ndarray, X_G: np.ndarray) -> np.ndarray:
    """Monte-Carlo set membership (Hilprecht et al.); eps = 10th pct of min distance."""
    dist = _sq_dists(X_test, X_G)
    eps = np.percentile(dist.min(1), 10)
    return (dist < eps).sum(1) / X_G.shape[0]


def gan_leaks(X_test: np.ndarray, X_G: np.ndarray) -> np.ndarray:
    """The repo's GAN_leaks_modified: exp(-d_min / median d_min)."""
    dmin = _sq_dists(X_test, X_G).min(1)
    return np.exp(-dmin / np.median(dmin))


def confidence(X_test, X_G, y_G, model_type: str, seed: int = 42) -> np.ndarray:
    if model_type == "lr":
        clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=seed)
    else:
        clf = RandomForestClassifier(n_estimators=200, max_depth=10, n_jobs=-1,
                                     random_state=seed)
    clf.fit(X_G, y_G)
    return clf.predict_proba(X_test).max(1)


def gan_leaks_cal(X_test, X_G, X_ref) -> np.ndarray:
    """DOMIAS's GAN_leaks_cal: sigmoid(-(d_min(x, G) - d_min(x, ref)))."""
    z = -(_sq_dists(X_test, X_G).min(1) - _sq_dists(X_test, X_ref).min(1))
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def domias_kde(X_test, X_G, X_ref) -> np.ndarray:
    from scipy import stats
    pca = lambda Z: PCA(n_components=150).fit_transform(Z)  # noqa: E731  (as submitted)
    t, g, r = pca(X_test), pca(X_G), pca(X_ref)
    pg = stats.gaussian_kde(g.T)(t.T)
    pr = stats.gaussian_kde(r.T)(t.T)
    return pg / (pr + 1e-10)


@dataclass
class GenericBaseline(Attack):
    method: str = "gan_leaks"

    name = "generic"

    def tag(self) -> str:
        return self.method

    def score(self, dataset: str, generator: str, split: int) -> np.ndarray:
        if self.method not in METHODS:
            raise ValueError(f"unknown method {self.method!r}")
        std = lambda Z: StandardScaler().fit_transform(Z)  # noqa: E731  (per set, as submitted)
        X = std(D.load_expression(dataset).values.astype(np.float64))
        tg = T.load_target(dataset, generator, split)
        Xs = std(np.asarray(tg["X"], dtype=np.float64))
        ys = np.asarray(tg["y_int"]).astype(np.int64)
        if self.method == "mc":
            return mc(X, Xs)
        if self.method == "gan_leaks":
            return gan_leaks(X, Xs)
        if self.method in ("conf_lr", "conf_rf"):
            return confidence(X, Xs, ys, self.method[-2:], self.seed)
        ref = D.load_reference(dataset)
        if ref is None:
            raise ValueError(f"{self.method} needs a reference set; {dataset} has none")
        Xr = std(ref.values.astype(np.float64))
        if self.method == "gan_leaks_cal":
            return gan_leaks_cal(X, Xs, Xr)
        if self.method == "domias_kde":
            return domias_kde(X, Xs, Xr)
        import torch
        from domias.baselines import LOGAN_D1
        torch.manual_seed(self.seed)
        return np.asarray(LOGAN_D1(X, Xs, Xr), dtype=float)


register(GenericBaseline)
