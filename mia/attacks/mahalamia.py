"""MahalaMIA -- Mahalanobis-distance membership inference.

Aimed at the multivariate-normal generator, whose released data is literally a
draw from a per-class Gaussian fitted to the training split.  The synthetic
set's mean and covariance are therefore a direct, low-noise imprint of the
members, and a candidate's Mahalanobis distance to that distribution,

    d_syn(x) = sqrt( (x - mu_syn)^T Sigma_syn^-1 (x - mu_syn) ),

separates members from non-members without any shadow modelling at all.  Small
distance means the synthetic distribution explains the sample well, so the score
is inverted.

With an auxiliary set of known non-members (TCGA-COMBINED only) the raw distance
is replaced by the ratio d_aux / d_syn.  That calibration matters because
d_syn(x) also reflects how unusual x is in general: dividing by the distance to
a member-free reference distribution cancels the part of the distance that is
about the sample rather than about membership.

Sigma_syn is 978x978 estimated from ~870 samples, so it is rank-deficient and
badly conditioned.  Three ways of handling that are available:

  pinv          the pseudo-inverse, which restricts the quadratic form to the
                span the synthetic data actually covers (default, and what the
                CAMDA submission used)
  ledoit_wolf   Ledoit-Wolf shrinkage toward a scaled identity, which trades a
                little bias for a well-conditioned, invertible estimate
  ridge         a fixed ridge on the diagonal

`n_components` additionally projects onto the leading PCA directions of the
*synthetic* data first.  Fitting the basis on the synthetic set keeps the threat
model honest -- the adversary holds it -- and it also puts the covariance
estimate back in a regime where the sample count exceeds the dimension.

Reported in the abstract as AUC 0.922 (BRCA) / 0.899 (COMBINED) against MVN.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.linalg import pinv
from scipy import stats
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA

from .. import datasets as D
from .. import targets as T
from .base import Attack, register


def nearest_psd(cov: np.ndarray) -> np.ndarray:
    cov = (cov + cov.T) / 2
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 0)
    cov = eigvecs @ np.diag(eigvals) @ eigvecs.T
    min_eig = float(np.min(np.real(np.linalg.eigvals(cov))))
    if min_eig < 0:
        cov -= 10 * min_eig * np.eye(*cov.shape)
    return cov


def sigmoid_calibrate(raw: np.ndarray, confidence: float = 1.0,
                      center_pct: float = 20.0) -> np.ndarray:
    """Map raw scores onto (0, 1) without changing their order.

    Log, z-score, then a logistic centred at the `center_pct` percentile, which
    is where the member/non-member boundary sits under the challenge's 80/20
    split.  Rank-preserving, so AUC is untouched; it only makes the scores
    readable as probabilities and comparable across targets.
    """
    raw = np.asarray(raw, dtype=float)
    z = stats.zscore(np.log(np.maximum(raw, 1e-300)))
    return 1.0 / (1.0 + np.exp(-confidence * (z - np.percentile(z, center_pct))))


def mahalanobis(X: np.ndarray, mean: np.ndarray, inv_cov: np.ndarray) -> np.ndarray:
    delta = X - mean
    q = np.einsum("ij,jk,ik->i", delta, inv_cov, delta)
    return np.sqrt(np.maximum(q, 0.0))


def precision(X: np.ndarray, method: str, ridge_alpha: float) -> np.ndarray:
    """Inverse covariance of `X` under the chosen conditioning strategy."""
    if method == "ledoit_wolf":
        return LedoitWolf(assume_centered=False).fit(X).get_precision()
    cov = nearest_psd(np.cov(X, rowvar=False))
    if method == "ridge":
        cov = cov + ridge_alpha * np.trace(cov) / cov.shape[0] * np.eye(cov.shape[0])
        return np.linalg.inv(cov)
    if method == "pinv":
        return pinv(cov)
    raise ValueError(f"Unknown covariance method {method!r}")


@dataclass
class MahalaMIA(Attack):
    use_reference: bool = True     # ignored for cohorts without an auxiliary set
    covariance: str = "pinv"       # "pinv" | "ledoit_wolf" | "ridge"
    ridge_alpha: float = 1e-3
    n_components: int | None = None    # PCA dimension, fitted on the synthetic set
    calibrate: bool = True

    name = "mahalamia"

    def tag(self) -> str:
        t = "aux" if self.use_reference else "noaux"
        if self.covariance == "ridge":
            t += f"_ridge{self.ridge_alpha:g}"
        elif self.covariance != "pinv":
            t += f"_{self.covariance}"
        if self.n_components:
            t += f"_pca{self.n_components}"
        return t

    def score(self, dataset: str, generator: str, split: int) -> np.ndarray:
        X_real = D.load_expression(dataset).values.astype(np.float64)
        X_syn = T.load_target(dataset, generator, split)["X"].astype(np.float64)
        ref = D.load_reference(dataset) if self.use_reference else None
        X_aux = ref.values.astype(np.float64) if ref is not None else None

        if self.n_components:
            # Basis fitted on the synthetic data only: that is what the
            # adversary holds, and it keeps the projection from seeing D_real.
            pca = PCA(n_components=min(self.n_components, *X_syn.shape)).fit(X_syn)
            X_real, X_syn = pca.transform(X_real), pca.transform(X_syn)
            if X_aux is not None:
                X_aux = pca.transform(X_aux)

        d_syn = mahalanobis(X_real, X_syn.mean(axis=0),
                            precision(X_syn, self.covariance, self.ridge_alpha))

        if X_aux is None:
            raw = 1.0 / (d_syn + 1e-10)
        else:
            d_aux = mahalanobis(X_real, X_aux.mean(axis=0),
                                precision(X_aux, self.covariance, self.ridge_alpha))
            raw = d_aux / (d_syn + d_aux + 1e-10)

        raw = np.nan_to_num(raw, nan=float(np.nanmedian(raw)))
        return sigmoid_calibrate(raw) if self.calibrate else raw


register(MahalaMIA)
