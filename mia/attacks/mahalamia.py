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


Conditioning
------------

Sigma_syn is 978x978 estimated from ~870 samples, so it is rank-deficient and
badly conditioned.  Four ways of handling that are available:

  pinv          the pseudo-inverse, which restricts the quadratic form to the
                span the synthetic data actually covers (what the CAMDA
                submission used)
  ledoit_wolf   Ledoit-Wolf shrinkage toward a scaled identity
  ridge         a fixed ridge on the diagonal, `ridge_alpha` relative to the
                mean eigenvalue
  exact         a plain inverse, only valid once the space has been reduced
                below the sample count

The choice is not cosmetic.  On BRCA/MVN, `pinv` scores 0.928 AUC and a ridge of
1e-6 scores 1.000 with TPR@1%FPR also 1.000.  The leak lives in the directions
the pseudo-inverse throws away, and a whisker of ridge keeps them while stopping
them blowing up.


Projection
----------

`n_components` projects onto PCA directions before any of that happens, and
`drop_leading` discards the top few, so the two together select a *band* of the
spectrum.  Truncating and regularising do not commute with the distance:
Mahalanobis is invariant to any invertible linear map, so PCA on its own would
change nothing -- it is only because the regulariser is applied in the projected
basis that `n_components` has any effect at all.  The same is true of
`standardize`, which z-scores genes first: it makes the ridge isotropic in
correlation space rather than in covariance space.

`pca_basis` chooses whose covariance defines the directions.  Fitting on the
synthetic set keeps the threat model honest and is the default, but it also
means d_syn and d_aux are measured in a basis aligned to one of them.  Fitting
on the reference set, or on both pooled, puts the two distances on a common
footing -- which is what the ratio assumes.


Class conditioning
------------------

`class_conditional` is the axis the pooled attack was missing.  The MVN
generator fits *one Gaussian per class* and the CVAE conditions on the label, so
the released data is a mixture, not a single ellipsoid; pooling TCGA-COMBINED's
twelve tissues into one covariance estimates a between-tissue structure that no
generator ever fitted.  Scoring each candidate against its own class's synthetic
Gaussian matches the attack to the generator.

It costs sample count: BRCA-Normal has about 32 training members, so a per-class
covariance in 978 dimensions has rank 31.  Class conditioning and dimension
reduction therefore go together -- which is the most promising reason to expect
PCA to pay off here, rather than as a denoiser on the pooled estimate.

The reference set ships without labels, so when the ratio is in play the aux
side is pseudo-labelled by nearest synthetic class centroid.  That is something
the adversary can do unaided, and it keeps numerator and denominator in matched
per-class form.

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
    """Symmetrise and clip negative eigenvalues.

    The re-check uses `eigvalsh`, not `eigvals`: the matrix is symmetric by
    construction two lines earlier, and the general solver costs an order of
    magnitude more.  Class-conditional scoring calls this once per class per
    side, so on TCGA-COMBINED the difference is minutes, not milliseconds.
    """
    cov = (cov + cov.T) / 2
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 0)
    cov = eigvecs @ np.diag(eigvals) @ eigvecs.T
    min_eig = float(np.min(np.linalg.eigvalsh((cov + cov.T) / 2)))
    if min_eig < 0:
        cov -= 10 * min_eig * np.eye(*cov.shape)
    return cov


def sigmoid_calibrate(raw: np.ndarray, confidence: float = 1.0,
                      center_pct: float = 20.0,
                      log_transform: bool = True) -> np.ndarray:
    """Map raw scores onto (0, 1) without changing their order.

    Optionally log, then z-score, then a logistic centred at the `center_pct`
    percentile, which is where the member/non-member boundary sits under the
    challenge's 80/20 split.  Rank-preserving, so AUC is untouched; it only
    makes the scores readable as probabilities and comparable across targets.

    `log_transform` must be False for scores that are ALREADY in log space, or
    otherwise signed.  The log step clamps at 1e-300, so a negative score would
    be squashed onto the same value as every other negative score -- turning
    the ranking into one giant tie and collapsing AUC toward 0.5.  That is not
    hypothetical: it silently cost MAMA-MIA's log-ratio arm 12 points of AUC on
    BRCA (0.625 measured directly, 0.508 through this function) before the
    parameter existed.
    """
    raw = np.asarray(raw, dtype=float)
    if log_transform:
        if np.any(raw <= 0):
            raise ValueError(
                "sigmoid_calibrate(log_transform=True) needs strictly positive "
                "scores; got values <= 0.  Pass log_transform=False if the "
                "scores are already in log space.")
        raw = np.log(raw)
    z = stats.zscore(raw)
    return 1.0 / (1.0 + np.exp(-confidence * (z - np.percentile(z, center_pct))))


def mahalanobis(X: np.ndarray, mean: np.ndarray, inv_cov: np.ndarray) -> np.ndarray:
    """Row-wise distance to (mean, inv_cov).

    Written as a matmul rather than the equivalent
    `einsum("ij,jk,ik->i", delta, inv_cov, delta)`, which numpy evaluates
    without dispatching to BLAS: on a 4323x978 COMBINED cohort the einsum takes
    4.3 s and this takes 0.04 s, agreeing to 1e-15.  The sweep calls it once per
    class per side per target, so the difference is the difference between a
    17-hour sweep and a 20-minute one.
    """
    delta = X - mean
    q = ((delta @ inv_cov) * delta).sum(axis=1)
    return np.sqrt(np.maximum(q, 0.0))


def precision(X: np.ndarray, method: str, ridge_alpha: float) -> np.ndarray:
    """Inverse covariance of `X` under the chosen conditioning strategy."""
    if method == "ledoit_wolf":
        return LedoitWolf(assume_centered=False).fit(X).get_precision()
    cov = nearest_psd(np.cov(X, rowvar=False))
    if method == "ridge":
        cov = cov + ridge_alpha * np.trace(cov) / cov.shape[0] * np.eye(cov.shape[0])
        return np.linalg.inv(cov)
    if method == "exact":
        return np.linalg.inv(cov)
    if method == "pinv":
        try:
            return pinv(cov)
        except np.linalg.LinAlgError:
            # LAPACK's SVD occasionally fails to converge on near-singular
            # class covariances (small classes); the covariance is symmetric,
            # so the eigendecomposition gives the same pseudo-inverse.
            from scipy.linalg import pinvh
            return pinvh(cov)
    raise ValueError(f"Unknown covariance method {method!r}")


@dataclass
class ClassMoments:
    """What the paper's MahalaMIA reads from a distribution: per-class means and
    covariances, and the pooled mean and covariance.

    Black box, they are the sample moments of the release (`from_rows`).  White
    box, they are the moments of the target model itself (`from_mixture` when
    the model has them in closed form, `from_rows` on a large sample from it
    otherwise), so the only thing that changes between the two is how much
    sampling noise sits on top of what the generator learnt.
    """
    means: dict                 # class -> (p,)
    covs: dict                  # class -> (p, p); classes too small to estimate are absent
    pooled_mean: np.ndarray
    pooled_cov: np.ndarray      # already made PSD

    @classmethod
    def from_rows(cls, X: np.ndarray, y: np.ndarray, min_rows: int = 3) -> "ClassMoments":
        means, covs = {}, {}
        for c in np.unique(y):
            rows = X[y == c]
            means[int(c)] = rows.mean(axis=0)
            if len(rows) >= min_rows:
                covs[int(c)] = np.cov(rows, rowvar=False)
        return cls(means, covs, X.mean(axis=0), nearest_psd(np.cov(X, rowvar=False)))

    @classmethod
    def from_mixture(cls, components: dict) -> "ClassMoments":
        """Exact moments of a mixture {class: (mean, cov, weight)}.

        The pooled covariance is the law of total covariance: the weighted
        within-class covariances plus the spread of the class means.
        """
        w = np.array([components[c][2] for c in components], dtype=np.float64)
        w = w / w.sum()
        mu = {c: np.asarray(components[c][0], dtype=np.float64) for c in components}
        cov = {c: np.asarray(components[c][1], dtype=np.float64) for c in components}
        m = sum(wi * mu[c] for wi, c in zip(w, components))
        S = sum(wi * (cov[c] + np.outer(mu[c] - m, mu[c] - m)) for wi, c in zip(w, components))
        return cls(mu, cov, m, nearest_psd(S))


def class_shrunk_distance_from_moments(X_query, y_query, mom: ClassMoments, shrink: float,
                                       ridge_alpha: float) -> np.ndarray:
    """Distance of each query row to its own class, class covariance shrunk to the pooled one.

    The covariance used for class c is (1 - shrink) * S_c + shrink * S, where
    S_c is class c's covariance and S the pooled one, with the usual ridge on
    top (regularised discriminant analysis, Friedman 1989).  shrink = 1 is the
    pooled covariance around the class mean; shrink = 0 is the pure per-class
    distance.  The class term keeps the directions in which a class of the
    release does not vary, which is where a per-class generator such as MVN
    leaves its members; the pooled term makes a class of 30 rows usable and
    puts the classes on one scale.  Classes without a covariance (fewer than
    three rows), and query classes the distribution never emitted, fall back
    to the pooled distance.
    """
    S = mom.pooled_cov
    p = S.shape[0]

    def inv(C):
        return np.linalg.inv(C + ridge_alpha * np.trace(C) / p * np.eye(p))

    d = np.full(len(X_query), np.nan)
    for c, S_c in mom.covs.items():
        take = y_query == c
        if not take.any():
            continue
        C = (1.0 - shrink) * S_c + shrink * S
        d[take] = mahalanobis(X_query[take], mom.means[c], inv(C))
    missing = np.isnan(d)
    if missing.any():
        d[missing] = mahalanobis(X_query[missing], mom.pooled_mean, inv(S))
    return d


def class_shrunk_distance(X_query, y_query, X_fit, y_fit, shrink: float,
                          ridge_alpha: float) -> np.ndarray:
    """`class_shrunk_distance_from_moments` on the sample moments of (X_fit, y_fit)."""
    return class_shrunk_distance_from_moments(
        X_query, y_query, ClassMoments.from_rows(X_fit, y_fit), shrink, ridge_alpha)


def model_moments(gen, n_samples: int, seed: int) -> tuple[ClassMoments, str]:
    """A trained generator's moments: exact if it has them, else from a large sample.

    Returns the moments and how they were obtained ("exact" or "sampled:<n>").
    The sample is drawn with its own seed, independent of the release's.
    """
    try:
        return ClassMoments.from_mixture(gen.moments()), "exact"
    except NotImplementedError:
        pass
    import torch
    gen.seed = seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    X, y = gen.sample(n_samples)
    return ClassMoments.from_rows(np.asarray(X, dtype=np.float64), np.asarray(y)), \
        f"sampled:{n_samples}"


class Projection:
    """Gene space -> attack space: optional gene z-score, then a PCA band.

    Fitted on whatever `pca_basis` selects.  `drop_leading` removes the top
    components after the fit, so `n_components=800, drop_leading=50` keeps
    components 50..799 -- the band below the dominant tissue axes but above the
    numerical floor.
    """

    def __init__(self, X_fit: np.ndarray, n_components=None,
                 drop_leading: int = 0, standardize: bool = False):
        self.standardize = standardize
        if standardize:
            self.mu = X_fit.mean(axis=0)
            self.sd = X_fit.std(axis=0)
            self.sd[self.sd < 1e-12] = 1.0
        Z = self._scale(X_fit)

        self.pca = None
        self.lo = 0
        if n_components or drop_leading:
            k = min(n_components or Z.shape[1], *Z.shape)
            self.pca = PCA(n_components=k).fit(Z)
            self.lo = min(drop_leading, k - 1)

    def _scale(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mu) / self.sd if self.standardize else X

    def __call__(self, X: np.ndarray) -> np.ndarray:
        Z = self._scale(X)
        if self.pca is None:
            return Z
        return self.pca.transform(Z)[:, self.lo:]

    @property
    def dim(self) -> int:
        return (self.pca.n_components_ - self.lo) if self.pca else -1


@dataclass
class MahalaMIA(Attack):
    use_reference: bool = True     # ignored for cohorts without an auxiliary set
    covariance: str = "pinv"       # "pinv" | "ledoit_wolf" | "ridge" | "exact"
    ridge_alpha: float = 1e-3
    n_components: int | None = None    # PCA dimension
    drop_leading: int = 0              # discard this many leading components
    pca_basis: str = "synthetic"       # "synthetic" | "reference" | "pooled"
    standardize: bool = False          # z-score genes before PCA
    class_conditional: bool = False    # per-class mean and covariance
    class_shrink: float | None = None  # distance to the own class, class covariance shrunk
                                       # this far toward the pooled one (the paper's default: 0.5)
    pca_per_class: bool = False        # ...and a per-class PCA basis with it
    calibrate: bool = True
    #: White box: read the class moments from the target *model* instead of
    #: estimating them from its release -- exactly for a generator that has them
    #: in closed form (MVN), from `model_samples` fresh draws otherwise.  The
    #: reference side is unchanged.  Only the paper's form (`class_shrink`).
    white_box: bool = False
    model_samples: int = 100_000

    name = "mahalamia"

    def __post_init__(self):
        if self.white_box and (self.class_shrink is None or self.n_components
                               or self.drop_leading or self.standardize):
            raise ValueError("white-box MahalaMIA is defined for the paper's form only: "
                             "class_shrink set, no PCA band, no gene standardisation")

    def tag(self) -> str:
        t = "aux" if self.use_reference else "noaux"
        if self.covariance == "ridge":
            t += f"_ridge{self.ridge_alpha:g}"
        elif self.covariance != "pinv":
            t += f"_{self.covariance}"
        if self.n_components:
            t += f"_pca{self.n_components}"
        if self.drop_leading:
            t += f"_drop{self.drop_leading}"
        if self.pca_basis != "synthetic":
            t += f"_basis{self.pca_basis}"
        if self.standardize:
            t += "_std"
        if self.class_conditional:
            t += "_cc" + ("pc" if self.pca_per_class else "")
        if self.class_shrink is not None:
            t += f"_shrink{self.class_shrink:g}"
        if self.white_box:
            t += f"_whitebox{self.model_samples}"
        return t

    # ── Distance models ─────────────────────────────────────────────────────

    def _projection(self, X_syn, X_aux):
        """Fit the shared gene-space -> attack-space map."""
        if self.pca_basis == "reference" and X_aux is not None:
            X_fit = X_aux
        elif self.pca_basis == "pooled" and X_aux is not None:
            X_fit = np.vstack([X_syn, X_aux])
        else:
            X_fit = X_syn
        return Projection(X_fit, self.n_components, self.drop_leading,
                          self.standardize)

    def _pooled_distance(self, X_query, X_fit):
        return mahalanobis(X_query, X_fit.mean(axis=0),
                           precision(X_fit, self.covariance, self.ridge_alpha))

    def _conditional_distance(self, X_query, y_query, X_fit, y_fit):
        """Distance of each query row to its *own* class's Gaussian.

        Classes the generator never emitted, or emitted too few times to
        estimate anything from, fall back to the pooled fit -- silently, because
        the alternative is a NaN column in the middle of a grid.
        """
        d = np.full(len(X_query), np.nan)
        for c in np.unique(y_fit):
            rows = X_fit[y_fit == c]
            if len(rows) < 3:
                continue
            take = y_query == c
            if not take.any():
                continue
            if self.pca_per_class:
                proj = Projection(rows, self.n_components, self.drop_leading,
                                  self.standardize)
                d[take] = self._pooled_distance(proj(X_query[take]), proj(rows))
            else:
                d[take] = self._pooled_distance(X_query[take], rows)

        missing = np.isnan(d)
        if missing.any():
            d[missing] = self._pooled_distance(X_query[missing], X_fit)
        return d

    @staticmethod
    def _pseudo_label(X, X_ref, y_ref):
        """Nearest-class-centroid labels, for the unlabelled reference set."""
        cls = np.unique(y_ref)
        centroids = np.stack([X_ref[y_ref == c].mean(axis=0) for c in cls])
        dist = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        return cls[dist.argmin(axis=1)]

    # ── Scoring ─────────────────────────────────────────────────────────────

    def _target_moments(self, dataset: str, generator: str, split: int) -> ClassMoments:
        """White box: the moments of the target model (see `model_moments`)."""
        gen = T.load_model(dataset, generator, split, device=self.device)
        mom, how = model_moments(gen, self.model_samples, seed=self.seed + 61_000 + split)
        self.last_moments_source = how
        if self.verbose:
            print(f"  [mahalamia] white box {dataset}/{generator}/split_{split}: "
                  f"moments {how}", flush=True)
        return mom

    def score(self, dataset: str, generator: str, split: int) -> np.ndarray:
        X_real = D.load_expression(dataset).values.astype(np.float64)
        ref = D.load_reference(dataset) if self.use_reference else None
        X_aux = ref.values.astype(np.float64) if ref is not None else None

        if self.white_box:
            # No projection (refused in __post_init__), so gene space throughout.
            y_real = D.encode_subtypes(dataset, D.load_subtypes(dataset).values)
            d_syn = class_shrunk_distance_from_moments(
                X_real, y_real, self._target_moments(dataset, generator, split),
                self.class_shrink, self.ridge_alpha)
            return self._finish(d_syn, self._pooled_distance(X_real, X_aux)
                                if X_aux is not None else None)

        target = T.load_target(dataset, generator, split)
        X_syn = target["X"].astype(np.float64)
        y_syn = target["y_int"]

        proj = self._projection(X_syn, X_aux)
        if not self.pca_per_class:
            X_real, X_syn = proj(X_real), proj(X_syn)
            if X_aux is not None:
                X_aux = proj(X_aux)

        if self.class_shrink is not None:
            # The paper's MahalaMIA.  The reference set has no labels, so its
            # distance stays pooled.
            y_real = D.encode_subtypes(dataset, D.load_subtypes(dataset).values)
            d_syn = class_shrunk_distance(X_real, y_real, X_syn, y_syn,
                                          self.class_shrink, self.ridge_alpha)
            if X_aux is not None:
                d_aux = self._pooled_distance(X_real, X_aux)
        elif self.class_conditional:
            y_real = D.encode_subtypes(dataset, D.load_subtypes(dataset).values)
            d_syn = self._conditional_distance(X_real, y_real, X_syn, y_syn)
            if X_aux is not None:
                y_aux = self._pseudo_label(X_aux, X_syn, y_syn)
                d_aux = self._conditional_distance(X_real, y_real, X_aux, y_aux)
        else:
            d_syn = self._pooled_distance(X_real, X_syn)
            if X_aux is not None:
                d_aux = self._pooled_distance(X_real, X_aux)

        return self._finish(d_syn, d_aux if X_aux is not None else None)

    def _finish(self, d_syn: np.ndarray, d_aux: np.ndarray | None) -> np.ndarray:
        """Distances -> membership score: closeness to the release, relative to the reference."""
        if d_aux is None:
            raw = 1.0 / (d_syn + 1e-10)
        else:
            raw = d_aux / (d_syn + d_aux + 1e-10)

        raw = np.nan_to_num(raw, nan=float(np.nanmedian(raw)))
        return sigmoid_calibrate(raw) if self.calibrate else raw


register(MahalaMIA)
