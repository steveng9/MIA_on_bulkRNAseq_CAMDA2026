"""Multivariate-normal generator (target "mvn").

The CAMDA 2025 statistical baseline: fit a per-class Gaussian to the training
split and draw from it, then add extra Gaussian noise whose covariance is a
fixed multiple of the class covariance.

`noise_level` is the knob that trades fidelity for privacy; the abstract's
headline MVN column uses 0.7.  Because the released synthetic data is literally
a sample from N(mu_c, (1 + noise_level) * Sigma_c), the covariance of the
synthetic set is the direct imprint of the training set -- which is exactly the
signal MahalaMIA exploits.

Ported from the challenge starter package
(src/generators/models/multivariate.py), with the CSV round-trip removed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .base import Generator, register


def nearest_psd(cov: np.ndarray) -> np.ndarray:
    """Symmetrise and clip negative eigenvalues so the matrix can be sampled."""
    cov = (cov + cov.T) / 2
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 0)
    cov = eigvecs @ np.diag(eigvals) @ eigvecs.T
    min_eig = float(np.min(np.real(np.linalg.eigvals(cov))))
    if min_eig < 0:
        cov -= 10 * min_eig * np.eye(*cov.shape)
    return cov


@dataclass
class MVNGenerator(Generator):
    noise_level: float = 0.7

    name = "mvn"

    def __post_init__(self):
        self._means: dict = {}
        self._covs: dict = {}
        self._counts: dict = {}
        self._n_genes = None

    def fit(self, X: np.ndarray, y: np.ndarray, n_classes: int) -> "MVNGenerator":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        self._n_genes = X.shape[1]
        for c in np.unique(y):
            Xc = X[y == c]
            if len(Xc) < 2:
                raise ValueError(f"class {c} has {len(Xc)} training samples; cannot estimate covariance")
            self._means[int(c)] = Xc.mean(axis=0)
            self._covs[int(c)] = nearest_psd(np.cov(Xc, rowvar=False))
            self._counts[int(c)] = int(len(Xc))
        return self

    def sample(self, n: int) -> tuple:
        """Draw `n` samples, preserving the training class proportions."""
        rng = np.random.default_rng(self.seed)
        total = sum(self._counts.values())
        parts_X, parts_y = [], []
        for c, count in self._counts.items():
            n_c = max(1, int(round(n * count / total)))
            mean, cov = self._means[c], self._covs[c]
            base = rng.multivariate_normal(mean, cov, size=n_c)
            noise = rng.multivariate_normal(
                np.zeros(self._n_genes), cov * self.noise_level, size=n_c
            )
            parts_X.append(base + noise)
            parts_y.append(np.full(n_c, c, dtype=np.int64))

        X = np.concatenate(parts_X).astype(np.float32)
        y = np.concatenate(parts_y)
        order = rng.permutation(len(X))[:n]
        return X[order], y[order]

    def moments(self) -> dict:
        """Class c is drawn from N(mu_c, Sigma_c) plus N(0, noise_level Sigma_c)."""
        total = sum(self._counts.values())
        return {c: (self._means[c], (1.0 + self.noise_level) * self._covs[c],
                    self._counts[c] / total)
                for c in self._counts}

    # ── Persistence ─────────────────────────────────────────────────────────
    # The targets were built before MVN could be saved; the fit is a
    # deterministic function of the training split, so
    # scripts/whitebox/materialise_mvn_models.py refits each one, checks that
    # it re-emits the released data, and saves it with this.

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        classes = sorted(self._counts)
        with open(path, "wb") as f:        # a file object: np.savez must not append .npz
            np.savez(f, classes=np.array(classes, dtype=np.int64),
                     means=np.stack([self._means[c] for c in classes]),
                     covs=np.stack([self._covs[c] for c in classes]),
                     counts=np.array([self._counts[c] for c in classes], dtype=np.int64),
                     noise_level=np.float64(self.noise_level))

    def load(self, path: Path) -> "MVNGenerator":
        d = np.load(Path(path))
        if float(d["noise_level"]) != float(self.noise_level):
            raise ValueError(f"{path} was fitted with noise_level={float(d['noise_level'])}, "
                             f"this generator has {self.noise_level}")
        classes = [int(c) for c in d["classes"]]
        self._means = {c: d["means"][i] for i, c in enumerate(classes)}
        self._covs = {c: d["covs"][i] for i, c in enumerate(classes)}
        self._counts = {c: int(d["counts"][i]) for i, c in enumerate(classes)}
        self._n_genes = d["means"].shape[1]
        return self


register(MVNGenerator)
