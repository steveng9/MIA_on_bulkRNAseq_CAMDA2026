"""Differentially private graphical-model target generator ("pgm").

Thin adapter over the StratHiM-PGM implementation in the sibling
`private-pgm-rnaseq-camda2026` repo (Private-PGM / mbi under the hood).  That
repo is added to `sys.path` lazily so importing `mia.generators` still works on
a machine that does not have it.

Unlike the other three, this generator is genuinely differentially private: the
budget is spent on noisy low-order marginals of the discretised expression
matrix, which is why it is both the lowest-fidelity target and the hardest to
attack.  Those same marginals are the surface MAMA-MIA aims at.

`joint_mode=True` reproduces the CAMDA 2025 winner's structure (one PGM over all
classes with the label as a node plus every gene x label marginal), which is the
configuration the challenge's DP-PGM synthetic data came from.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import paths
from .base import Generator, register


def _import_upstream():
    src = paths.PGM_REPO / "src"
    if not src.exists():
        raise FileNotFoundError(
            f"Private-PGM repo not found at {paths.PGM_REPO}. "
            "Set CAMDA_PGM_REPO or clone private-pgm-rnaseq-camda2026."
        )
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from generator import StratHiMPGMGenerator  # noqa: E402
    return StratHiMPGMGenerator


@dataclass
class PGMGenerator(Generator):
    epsilon: float = 10.0
    delta: float = 1e-5
    n_bins: int = 4
    n_1way: int = 978
    n_2way: int = 0
    n_3way: int = 0
    n_4way: int = 0
    budget_weights: tuple = (0.33, 0.67, 0.0, 0.0)
    pgm_iters: int = 1000
    joint_mode: bool = True

    name = "pgm"

    def __post_init__(self):
        self._gen = None
        self._gene_names = None
        self._n_train = None

    def fit(self, X: np.ndarray, y: np.ndarray, n_classes: int) -> "PGMGenerator":
        StratHiMPGMGenerator = _import_upstream()
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self._n_train = len(X)
        self._gene_names = [f"gene_{i}" for i in range(X.shape[1])]

        self._gen = StratHiMPGMGenerator(
            epsilon=self.epsilon, delta=self.delta, n_bins=self.n_bins,
            n_1way=self.n_1way, n_2way=self.n_2way, n_3way=self.n_3way,
            n_4way=self.n_4way, budget_weights=tuple(self.budget_weights),
            pgm_iters=self.pgm_iters, joint_mode=self.joint_mode,
            random_seed=self.seed,
        )
        # The upstream generator takes string labels; integers round-trip fine.
        self._gen.fit(X, y.astype(str), gene_names=self._gene_names)
        return self

    def sample(self, n: int) -> tuple:
        X, y = self._gen.generate(n)
        return np.asarray(X, dtype=np.float32), np.asarray(y).astype(np.int64)

    def save(self, path: Path) -> None:
        import pickle
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._gen, f)

    def load(self, path: Path) -> "PGMGenerator":
        import pickle
        _import_upstream()  # make the unpickled classes importable
        with open(Path(path), "rb") as f:
            self._gen = pickle.load(f)
        return self


register(PGMGenerator)
