"""The CAMDA-25 winner's DP-PGM, exactly as the CAMDA-26 challenge ran it ("pgg").

Wraps `pgg_pgm.py` from the winning blue-team submission
(github.com/sikhapentyala/Health-Privacy-Challenge, `submission/`), which in turn
calls PRO-GENE-GEN's Private_PGM (github.com/MarieOestreich/PRO-GENE-GEN,
`models/Private_PGM`).  The CAMDA-26 red-team README names this code as the
source of `synthetic_data_2.csv`, the challenge's DP-PGM release (eps = 10).

This is NOT the same generator as `pgm@composition=basic,neighboring=
legacy_exact_n`, our earlier stand-in built from Steven's StratHiM-PGM fork
(2026-09-30).  The differences, all in the winner's favour for fidelity:

  * Noise: Gaussian, sigma calibrated by RDP for the whole release and spread
    over the ~1,957 marginals by L2-normalised weights, i.e. proper Gaussian
    composition (sigma 0.76 at eps = 10, weights normalised per round over
    979 + 978 marginals, so per-cell sd ~ 0.76 * sqrt(979) ~ 24).  The fork's "basic" mode split
    epsilon linearly (sd ~ 1,440), which drowned every label link.
  * Decoding: each sampled bin is replaced by the mean of the TRAINING values in
    that bin, so the release has exactly 4 distinct values per gene and each
    gene's mean is right.  The fork samples uniformly within the bin.
  * Neither the quartile bin edges nor the bin means are privatised, and the
    exact row count is passed to inference, so the release is not DP end to end.

The code below mirrors `PGG_PGM_DataGenerator.discretize / label / train /
generate / dediscretize` line for line; only the file I/O is replaced by the
`fit(X, y)` interface.  PRO-GENE-GEN ships its own (older) `mbi`, which is put
first on sys.path; build pgg targets in their own process.
"""

from __future__ import annotations

import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .base import Generator, register

PGG_REPO = Path(os.environ.get(
    "CAMDA_PGG_REPO",
    "~/sikha-Health-Privacy-Challenge/submission/blueteam_PPML-Huskies_TCGA-BRCA",
)).expanduser()


def _import_private_pgm():
    root = PGG_REPO / "PRO-GENE-GEN"
    if not (root / "models" / "Private_PGM" / "model.py").exists():
        raise FileNotFoundError(
            f"PRO-GENE-GEN not found under {PGG_REPO}. Clone "
            "sikhapentyala/Health-Privacy-Challenge and run `git submodule update --init`."
        )
    # utils/__init__ imports `progress.bar` for a progress bar Private_PGM never uses.
    if "progress" not in sys.modules:
        pb, bar = types.ModuleType("progress"), types.ModuleType("progress.bar")
        bar.Bar = object
        pb.bar = bar
        sys.modules["progress"], sys.modules["progress.bar"] = pb, bar
    for p in (root, root / "models" / "Private_PGM"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from model import Private_PGM  # noqa: E402
    return Private_PGM


@dataclass
class PGGGenerator(Generator):
    epsilon: float = 10.0
    delta: float = 1e-5
    iterations: int = 10000

    name = "pgg"

    def __post_init__(self):
        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray, n_classes: int) -> "PGGGenerator":
        Private_PGM = _import_private_pgm()
        np.random.seed(self.seed)
        cols = [f"g{i}" for i in range(X.shape[1])]
        data_copy = pd.DataFrame(np.asarray(X, dtype=np.float64), columns=cols)

        # --- discretize (4 bins at the training quartiles; bin means kept) ---
        alphas = [0.25, 0.5, 0.75]
        self.num_bins = len(alphas) + 1
        data = data_copy.copy()
        data_quantile = np.quantile(data, alphas, axis=0)
        self.mean_dict = {}
        for j, col in enumerate(cols):
            q = data_quantile[:, j]
            discrete_col = np.digitize(data[col], q)
            data[col] = discrete_col
            means = []
            for b in range(self.num_bins):
                m = np.mean(data_copy[col][discrete_col == b]) if (discrete_col == b).any() else np.nan
                if np.isnan(m):
                    if b == 0:
                        m = (np.min(data_copy[col]) + q[0]) / 2
                    elif b == self.num_bins - 1:
                        m = (np.max(data_copy[col]) + q[-1]) / 2
                    else:
                        m = (q[b] + q[b + 1]) / 2
                means.append(m)
            self.mean_dict[col] = np.asarray(means)
        self._cols = cols

        # --- label ---
        uniq = np.unique(np.asarray(y))
        mapping = {v: i for i, v in enumerate(uniq)}
        self._delabel = uniq
        self._label = "label"
        yl = pd.DataFrame({self._label: [mapping[v] for v in np.asarray(y)]})

        # --- train ---
        domain = {self._label: len(mapping)}
        for col in cols:
            domain[col] = self.num_bins
        train_data = pd.concat([data, yl], axis=1)
        self._model = Private_PGM(self._label, True, self.epsilon, self.delta)
        self._model.train(train_data, domain, num_iters=self.iterations)
        return self

    def sample(self, n: int) -> tuple:
        out = self._model.generate(num_rows=n)
        Xb, yb = out[:, :-1].astype(int), out[:, -1].astype(int)
        X = np.column_stack([self.mean_dict[c][Xb[:, j]] for j, c in enumerate(self._cols)])
        return X.astype(np.float32), self._delabel[yb].astype(np.int64)


register(PGGGenerator)
