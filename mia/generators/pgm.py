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

Composition
-----------
`composition="zcdp"` is the default and is the accounting McKenna's own
mechanisms (MST, AIM) use: Gaussian releases compose additively in rho, so the
noise scale grows as sqrt(k) in the number of marginals rather than linearly.
At our configuration -- 978 genes, joint mode, 1956 marginals, eps=10 -- that is
sigma 29 against sigma 1436, a 50x reduction at an identical (eps, delta).
`composition="basic"` keeps the original linear split so earlier results
reproduce.

With the default `binning="quantile"` the reported epsilon is not end to end:
discretisation takes bin edges from percentiles of the private data and spends
zero budget (and at n_1way < 978, so does marginal selection).  `binning=
"uniform"` or `"dp_quantile"` closes the discretisation gap.  See
docs/PGM_ATTACK_SURFACE.md.
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
    #: "zcdp" (default) or "basic".  See the composition note in the module
    #: docstring; "basic" reproduces every DP-PGM result recorded before
    #: 2026-09-20.
    composition: str = "zcdp"

    #: "add_remove" (default, unbounded DP, L2 sensitivity 1, matches MST) or
    #: "replace" (bounded DP, sensitivity sqrt(2), so sqrt(2)x the noise but the
    #: row count is public).  Under "add_remove" the fitted model's row count is
    #: estimated from the noisy marginals rather than released exactly.
    neighboring: str = "add_remove"

    #: How bin edges are chosen.  "quantile" (default, reproduces every earlier
    #: target) takes private percentiles and spends no budget, so epsilon does
    #: not cover it.  "uniform" is equal width over the public `bin_range` and
    #: costs nothing; "dp_quantile" is equal depth from a noisy histogram and
    #: spends `binning_budget` of rho, the marginals getting the rest.
    binning: str = "quantile"
    bin_range: tuple = (0.0, 24.0)
    binning_budget: float = 0.1
    bin_grid: int = 48
    #: How the dp_* strategies read edges off their noisy histogram: "clip"
    #: (legacy, every target built before 2026-09-23; biased by the positive
    #: noise of empty cells, so bounds land near the ends of `bin_range` even at
    #: eps=1000) or "threshold" (drop cells under a noise threshold, as
    #: smartnoise's approx_bounds does; interpolate within cells).
    edge_estimator: str = "clip"

    #: Gene-gene structure.  "hierarchical" (default) is the upstream selector,
    #: which with n_2way=0 is the star of last year's winner.  "tree" adds a
    #: spanning tree of (gene, gene) marginals and "tree_label" of (gene, gene,
    #: label) marginals, selected by the exponential mechanism as in MST, at
    #: `select_budget` of rho.  Both need joint_mode and zcdp.  "forest"
    #: (Steven, 2026-09-24) keeps only `k_label` DP-chosen (gene, label)
    #: tables, `l_pairs` DP-chosen (gene, gene) tables forming a forest, and a
    #: 1-way table for every gene neither covers.
    structure: str = "hierarchical"
    select_budget: float = 0.3
    k_label: int = 978
    l_pairs: int = 0

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
            random_seed=self.seed, composition=self.composition,
            neighboring=self.neighboring, binning=self.binning,
            bin_range=tuple(self.bin_range), binning_budget=self.binning_budget,
            bin_grid=self.bin_grid, edge_estimator=self.edge_estimator,
            structure=self.structure, select_budget=self.select_budget,
            k_label=self.k_label, l_pairs=self.l_pairs,
        )
        # The upstream generator takes string labels; integers round-trip fine.
        self._gen.fit(X, y.astype(str), gene_names=self._gene_names)
        return self

    def sample(self, n: int) -> tuple:
        """Draw `n` rows, restored to the cohort's canonical gene order.

        The upstream generator selects genes by variance and returns its matrix
        in *that* order, naming the columns through `selected_gene_names` -- its
        own runner writes the CSV with exactly those labels.  This adapter used
        to drop the labels and hand the matrix back as though column j were gene
        j, which silently transposed every gene's distribution onto another
        gene.  Global moments survive that (the multiset of columns is
        unchanged), so it was invisible to any check that did not look per gene:
        mean per-gene Wasserstein distance was 2.9 training SDs, downstream
        macro-F1 was 0.09 against a real ceiling of 0.81, and a real-vs-synthetic
        discriminator scored AUC 1.000.  Every DP-PGM result recorded before
        2026-09-20 is affected.

        When `n_1way` is smaller than the gene count the model never saw the
        remaining genes, so there is nothing to place in those columns.  They
        are filled with the median of the synthetic matrix -- a constant derived
        from released data, so it adds no leak -- and the utility metrics take
        the hit, which is the honest accounting: those genes were not released.
        """
        X, y = self._gen.generate(n)
        X = np.asarray(X, dtype=np.float32)

        selected = list(self._gen.selected_gene_names)
        if selected == self._gene_names:
            out = X
        else:
            pos = {g: j for j, g in enumerate(self._gene_names)}
            out = np.full((len(X), len(self._gene_names)), np.median(X),
                          dtype=np.float32)
            for j, gene in enumerate(selected):
                out[:, pos[gene]] = X[:, j]

        return out, np.asarray(y).astype(np.int64)

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
