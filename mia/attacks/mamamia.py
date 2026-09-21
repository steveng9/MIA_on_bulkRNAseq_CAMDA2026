"""MAMA-MIA -- marginal-ratio membership inference against DP-PGM.

DP-PGM's entire released signal is a set of noisy low-order marginals of the
discretised training data: it picks marginals, perturbs their counts under the
Gaussian mechanism, fits a graphical model to them, and samples.  Anything the
synthetic data says about the training set, it says through those marginals --
so that is where the attack looks.

For each targeted marginal m and candidate x, the domain ratio

    r_m(x) = p_syn(x_m) / p_aux(x_m)

asks whether x's value pattern is over-represented in the synthetic data
relative to a population baseline.  A member pushes its own cell of every
marginal it participates in slightly upward, and averaging r_m over ~2000
marginals accumulates those individually tiny effects into a usable score.
Dividing by p_aux is what stops the score from simply rewarding common values.

Targeted marginals (following the CAMDA abstract):
  * one-way per gene, p(g_i) for all 978 genes
  * the subtype marginal p(s)
  * two-way gene x subtype, p(g_i, s) for all 978 genes

Genes are continuous, so they are quantile-binned first, using edges estimated
from the auxiliary pool.  The bin count should match the generator's own
discretisation: attacking at a finer resolution than DP-PGM modelled just adds
noise, since the generator never saw the finer structure.

Aggregation
-----------
`aggregation="ratio"` takes the mean of p_syn/p_aux, as first written.  A ratio
is bounded below by 0 and unbounded above, so that mean is dominated by cells
where p_aux happens to be small -- precisely the cells whose ratio is least
reliable.  `"log"` sums log-ratios instead, which is the Neyman-Pearson
statistic for "was this record in the set that produced p_syn".  `"log_ivw"`
additionally weights each marginal family by inverse noise variance, which the
adversary can compute from published (epsilon, delta, budget weights).

Focal points
------------
For this generator the targeted marginals need no selection step.  The cohorts
carry exactly 978 genes and the release is configured with n_1way=978, so
"top 978 by variance" is every gene; with n_2way=0 and label marginals on, the
clique set is 978 one-way plus 978 gene x label, fixed by public configuration
and independent of the data.  Shadow models exist in MAMA-MIA to discover which
marginals a nondeterministic selector chose (MST, PrivBayes, GSD); here there is
nothing to discover.  That stops being true at n_1way < 978, where the variance
ranking becomes a genuine data-dependent choice.

Reference: Golob et al., "Privacy Vulnerabilities in Marginals-based Synthetic
Data" (MAMA-MIA).  This is a from-scratch reimplementation for expression data;
the original targets categorical tabular generators (MST, PrivBayes, GSD, RAP).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .. import datasets as D
from .. import targets as T
from .base import Attack, register
from .mahalamia import sigmoid_calibrate

EPS = 1e-10


def quantile_bin_edges(X: np.ndarray, n_bins: int) -> np.ndarray:
    """Per-gene bin edges at equal-frequency quantiles.  Shape (n_genes, n_bins-1)."""
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    return np.quantile(X, qs, axis=0).T


def digitize(X: np.ndarray, edges: np.ndarray, n_bins: int) -> np.ndarray:
    """Assign every value to a bin index in [0, n_bins).  Shape preserved."""
    out = np.empty(X.shape, dtype=np.int64)
    for j in range(X.shape[1]):
        out[:, j] = np.searchsorted(edges[j], X[:, j], side="right")
    return np.clip(out, 0, n_bins - 1)


def _oneway_probs(bins: np.ndarray, n_bins: int) -> np.ndarray:
    """p(g_i = b) for every gene.  Shape (n_genes, n_bins).

    Counted with a single `bincount` over flattened (gene, bin) codes rather
    than a loop over bins, so cost is independent of the bin count -- which
    matters, because the interesting regime for deep generators turns out to be
    hundreds of bins.
    """
    n_rows, n_genes = bins.shape
    codes = (np.arange(n_genes, dtype=np.int64) * n_bins)[None, :] + bins
    counts = np.bincount(codes.ravel(), minlength=n_genes * n_bins).astype(np.float64)
    counts = counts.reshape(n_genes, n_bins)
    return counts / max(n_rows, 1)


def _twoway_probs(bins: np.ndarray, labels: np.ndarray, n_bins: int,
                  n_classes: int) -> np.ndarray:
    """p(g_i = b, s = c) for every gene.  Shape (n_genes, n_bins, n_classes)."""
    n_rows, n_genes = bins.shape
    gene = (np.arange(n_genes, dtype=np.int64) * n_bins * n_classes)[None, :]
    codes = gene + bins * n_classes + labels.astype(np.int64)[:, None]
    counts = np.bincount(codes.ravel(),
                         minlength=n_genes * n_bins * n_classes).astype(np.float64)
    return counts.reshape(n_genes, n_bins, n_classes) / max(n_rows, 1)


@dataclass
class MAMAMIA(Attack):
    n_bins: int = 4
    use_oneway: bool = True
    use_twoway: bool = True
    use_subtype_marginal: bool = True
    calibrate: bool = True

    #: How per-marginal evidence is combined.
    #:   "ratio" -- mean of p_syn/p_aux, the original formulation
    #:   "log"   -- mean of log(p_syn/p_aux), the Neyman-Pearson statistic
    #:   "log_ivw" -- log-ratios weighted by inverse noise variance (below)
    aggregation: str = "ratio"

    #: Known release parameters, used only by "log_ivw".  Under Kerckhoffs the
    #: adversary knows these, so it can compute each marginal family's sigma
    #: exactly and weight families by 1/(sigma^2 + sampling variance).
    dp_epsilon: float | None = None
    dp_delta: float = 1e-5
    dp_budget_weights: tuple = (0.33, 0.67)

    name = "mamamia"

    def tag(self) -> str:
        parts = [f"k{self.n_bins}"]
        if self.use_oneway:
            parts.append("1w")
        if self.use_twoway:
            parts.append("2w")
        if self.aggregation != "ratio":
            parts.append(self.aggregation)
        return "_".join(parts)

    def _family_sigma(self, n_cliques: int, weight_idx: int) -> float:
        """sigma for one marginal family under the generator's zCDP accounting.

        Mirrors PrivatePGMFitter._sigma_zcdp at L2 sensitivity 1 (add/remove
        neighbours).  Duplicated rather than imported because the adversary is
        not supposed to reach into the generator -- it reconstructs sigma from
        published parameters, which is exactly the point.
        """
        L = math.log(1.0 / self.dp_delta)
        try:
            from snsynth.utils import cdp_rho
            rho = float(cdp_rho(self.dp_epsilon, self.dp_delta))
        except Exception:
            rho = (-math.sqrt(L) + math.sqrt(L + self.dp_epsilon)) ** 2
        w = self.dp_budget_weights[weight_idx] / sum(self.dp_budget_weights)
        return math.sqrt(n_cliques / (2.0 * w * rho))

    def _combine(self, p_syn, p_aux):
        """Per-cell evidence, in whichever space `aggregation` asks for."""
        if self.aggregation not in ("ratio", "log", "log_ivw"):
            raise ValueError(
                f"aggregation must be 'ratio', 'log' or 'log_ivw', "
                f"got {self.aggregation!r}")
        r = np.maximum(p_syn, EPS) / np.maximum(p_aux, EPS)
        return r if self.aggregation == "ratio" else np.log(r)

    def score(self, dataset: str, generator: str, split: int) -> np.ndarray:
        n_classes = D.n_classes(dataset)

        X_real = D.load_expression(dataset).values.astype(np.float64)
        y_real = D.encode_subtypes(dataset, D.load_subtypes(dataset).values)

        target = T.load_target(dataset, generator, split)
        X_syn, y_syn = target["X"].astype(np.float64), target["y_int"]

        # Auxiliary baseline: the adversary's own view of the population.  The
        # abstract uses the full labelled real pool, since the provided
        # reference set for COMBINED carries no subtype labels and the two-way
        # marginals need them.
        X_aux, y_aux = X_real, y_real

        # Bin edges come from the auxiliary data only -- never from the target's
        # training half -- so the discretisation itself leaks nothing.
        edges = quantile_bin_edges(X_aux, self.n_bins)
        bins_real = digitize(X_real, edges, self.n_bins)
        bins_syn = digitize(X_syn, edges, self.n_bins)
        bins_aux = digitize(X_aux, edges, self.n_bins)

        n_real, n_genes = bins_real.shape
        total = np.zeros(n_real, dtype=np.float64)
        n_used = 0.0
        n_syn = len(bins_syn)

        # Inverse-variance weights.  The member's contribution to any cell is
        # +1 count; the noise against it is Gaussian DP noise (sigma, absolute)
        # plus multinomial sampling.  For independent estimates the optimal
        # combination weights by inverse variance, and the adversary can
        # compute both terms from published parameters.
        ivw = self.aggregation == "log_ivw" and self.dp_epsilon is not None
        w1 = w2 = 1.0

        if self.use_oneway:
            p_syn = _oneway_probs(bins_syn, self.n_bins)
            p_aux = _oneway_probs(bins_aux, self.n_bins)
            gene_idx = np.arange(n_genes)
            # ev[i, j] = evidence from gene j at sample i's bin
            ev = self._combine(p_syn[gene_idx, bins_real], p_aux[gene_idx, bins_real])
            if ivw:
                p = 1.0 / self.n_bins          # quantile bins: every cell is 1/k
                w1 = 1.0 / (self._family_sigma(n_genes, 0) ** 2
                            + n_syn * p * (1 - p))
            total += w1 * ev.sum(axis=1)
            n_used += w1 * n_genes

        if self.use_twoway:
            q_syn = _twoway_probs(bins_syn, y_syn, self.n_bins, n_classes)
            q_aux = _twoway_probs(bins_aux, y_aux, self.n_bins, n_classes)
            gene_idx = np.arange(n_genes)
            lab = y_real[:, None]
            ev = self._combine(q_syn[gene_idx, bins_real, lab],
                               q_aux[gene_idx, bins_real, lab])
            if ivw:
                pc = np.bincount(y_aux, minlength=n_classes) / max(len(y_aux), 1)
                p = (pc / self.n_bins).mean()
                w2 = 1.0 / (self._family_sigma(n_genes, 1) ** 2
                            + n_syn * p * (1 - p))
            total += w2 * ev.sum(axis=1)
            n_used += w2 * n_genes

        if self.use_subtype_marginal:
            p_syn_s = np.bincount(y_syn, minlength=n_classes) / max(len(y_syn), 1)
            p_aux_s = np.bincount(y_aux, minlength=n_classes) / max(len(y_aux), 1)
            total += self._combine(p_syn_s[y_real], p_aux_s[y_real])
            n_used += 1

        raw = total / max(n_used, 1e-12)
        raw = np.nan_to_num(raw, nan=float(np.nanmedian(raw)))
        if not self.calibrate:
            return raw
        # "log"/"log_ivw" already produce signed, log-space scores.
        return sigmoid_calibrate(raw, log_transform=self.aggregation == "ratio")


register(MAMAMIA)
