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

Reference: Golob et al., "Privacy Vulnerabilities in Marginals-based Synthetic
Data" (MAMA-MIA).  This is a from-scratch reimplementation for expression data;
the original targets categorical tabular generators (MST, PrivBayes, GSD, RAP).
"""

from __future__ import annotations

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

    name = "mamamia"

    def tag(self) -> str:
        parts = [f"k{self.n_bins}"]
        if self.use_oneway:
            parts.append("1w")
        if self.use_twoway:
            parts.append("2w")
        return "_".join(parts)

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
        n_used = 0

        if self.use_oneway:
            p_syn = _oneway_probs(bins_syn, self.n_bins)
            p_aux = _oneway_probs(bins_aux, self.n_bins)
            gene_idx = np.arange(n_genes)
            # ratios[i, j] = p_syn[j, bin of sample i in gene j] / p_aux[...]
            ratios = (np.maximum(p_syn[gene_idx, bins_real], EPS)
                      / np.maximum(p_aux[gene_idx, bins_real], EPS))
            total += ratios.sum(axis=1)
            n_used += n_genes

        if self.use_twoway:
            q_syn = _twoway_probs(bins_syn, y_syn, self.n_bins, n_classes)
            q_aux = _twoway_probs(bins_aux, y_aux, self.n_bins, n_classes)
            gene_idx = np.arange(n_genes)
            lab = y_real[:, None]
            ratios = (np.maximum(q_syn[gene_idx, bins_real, lab], EPS)
                      / np.maximum(q_aux[gene_idx, bins_real, lab], EPS))
            total += ratios.sum(axis=1)
            n_used += n_genes

        if self.use_subtype_marginal:
            p_syn_s = np.bincount(y_syn, minlength=n_classes) / max(len(y_syn), 1)
            p_aux_s = np.bincount(y_aux, minlength=n_classes) / max(len(y_aux), 1)
            total += (np.maximum(p_syn_s[y_real], EPS)
                      / np.maximum(p_aux_s[y_real], EPS))
            n_used += 1

        raw = total / max(n_used, 1)
        raw = np.nan_to_num(raw, nan=float(np.nanmedian(raw)))
        return sigmoid_calibrate(raw) if self.calibrate else raw


register(MAMAMIA)
