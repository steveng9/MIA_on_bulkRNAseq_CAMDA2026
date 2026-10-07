"""DESeq2's variance-stabilising transform, in the closed form the challenge used.

The challenge preprocessing (PMBio/Health-Privacy-Challenge, data/README.md):
keep genes with a count >= 1 in at least 10% of samples, `DESeq()`, then
`vst(dds, blind=FALSE)`, then keep the 978 landmark genes.  With the default
parametric dispersion trend  disp(mu) = asymptDisp + extraPois / mu  the
transform of a normalised count q = K / size_factor is

    vst(q) = log2( (1 + extraPois + 2 a q + 2 sqrt(a q (1 + extraPois + a q))) / (4 a) ),
    a = asymptDisp

so a fitted transform is two numbers plus one size factor per sample, and the
size factor of a sample is the median over genes of K / geometric_mean(K over
the fitting cohort).  That makes the transform *freezable*: a new sample is
placed in the cohort's space from its own counts and the cohort's gene-wise
geometric means, without refitting.
"""

from __future__ import annotations

import numpy as np


def gene_filter(counts: np.ndarray) -> np.ndarray:
    """Challenge rule on genes x samples counts: >= 1 count in >= 10% of samples."""
    return (counts >= 1).sum(1) >= 0.1 * counts.shape[1]


def log_geo_means(counts: np.ndarray) -> np.ndarray:
    """Per-gene mean log count; -inf for a gene with any zero (DESeq2's rule)."""
    with np.errstate(divide="ignore"):
        return np.log(counts.astype(np.float64)).mean(1)


def size_factors(counts: np.ndarray, loggeo: np.ndarray) -> np.ndarray:
    """Median-of-ratios size factors of genes x samples counts against `loggeo`."""
    ok = np.isfinite(loggeo)
    out = np.empty(counts.shape[1])
    for j in range(counts.shape[1]):
        k = counts[ok, j].astype(np.float64)
        pos = k > 0
        out[j] = np.exp(np.median(np.log(k[pos]) - loggeo[ok][pos]))
    return out


def vst(q: np.ndarray, asympt_disp: float, extra_pois: float) -> np.ndarray:
    a, e = asympt_disp, extra_pois
    return np.log2((1 + e + 2 * a * q + 2 * np.sqrt(a * q * (1 + e + a * q))) / (4 * a))


def fit_coefficients(q: np.ndarray, target: np.ndarray, start=(0.5, 3.0)):
    """Least-squares (asymptDisp, extraPois) such that vst(q) matches `target`."""
    from scipy.optimize import least_squares
    q, target = q.ravel(), target.ravel()
    fun = lambda p: vst(q, np.exp(p[0]), np.exp(p[1])) - target  # noqa: E731
    sol = least_squares(fun, np.log(start), x_scale=1.0, xtol=1e-15, ftol=1e-15, gtol=1e-15)
    return float(np.exp(sol.x[0])), float(np.exp(sol.x[1])), float(np.abs(sol.fun).max())
