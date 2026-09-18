"""Loss-feature representation shared by both MeLoMIA backends.

Both backends produce the same shape of evidence: a per-sample grid of
reconstruction losses indexed by (sweep point, noise draw), optionally plus a
block of extra per-sample scalars.

    losses : (n_samples, n_sweep, n_noise)
    extra  : (n_samples, n_extra)   -- may be empty

For NoisyDiffusion the sweep axis is diffusion timesteps and a "noise draw" is
one of the frozen epsilon vectors.  For the CVAE the sweep axis is posterior
temperature and a draw is one of the frozen latent perturbations.  Making them
the same object is what lets one meta-classifier, one Optuna search space and
one summariser serve both attacks.

Extraction is expensive and the useful subset of sweep points is not known in
advance, so the full grid is computed once at the widest setting and every later
choice -- which timesteps, how many draws -- is a cheap slice of that cached
array.  Draws are ordered by a fixed seed, so taking the first `noise_budget` of
them is a consistent subsample rather than an arbitrary one.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import skew as scipy_skew

#: per-sweep-point statistics, in the order they are concatenated
STAT_NAMES = ("mean", "std", "min", "max", "median", "skew")


def summary_dim(n_sweep: int, n_extra: int = 0) -> int:
    """Width of the summarised feature vector."""
    return n_sweep * len(STAT_NAMES) + max(n_sweep - 1, 0) + n_extra


def slice_losses(losses: np.ndarray, sweep_indices, noise_budget: int) -> np.ndarray:
    """Restrict the cached grid to a subset of sweep points and draws."""
    idx = list(sweep_indices)
    return losses[:, idx, :][:, :, :noise_budget]


def summarize(losses: np.ndarray, extra: np.ndarray | None = None) -> np.ndarray:
    """Collapse the draw axis of a dense grid into per-sweep-point statistics."""
    return prepare(losses, extra, range(losses.shape[1]), losses.shape[2])


def prepare(losses: np.ndarray, extra, sweep_indices, noise_budget: int) -> np.ndarray:
    """Slice and summarise in one pass -- the standard path from cache to model.

    Averaging away the draw axis is not just compression: an individual draw is
    dominated by which noise vector happened to be sampled, while the spread and
    shape of the loss distribution at a given sweep point is the part that
    reflects how well the model has memorised this particular sample.  The first
    difference of the means across sweep points adds the trajectory's slope,
    which separates samples whose losses are uniformly low from those that stay
    low as the sweep gets harder.

    Statistics are computed on views of the cached array rather than on a
    materialised slice -- the pooled grid runs to gigabytes and the Optuna
    search re-slices it hundreds of times.
    """
    idx = list(sweep_indices)
    n_sweep = len(idx)
    stats = {name: np.empty((len(losses), n_sweep), dtype=np.float32)
             for name in STAT_NAMES}

    for col, si in enumerate(idx):
        block = losses[:, si, :noise_budget]          # view, no copy
        stats["mean"][:, col] = block.mean(axis=1)
        stats["std"][:, col] = block.std(axis=1)
        stats["min"][:, col] = block.min(axis=1)
        stats["max"][:, col] = block.max(axis=1)
        stats["median"][:, col] = np.median(block, axis=1)
        stats["skew"][:, col] = np.nan_to_num(scipy_skew(block, axis=1), nan=0.0)

    parts = [stats[name] for name in STAT_NAMES]
    if n_sweep > 1:
        parts.append(np.diff(stats["mean"], axis=1))
    if extra is not None and np.size(extra):
        parts.append(np.asarray(extra, dtype=np.float32))
    return np.concatenate(parts, axis=1).astype(np.float32)


def calibrate_against_reference(losses: np.ndarray, ref_losses: np.ndarray) -> np.ndarray:
    """Z-score each sweep point against a known-non-member reference population.

    A sample's raw loss mixes two things: how well the model memorised it, and
    how intrinsically hard it is to reconstruct.  Only the first is membership
    evidence.  Standardising each sweep point by the reference set's mean and
    spread removes the per-model scale and leaves the deviation from what a
    non-member at that sweep point looks like.

    Only available for cohorts that ship an auxiliary set (TCGA-COMBINED).
    """
    if ref_losses is None or ref_losses.size == 0:
        return losses
    mu = ref_losses.mean(axis=(0, 2), keepdims=True)
    sd = ref_losses.std(axis=(0, 2), keepdims=True)
    return (losses - mu) / (sd + 1e-8)
