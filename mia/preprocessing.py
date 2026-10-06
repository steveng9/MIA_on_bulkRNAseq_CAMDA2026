"""Feature preprocessing shared by generators and attacks.

Which transform a model was trained under is part of its identity: loss features
extracted under different quantile boundaries are not comparable, so a fitted
transform is always persisted next to the weights rather than refitted at
inference.

A preprocessing choice is written as a *spec string*, a `+`-separated chain of
steps applied left to right and inverted right to left, each step's arguments
following its name after colons:

    "standard"                      one step (the CAMDA CVAE baseline)
    "clip:0.001:0.999+quantile"     winsorise each gene, then quantile-normal
    "standard+pca:64"               z-score, then keep 64 principal components
    "fixed:0:24"                    scale by public bounds, touching no data

The spec is what goes in a generator's `preprocess` field, so it lands in the
target's `meta.json` and in the variant name (`tabsyn@preprocess=standard+pca:64`,
which is why the syntax avoids commas, brackets and anything a shell or a
directory name would mind), and a preprocessing ablation is an ordinary target
sweep.

Three properties of a chain matter beyond its output, and `describe()` reports
them so they are recorded with every target:

  lossless          `inverse(transform(X)) == X` up to float error.  `clip` and
                    `pca` are not: what they discard is gone from the release.
  data_independent  nothing is estimated from the training rows.  A DP generator
                    whose preprocessing is *not* data-independent (or itself DP)
                    is not DP end to end -- the scaler's means, quantiles or
                    bin edges are released in the clear through the synthetic
                    data.  That is a real defect of the CAMDA DP baselines (see
                    `generators/pgg.py`, `generators/dpcvae.py`), not a
                    technicality.
  uses_labels       the step needs the class label in both directions.

The four original names -- "quantile", "standard", "minmax", "none" -- still
return the bare scikit-learn object they always did, so every existing target,
shadow and pickled scaler is reproduced bit for bit.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import (MinMaxScaler, QuantileTransformer,
                                   RobustScaler, StandardScaler)

LEGACY = ("quantile", "standard", "minmax", "none")


# ─────────────────────────────────────────────────────────────────────────────
# Steps
# ─────────────────────────────────────────────────────────────────────────────

class Step:
    """One invertible (or deliberately lossy) column transform."""

    name = "step"
    lossless = True
    data_independent = False
    uses_labels = False

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        raise NotImplementedError

    def inverse_transform(self, X, y=None):
        raise NotImplementedError

    def spec(self) -> str:
        return self.name


class _Sklearn(Step):
    """Wraps a scikit-learn transformer."""

    def __init__(self, name, est, spec=None):
        self.name, self.est, self._spec = name, est, spec or name

    def fit(self, X, y=None):
        self.est.fit(X)
        return self

    def transform(self, X, y=None):
        return self.est.transform(X)

    def inverse_transform(self, X, y=None):
        return self.est.inverse_transform(X)

    def spec(self):
        return self._spec


class Clip(Step):
    """Winsorise each gene at its training quantiles (lo, hi).

    Tames the handful of extreme values that otherwise dominate a min-max range
    or a Gaussian likelihood.  Not invertible: the tails are discarded.
    """

    name, lossless = "clip", False

    def __init__(self, lo=0.001, hi=0.999):
        self.lo, self.hi = float(lo), float(hi)

    def fit(self, X, y=None):
        self.lo_, self.hi_ = np.quantile(X, [self.lo, self.hi], axis=0)
        return self

    def transform(self, X, y=None):
        return np.clip(X, self.lo_, self.hi_)

    def inverse_transform(self, X, y=None):
        # also keeps *generated* values inside the training range
        return np.clip(X, self.lo_, self.hi_)

    def spec(self):
        return f"clip:{self.lo:g}:{self.hi:g}"


class Fixed(Step):
    """Affine map of the public range [lo, hi] onto [-1, 1]; estimates nothing.

    The data-independent alternative to `standard`/`minmax` for DP generators.
    DESeq2-VST expression lies in roughly (0, 24).  Values outside the bounds
    are clipped, which also bounds each record's norm.
    """

    name, lossless, data_independent = "fixed", False, True

    def __init__(self, lo=0.0, hi=24.0):
        self.lo, self.hi = float(lo), float(hi)
        if not self.hi > self.lo:
            raise ValueError(f"fixed:{lo}:{hi}: need hi > lo")

    def transform(self, X, y=None):
        X = np.clip(X, self.lo, self.hi)
        return 2.0 * (X - self.lo) / (self.hi - self.lo) - 1.0

    def inverse_transform(self, X, y=None):
        X = np.clip(X, -1.0, 1.0)
        return (X + 1.0) / 2.0 * (self.hi - self.lo) + self.lo

    def spec(self):
        return f"fixed:{self.lo:g}:{self.hi:g}"


class Log1p(Step):
    """log(1 + x): for cohorts that arrive as counts, CPM or TPM."""

    name, data_independent = "log1p", True

    def transform(self, X, y=None):
        return np.log1p(np.clip(X, 0, None))

    def inverse_transform(self, X, y=None):
        return np.clip(np.expm1(X), 0, None)


class ClassCenter(Step):
    """Subtract each class's training mean per gene.

    Bulk cohorts are dominated by between-subtype (or between-tissue) shifts.
    Removing them leaves the within-class residual, which is closer to a single
    mode and easier for an unconditional density model; the generator's label
    then restores the shift on the way out.  Classes unseen in training fall
    back to the global mean.
    """

    name, uses_labels = "classcenter", True

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("classcenter needs labels")
        y = np.asarray(y)
        self.global_ = X.mean(axis=0)
        self.means_ = {int(c): X[y == c].mean(axis=0) for c in np.unique(y)}
        return self

    def _offsets(self, y, n):
        if y is None:
            raise ValueError("classcenter needs labels")
        return np.stack([self.means_.get(int(c), self.global_) for c in np.asarray(y)])

    def transform(self, X, y=None):
        return X - self._offsets(y, len(X))

    def inverse_transform(self, X, y=None):
        return X + self._offsets(y, len(X))


class PCAStep(Step):
    """Keep the top-k principal components (k <= min(n, p)).

    The generator then models k numbers per sample instead of one per gene, and
    the release is exactly rank k (plus the mean).  This is the standard way to
    put a p >> n expression matrix in front of a model built for tens of
    columns, and its cost is explicit: every direction outside the span is
    absent from the synthetic data.  `whiten` gives the components unit variance.
    """

    name, lossless = "pca", False

    def __init__(self, k=64, whiten=0):
        self.k, self.whiten = int(k), bool(int(whiten))

    def fit(self, X, y=None):
        k = min(self.k, X.shape[0] - 1, X.shape[1])
        self.est = PCA(n_components=k, whiten=self.whiten, svd_solver="full").fit(X)
        return self

    def transform(self, X, y=None):
        return self.est.transform(X)

    def inverse_transform(self, X, y=None):
        return self.est.inverse_transform(X)

    def spec(self):
        return f"pca:{self.k}:1" if self.whiten else f"pca:{self.k}"


def _quantile(n_quantiles=1000, output="normal"):
    # scikit-learn's default n_quantiles is 1000, capped at n_samples
    est = QuantileTransformer(n_quantiles=int(n_quantiles),
                              output_distribution=str(output))
    plain = int(n_quantiles) == 1000 and output == "normal"
    return _Sklearn("quantile", est, "quantile" if plain
                    else f"quantile:{int(n_quantiles)}:{output}")


class TabularQuantile(Step):
    """Quantile-normal with the knot count the tabular-diffusion codebases use.

    TabDDPM and TabSyn both set `n_quantiles = max(min(n // 30, 1000), 10)`:
    sensible at the 10^4-10^5 rows they were tuned on, but on a bulk cohort of
    ~900 samples it leaves 29 knots per gene, a much coarser map than
    scikit-learn's default of one knot per sample ("quantile").  Kept as its own
    step so the official recipe and the finer one are both one word away.
    """

    name = "quantile_n30"

    def fit(self, X, y=None):
        n_q = max(min(X.shape[0] // 30, 1000), 10)
        self.est = QuantileTransformer(output_distribution="normal", n_quantiles=n_q,
                                       subsample=int(1e9), random_state=0).fit(X)
        return self

    def transform(self, X, y=None):
        return self.est.transform(X)

    def inverse_transform(self, X, y=None):
        return self.est.inverse_transform(X)


STEPS = {
    "none": None,
    "quantile_n30": TabularQuantile,
    "standard": lambda: _Sklearn("standard", StandardScaler()),
    "minmax": lambda lo=0.0, hi=1.0: _Sklearn(
        "minmax", MinMaxScaler(feature_range=(float(lo), float(hi))),
        "minmax" if (float(lo), float(hi)) == (0.0, 1.0) else f"minmax:{lo:g}:{hi:g}"),
    "robust": lambda: _Sklearn("robust", RobustScaler()),
    "quantile": _quantile,
    "clip": Clip,
    "fixed": Fixed,
    "log1p": Log1p,
    "classcenter": ClassCenter,
    "pca": PCAStep,
}

def _parse_step(text: str):
    name, *raw = (t.strip() for t in text.split(":"))
    if name not in STEPS:
        raise ValueError(f"Unknown preprocessing step {text!r}. Known: {sorted(STEPS)}")
    if STEPS[name] is None:
        return None
    args = []
    for a in raw:
        try:
            args.append(int(a))
        except ValueError:
            try:
                args.append(float(a))
            except ValueError:
                args.append(a)
    return STEPS[name](*args)


# ─────────────────────────────────────────────────────────────────────────────
# Chains
# ─────────────────────────────────────────────────────────────────────────────

class Pipeline:
    """A fitted chain of steps; quacks like a scikit-learn scaler."""

    def __init__(self, spec: str):
        self.steps = [s for s in (_parse_step(t) for t in spec.split("+")) if s is not None]

    def fit_transform(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        for s in self.steps:
            X = s.fit(X, y).transform(X, y)
        return X

    def fit(self, X, y=None):
        self.fit_transform(X, y)
        return self

    def transform(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        for s in self.steps:
            X = s.transform(X, y)
        return X

    def inverse_transform(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        for s in reversed(self.steps):
            X = s.inverse_transform(X, y)
        return X

    def spec(self) -> str:
        return "+".join(s.spec() for s in self.steps) or "none"

    @property
    def uses_labels(self) -> bool:
        return any(s.uses_labels for s in self.steps)


def describe(spec: str) -> dict:
    """Properties of a spec, for a target's record (no data needed)."""
    steps = Pipeline(spec).steps
    return {"spec": "+".join(s.spec() for s in steps) or "none",
            "lossless": all(s.lossless for s in steps),
            "data_independent": all(s.data_independent for s in steps),
            "uses_labels": any(s.uses_labels for s in steps)}


def make_scaler(kind: str):
    """A fresh, unfitted transform for `kind` (a legacy name or a spec string)."""
    if kind == "quantile":
        return QuantileTransformer(output_distribution="normal")
    if kind == "standard":
        return StandardScaler()
    if kind == "minmax":
        return MinMaxScaler()
    if kind == "none":
        return None
    pipe = Pipeline(kind)
    return pipe if pipe.steps else None


def _with_y(scaler, y):
    return {"y": y} if isinstance(scaler, Pipeline) else {}


def fit_scaler(kind: str, X: np.ndarray, y: np.ndarray | None = None):
    """Returns (scaler_or_None, X_transformed float32)."""
    scaler = make_scaler(kind)
    if scaler is None:
        return None, np.asarray(X, dtype=np.float32)
    Xt = scaler.fit_transform(np.asarray(X, dtype=np.float64), **_with_y(scaler, y))
    return scaler, Xt.astype(np.float32)


def apply_scaler(scaler, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
    if scaler is None:
        return np.asarray(X, dtype=np.float32)
    return scaler.transform(np.asarray(X, dtype=np.float64),
                            **_with_y(scaler, y)).astype(np.float32)


def invert_scaler(scaler, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
    if scaler is None:
        return np.asarray(X, dtype=np.float32)
    return scaler.inverse_transform(np.asarray(X, dtype=np.float64),
                                    **_with_y(scaler, y)).astype(np.float32)


def save_scaler(scaler, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(scaler, f)


def load_scaler(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def scaler_path(model_path: Path) -> Path:
    """Convention: weights at foo.pt keep their scaler at foo_scaler.pkl."""
    return Path(model_path).with_suffix("").with_name(Path(model_path).stem + "_scaler.pkl")


# ─────────────────────────────────────────────────────────────────────────────
# Cohort-level preparation
# ─────────────────────────────────────────────────────────────────────────────
#
# Everything above is *model-level*: fitted on one training split, inverted on
# the way out, invisible to the attacker.  What follows is *cohort-level*: it
# runs once when a cohort is loaded and defines the space every generator,
# attack and metric then calls "raw" -- the analogue of the DESeq2 VST and the
# landmark-gene filter the challenge applied to TCGA before distributing it.
#
# The distinction matters for membership inference.  A cohort-level step that
# estimates something across samples (which genes vary most, say) sees members
# and non-members alike, exactly as the challenge's VST did, so it cannot tell
# them apart; a model-level step sees members only and is part of what leaks.
# Per-sample steps (`cpm`, `log1p`) estimate nothing across samples at all.

def prepare_cohort(df, spec: str | None, labels=None):
    """Apply a `+`-separated chain of cohort-level steps to a samples x genes
    DataFrame.

        cpm            scale each sample to 10^6 total counts (per sample)
        log1p, log2p1  log(1 + x), log2(1 + x)                (per sample)
        dropconst      drop genes with zero variance          (across samples)
        hvg:k          keep the k highest-variance genes      (across samples)
        anova:k        keep the k genes that best separate the classes:
                       largest one-way ANOVA F statistic against `labels`
                                                              (across samples)
        random:k:seed  keep k genes drawn uniformly (the control for the two
                       rules above)
        genes:<file>   keep the genes listed one per line in <file>, in that
                       order, e.g. the 978 L1000 landmark genes; names missing
                       from the cohort raise

    Gene-selecting steps keep the cohort's column order.  `labels` (one class
    per row of `df`) is needed by `anova` only.
    """
    if not spec or spec == "none":
        return df
    for step in spec.split("+"):
        name, _, arg = step.strip().partition(":")
        if name == "cpm":
            df = df.div(df.sum(axis=1).replace(0, 1), axis=0) * 1e6
        elif name == "log1p":
            df = np.log1p(df.clip(lower=0))
        elif name == "log2p1":
            df = np.log2(1 + df.clip(lower=0))
        elif name == "dropconst":
            df = df.loc[:, df.var(axis=0) > 0]
        elif name == "hvg":
            keep = df.var(axis=0).sort_values(ascending=False).index[:int(arg)]
            df = df.loc[:, [g for g in df.columns if g in set(keep)]]
        elif name == "anova":
            if labels is None:
                raise ValueError("anova:k needs the class labels")
            from sklearn.feature_selection import f_classif
            F = np.nan_to_num(f_classif(df.values, np.asarray(labels))[0], nan=0.0)
            keep = set(df.columns[np.argsort(-F, kind="stable")[:int(arg)]])
            df = df.loc[:, [g for g in df.columns if g in keep]]
        elif name == "random":
            k, _, seed = arg.partition(":")
            keep = set(np.random.default_rng(int(seed or 0)).choice(df.columns, int(k), replace=False))
            df = df.loc[:, [g for g in df.columns if g in keep]]
        elif name == "genes":
            wanted = [g.strip() for g in Path(arg).expanduser().read_text().split() if g.strip()]
            missing = [g for g in wanted if g not in df.columns]
            if missing:
                raise KeyError(f"{len(missing)} genes in {arg} are not in the cohort, "
                               f"e.g. {missing[:3]}")
            df = df.loc[:, wanted]
        else:
            raise ValueError(f"Unknown cohort-level step {step!r}")
    return df
