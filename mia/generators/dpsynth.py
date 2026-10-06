"""DPSynth target generators: Google's library of marginal-based DP synthesis
(github.com/google/dpsynth, McKenna et al.) -- select low-dimensional
marginals, measure them with Gaussian noise, fit a graphical model with
Private-PGM, sample.  `mechanism` picks how the marginals are chosen:

    mst          private maximum spanning tree over all column pairs
                 (McKenna et al. 2021, winner of the NIST 2018 contest)
    aim          adaptive, workload-aware selection (McKenna et al. 2022)
    swift        DPSynth's scalable workload-informed factor tree
    independent  one-way marginals only: every gene independent, no label link
    star         every (gene, label) pair, nothing else -- DPSynth's `Direct`
                 mechanism on the clique set of our own DP-PGM (`pgm`), so the
                 table-selection question can be asked inside one library

The library is used as shipped (`TabularConfig`, its in-memory path), with its
defaults: 32 bins per numerical column, 10% of the budget on per-column
initialisation, calibration through `dpsynth.calibrate(epsilon, delta)`.
DPSynth calibrates with both an RDP and a PLD accountant and keeps the tighter;
past 64 columns only RDP is run (`accountant="auto"`), because its PLD pass
takes over an hour at 978 genes and RDP was the tighter one where both ran.

Unlike the challenge's DP-PGM (`pgg`) and DP-CVAE baselines, this pipeline is
DP end to end provided the bounds are public: DPSynth discretises each
numerical column itself with a DP quantile tree paid for out of the budget, and
the number of released rows is taken from a noisy total.  The price is that 978
genes share the initialisation budget, so at small epsilon the bins are poor.
`bounds="public"` uses [lo, hi] for every gene (DESeq2-VST values lie in
roughly 0-24); `bounds="data"` uses each gene's training min and max, which is
tighter and no longer private (`report()` says so).

`interval_handling="sample"` draws uniformly inside the released bin;
"midpoint" returns its centre, leaving at most `numerical_bins` distinct values
per gene.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import numpy as np

from .base import Generator, register


@contextlib.contextmanager
def _quiet_estimation():
    """Switch off mbi's progress table while a mechanism runs.

    Every 50 iterations mbi logs losses including a "primal feasibility" check
    over all pairs of cliques that share an attribute.  In a star every clique
    shares the label, so that is ~478,000 pairs at 978 genes and its JIT
    compilation does not finish (>10 min already at 100 genes).  The table is
    a diagnostic only; the fitted model is identical without it.
    """
    import mbi
    original = mbi.callbacks.default
    mbi.callbacks.default = lambda *a, **k: (lambda marginals: None)
    try:
        yield
    finally:
        mbi.callbacks.default = original


@dataclass
class DPSynthGenerator(Generator):
    mechanism: str = "mst"
    epsilon: float = 10.0
    delta: float = 1e-5

    numerical_bins: int = 32
    init_budget_fraction: float = 0.1
    bounds: str = "public"            # "public" | "data"
    lo: float = 0.0
    hi: float = 24.0
    interval_handling: str = "sample"

    accountant: str = "auto"          # "auto" | "rdp" | "pld" | "both"
    oracle: str = "default"           # mbi marginal oracle: "default" | "hugin" | "shafer_shenoy" | "implicit"
    pgm_iters: int | None = None      # None: the mechanism's own default
    max_model_size: int = 80          # AIM only (megabytes)

    name = "dpsynth"
    requires = {"dpsynth": "0.4"}
    env = "sota"

    def __post_init__(self):
        self._result = None
        self._report = {}

    def _mechanism_config(self, genes, label):
        from dpsynth import discrete_mechanisms as dm
        it = {} if self.pgm_iters is None else {"pgm_iters": int(self.pgm_iters)}
        if self.oracle != "default":
            import mbi
            it["marginal_oracle"] = getattr(mbi.marginal_oracles, f"message_passing_{self.oracle}")
        m = self.mechanism
        if m == "mst":
            return dm.MSTConfig(**it)
        if m == "aim":
            return dm.AIMConfig(max_model_size=self.max_model_size, **it)
        if m == "swift":
            return dm.SWIFTConfig(**it)
        if m == "independent":
            return dm.IndependentConfig(**it)
        if m == "star":
            return dm.DirectConfig(
                prespecified_marginal_queries=[(g, label) for g in genes], **it)
        raise ValueError(f"Unknown DPSynth mechanism {m!r}")

    # Above this many columns the PLD accountant is not run: DPSynth composes
    # ~7 events per numerical column one by one, which is ~3 s per column per
    # calibration (over an hour at 978 genes).  RDP calibrates in seconds.
    PLD_MAX_COLUMNS = 64

    def _accountant_fn(self, acct):
        import functools
        import dp_accounting
        if acct == "both":            # DPSynth's default: the tighter of the two
            return None
        if acct == "rdp":
            return dp_accounting.rdp.RdpAccountant
        if acct == "pld":
            return functools.partial(
                dp_accounting.pld.PLDAccountant,
                value_discretization_interval=min(1e-4, 1e-1 * self.epsilon))
        raise ValueError(f"accountant must be auto, rdp, pld or both, not {acct!r}")

    def fit(self, X: np.ndarray, y: np.ndarray, n_classes: int) -> "DPSynthGenerator":
        import dpsynth
        import pandas as pd

        X = np.asarray(X, dtype=np.float64)
        genes = [f"g{i}" for i in range(X.shape[1])]
        self._genes, self._label = genes, "label"
        df = pd.DataFrame(X, columns=genes)
        df[self._label] = np.asarray(y, dtype=int)

        if self.bounds == "public":
            lo, hi = np.full(X.shape[1], self.lo), np.full(X.shape[1], self.hi)
        elif self.bounds == "data":
            lo, hi = X.min(axis=0), X.max(axis=0)
        else:
            raise ValueError(f"bounds must be 'public' or 'data', not {self.bounds!r}")
        schema = {g: dpsynth.NumericalAttribute(
                      min_value=float(lo[j]), max_value=float(hi[j]),
                      interval_handling=self.interval_handling)
                  for j, g in enumerate(genes)}
        schema[self._label] = dpsynth.CategoricalAttribute(
            possible_values=list(range(int(n_classes))))

        config = dpsynth.TabularConfig(
            discrete_mechanism=self._mechanism_config(genes, self._label),
            numerical_bins=self.numerical_bins,
            init_budget_fraction=self.init_budget_fraction)
        acct = self.accountant
        if acct == "auto":
            acct = "rdp" if X.shape[1] > self.PLD_MAX_COLUMNS else "both"
        mech = dpsynth.calibrate(config, schema, epsilon=self.epsilon, delta=self.delta,
                                 accountant_fn=self._accountant_fn(acct))
        self._rng = np.random.default_rng(self.seed)
        with _quiet_estimation():
            self._result = mech(self._rng, df)

        model = self._result.discrete_mechanism_result.model
        cliques = [tuple(c) for c in getattr(model, "cliques", [])]
        self._report = {
            "mechanism": f"DPSynth {self.mechanism} (marginals + Private-PGM)",
            "epsilon": float(self.epsilon), "delta": float(self.delta),
            "accounting": f"dpsynth.calibrate ({acct})",
            "n_cliques": len(cliques),
            "n_label_cliques": sum(self._label in c for c in cliques),
            "max_clique_size": max((len(c) for c in cliques), default=0),
            "noisy_total": float(model.total),
            "preprocessing_private": self.bounds == "public",
            "labels_private": True,
            "dp_end_to_end": self.bounds == "public",
        }
        return self

    def sample(self, n: int) -> tuple:
        from dpsynth.discrete_mechanisms import common as dm_common
        res = self._result
        discrete = dm_common.generate_synthetic_data(
            res.discrete_mechanism_result.model, self._rng, rows=int(n))
        df = res.codec.decode(discrete, self._rng)
        X = df[self._genes].to_numpy(dtype=np.float32)
        y = df[self._label].astype(int).to_numpy(dtype=np.int64)
        return X, y

    def report(self) -> dict:
        return dict(self._report)


register(DPSynthGenerator)
