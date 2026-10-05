"""MeLoMIA-E2E -- synth-shadows, proxy and classifier trained as one module.

In MeLoMIA the synth-shadows and the proxy are fitted with a generative loss
and nothing else; the classifier then sees whatever membership signal that
overfitting happened to leave in their losses.  Here the classifier's loss also
decides *how* those models are fitted (Steven's idea, 2026-10-05).

The obvious construction does not work: letting the membership gradient into
synth-shadow k's own weights lets them memorise shadow k's labels, and the
proxy -- which has no labels -- would then look nothing like them.  It is the
real-data-shadow failure again (cross-validation near 1, deployment near 0.69).
So the membership gradient is only allowed to reach weights the proxy shares
with the shadows:

    internal synthetic k (from base shadow k, exactly as in MeLoMIA)
            |
            v   T generative-loss steps, learned step sizes
    p0  ------------------------------------------>  synth-shadow k
     |                                                    |
     +--- loss signature of the candidates under p0       +--- ... under shadow k
                              \                          /
                               head ( signature_k , signature_k - signature_0 )
                                              |
                                   membership loss of shadow k
                                              |
                  backpropagated into head, p0 and the step sizes

`p0` is a shared initialisation: every synth-shadow, and at inference the
proxy, is `p0` adapted to its own synthetic data for the same T steps.  Tuning
`p0` and the step sizes by the membership loss therefore tunes how every one of
those models overfits, and the proxy inherits it without ever needing a label.
`p0` also serves as each record's difficulty reference -- the head sees the
*change* in a record's signature caused by adaptation -- which is the job
per-record calibration does in MeLoMIA.

The base shadows and their internal synthetic data are MeLoMIA-CVAE's, read
from (or built into) that attack's cache, so the two attacks differ only in
what happens after the internal synthetic data exists.

The last `n_holdout` shadows play the internal-proxy role: they never give a
membership gradient, are adapted and scored as the proxy will be, and select
the training step that is kept.  `joint=False` freezes `p0` and the step sizes
and trains the head alone -- the control for the joint training.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from ... import datasets as D
from ... import paths
from ... import targets as TG
from ..base import Attack, register
from ..melomia import MeLoMIACVAE
from .module import E2EConfig, E2EModule


@dataclass
class MeLoMIAE2E(Attack):
    n_shadows: int = 30
    n_holdout: int = 6                  # internal proxies, the last shadows of the stack
    base_generator: str | None = None   # as in MeLoMIA: family of the base shadows
    stack_noise: int = 50               # only locates MeLoMIA-CVAE's cache (its `n_noise`)
    joint: bool = True
    #: overrides of `module.E2EConfig` (adaptation length, outer steps, ...)
    module_params: dict = field(default_factory=dict)
    label: str = ""

    name: str = field(init=False, default="melomia_e2e_cvae")

    def __post_init__(self):
        self._module: dict = {}

    # ── Identity / paths ────────────────────────────────────────────────────

    def config(self) -> E2EConfig:
        return E2EConfig(**{"seed": self.seed, **self.module_params})

    def tag(self) -> str:
        c = self.config()
        t = f"k{self.n_shadows}h{self.n_holdout}_T{c.inner_steps}"
        if c.grad_from:
            t += f"g{c.grad_from}"
        if self.base_generator:
            t += f"_base{self.base_generator}"
        t += "_joint" if self.joint else "_frozen"
        extra = {k: v for k, v in sorted(self.module_params.items())
                 if k not in ("inner_steps", "grad_from")}
        if extra:
            t += "_" + "_".join(f"{k}{v}" for k, v in extra.items())
        if self.label:
            t += f"_{self.label}"
        return t

    def cache(self, dataset: str) -> Path:
        return paths.attack_cache(self.name, dataset, self.tag())

    def _stack(self) -> MeLoMIACVAE:
        """The MeLoMIA-CVAE attack whose base shadows this one reuses."""
        return MeLoMIACVAE(n_shadows=self.n_shadows, n_noise=self.stack_noise,
                           base_generator=self.base_generator, seed=self.seed,
                           device=self.device, verbose=self.verbose)

    def _say(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    # ── Data ────────────────────────────────────────────────────────────────

    def _candidates(self, dataset: str) -> tuple:
        """Candidates, and the scaler every model of the module shares.

        Fitted on the candidate records themselves, which the adversary holds;
        no label is involved.  One scaler for the shadows and the proxy keeps
        a record's input identical under every model, which the comparison
        against `p0` relies on.
        """
        X = D.load_expression(dataset).values.astype(np.float32)
        mean, sd = X.mean(axis=0), X.std(axis=0) + 1e-6
        return X, mean, sd

    def _tensor(self, X, mean, sd) -> torch.Tensor:
        return torch.tensor((np.asarray(X, dtype=np.float32) - mean) / sd, device=self.device)

    # ── Prepare ─────────────────────────────────────────────────────────────

    def prepare(self, dataset: str) -> E2EModule:
        if dataset in self._module:
            return self._module[dataset]
        X, mean, sd = self._candidates(dataset)
        module = E2EModule(X.shape[1], self.config(), self.device)
        state_path = self.cache(dataset) / "module.pt"
        if state_path.exists():
            module.load_state(torch.load(state_path, map_location=self.device))
            self._module[dataset] = module
            return module

        self._say(f"[melomia_e2e] preparing {dataset} (K={self.n_shadows}, "
                  f"{self.n_holdout} held out, tag={self.tag()})")
        stack = self._stack()
        stack._ensure_splits(dataset)
        stack._ensure_internal_synth(dataset)
        synth, labels = [], []
        for k in range(1, self.n_shadows + 1):
            d = np.load(stack._internal_synth_path(dataset, k))
            synth.append(self._tensor(d["X"], mean, sd))
            labels.append(stack._shadow_membership(dataset, k))
        train = list(range(self.n_shadows - self.n_holdout))
        held = list(range(self.n_shadows - self.n_holdout, self.n_shadows))

        summary = module.fit(synth, labels, self._tensor(X, mean, sd), train, held,
                             joint=self.joint, say=self._say)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(module.state(), state_path)
        (self.cache(dataset) / "training.json").write_text(json.dumps(
            {"summary": summary, "history": module.history, "config": asdict(self.config()),
             "train_shadows": [k + 1 for k in train], "held_out_shadows": [k + 1 for k in held]},
            indent=2))
        self._say(f"  [melomia_e2e] held-out shadow AUC {summary['best_auc']:.4f} "
                  f"(frozen {summary['frozen_auc']:.4f}, kept step {summary['best_step']})")
        self._module[dataset] = module
        return module

    # ── Score ───────────────────────────────────────────────────────────────

    def score(self, dataset: str, generator: str, split: int) -> np.ndarray:
        module = self.prepare(dataset)
        X, mean, sd = self._candidates(dataset)
        target = TG.load_target(dataset, generator, split)
        # The proxy: p0 adapted to the released synthetic data, as every
        # synth-shadow was adapted to its internal synthetic data.
        return module.logits(self._tensor(target["X"], mean, sd),
                             self._tensor(X, mean, sd)).astype(np.float64)


register(MeLoMIAE2E)
