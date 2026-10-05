"""The single module of MeLoMIA-E2E and its training loop.

Three groups of weights, trained by one membership loss:

    p0       the probe's shared initialisation.  Every synth-shadow, and at
             inference the proxy, is `p0` adapted to its own synthetic dataset
             by the generative loss alone.
    log_lr   one adaptation step size per weight tensor.
    head     the classifier, reading the candidates' loss signatures under the
             adapted model and their change from the signatures under `p0`.

The membership loss is backpropagated through the head, through the readout
and through the unrolled adaptation steps into `p0` and `log_lr`.  No weight
that belongs to a single shadow ever receives a membership gradient: a shadow's
own weights are a function of (p0, log_lr, its synthetic data) and nothing
else, which is exactly what the proxy's will be.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from . import core as C


@dataclass
class E2EConfig:
    z_dim: int = 128
    beta: float = 0.001
    # adaptation (inner loop)
    inner_steps: int = 50
    inner_lr: float = 1e-3            # initial step size of every tensor
    inner_optimizer: str = "adam"
    inner_batch: int | None = None    # None = full batch
    second_order: bool = True
    grad_from: int = 0                # truncate backpropagation before this step
    # readout
    n_draws: int = 8
    # shared initialisation
    pretrain_epochs: int = 40
    # joint training (outer loop)
    head_warmup_epochs: int = 60
    outer_steps: int = 300
    shadows_per_step: int = 4
    lr_init: float = 3e-5
    lr_inner_lr: float = 1e-2
    lr_head: float = 1e-3
    head_hidden: int = 256
    head_dropout: float = 0.3
    head_weight_decay: float = 1e-3
    #: fraction of a shadow's synthetic rows it is adapted to in each joint
    #: step (a fresh random subset every time); 1.0 uses them all.  Makes a
    #: synthetic dataset harder to recognise, and so its labels harder to
    #: memorise.
    train_subsample: float = 1.0
    grad_clip: float = 1.0
    eval_every: int = 25
    seed: int = 42


def _say(msg: str) -> None:
    print(msg, flush=True)


class Head(nn.Module):
    def __init__(self, d_in: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 4), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 4, 1),
        )

    def forward(self, x):
        return self.net(x.clamp(-10.0, 10.0)).squeeze(-1)


class E2EModule:
    """Shared initialisation + learned adaptation + readout + head."""

    def __init__(self, x_dim: int, cfg: E2EConfig, device: str = "cuda"):
        self.cfg, self.device, self.x_dim = cfg, device, x_dim
        self.p0 = C.init_params(x_dim, cfg.z_dim, cfg.seed, device)
        self.log_lr = {k: torch.tensor(math.log(cfg.inner_lr), device=device, requires_grad=True)
                       for k in self.p0}
        torch.manual_seed(cfg.seed + 1)
        self.head = Head(2 * C.readout_dim(cfg.z_dim), cfg.head_hidden, cfg.head_dropout).to(device)
        g = torch.Generator(device="cpu").manual_seed(cfg.seed + 2)
        self.bank = torch.randn(cfg.n_draws, cfg.z_dim, generator=g).to(device)
        self.history: list = []

    # ── Pieces ──────────────────────────────────────────────────────────────

    def _eps(self, n_rows: int):
        rows = n_rows if self.cfg.inner_batch is None else min(self.cfg.inner_batch, n_rows)
        return C.noise_bank(self.cfg.inner_steps, rows, self.cfg.z_dim, self.cfg.seed + 3, self.device)

    def features(self, x_syn, x_cand, create_graph: bool) -> torch.Tensor:
        """Head input for every candidate, for the model adapted to `x_syn`."""
        cfg = self.cfg
        with torch.enable_grad():
            pk = C.adapt(self.p0, self.log_lr, x_syn, self._eps(len(x_syn)), cfg.beta,
                         create_graph=create_graph and cfg.second_order,
                         batch=cfg.inner_batch, optimizer=cfg.inner_optimizer,
                         grad_from=cfg.grad_from)
            if create_graph and not cfg.second_order:
                # first order: the adapted weights move with p0 one-for-one
                pk = {k: self.p0[k] + (pk[k] - self.p0[k]).detach() for k in pk}
            ctx = torch.enable_grad() if create_graph else torch.no_grad()
            with ctx:
                Fk = C.readout(pk, x_cand, self.bank)
                F0 = C.readout(self.p0, x_cand, self.bank)
                return torch.cat([C.standardise(Fk), C.standardise(Fk - F0)], dim=1)

    def logits(self, x_syn, x_cand) -> np.ndarray:
        """Membership logit of every candidate against one synthetic dataset."""
        self.head.eval()
        f = self.features(x_syn, x_cand, create_graph=False).detach()
        with torch.no_grad():
            return self.head(f).cpu().numpy()

    # ── Shared initialisation ───────────────────────────────────────────────

    def pretrain(self, pool: torch.Tensor) -> None:
        """Fit `p0` to the pooled synthetic data of the training shadows.

        The starting point of joint training: a model of what synthetic data
        from this generator family looks like in general, so that adapting it
        to one dataset is a small, readable change.
        """
        cfg = self.cfg
        opt = torch.optim.Adam(self.p0.values(), lr=1e-3)
        g = torch.Generator(device="cpu").manual_seed(cfg.seed + 4)
        for _ in range(cfg.pretrain_epochs):
            perm = torch.randperm(len(pool), generator=g).to(self.device)
            for i in range(0, len(pool), 256):
                xb = pool[perm[i:i + 256]]
                eps = torch.randn(len(xb), cfg.z_dim, generator=g).to(self.device)
                loss = C.generative_loss(self.p0, xb, eps, cfg.beta)
                opt.zero_grad()
                loss.backward()
                opt.step()

    # ── Training ────────────────────────────────────────────────────────────

    def _bce(self, logit, y):
        # members are 80% of every shadow; weight the classes equally
        w = torch.where(y > 0.5, 0.5 / y.mean(), 0.5 / (1 - y.mean()))
        return nn.functional.binary_cross_entropy_with_logits(logit, y, weight=w)

    def evaluate(self, synth: list, labels: list, x_cand, which: list) -> float:
        return float(np.mean([C.auc(labels[k], self.logits(synth[k], x_cand)) for k in which]))

    def warmup_head(self, synth, labels, x_cand, train: list, say=_say) -> None:
        """Train the head alone on features of the frozen initial module."""
        cfg = self.cfg
        feats = torch.cat([self.features(synth[k], x_cand, False).detach() for k in train])
        y = torch.tensor(np.concatenate([labels[k] for k in train]), dtype=torch.float32,
                         device=self.device)
        opt = torch.optim.AdamW(self.head.parameters(), lr=cfg.lr_head,
                                weight_decay=cfg.head_weight_decay)
        g = torch.Generator(device="cpu").manual_seed(cfg.seed + 5)
        self.head.train()
        for _ in range(cfg.head_warmup_epochs):
            perm = torch.randperm(len(feats), generator=g).to(self.device)
            for i in range(0, len(feats), 512):
                idx = perm[i:i + 512]
                loss = self._bce(self.head(feats[idx]), y[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()

    def state(self) -> dict:
        return {"p0": {k: v.detach().clone() for k, v in self.p0.items()},
                "log_lr": {k: v.detach().clone() for k, v in self.log_lr.items()},
                "head": {k: v.detach().clone() for k, v in self.head.state_dict().items()}}

    def load_state(self, s: dict) -> None:
        for k in self.p0:
            self.p0[k].data.copy_(s["p0"][k])
            self.log_lr[k].data.copy_(s["log_lr"][k])
        self.head.load_state_dict(s["head"])

    def fit(self, synth: list, labels: list, x_cand, train: list, held: list,
            joint: bool = True, say=_say) -> dict:
        """Pretrain, warm the head up, then train everything jointly.

        `held` shadows never contribute a membership gradient.  They are
        adapted and scored exactly as the proxy will be (the internal-proxy
        role), and the state with the best AUC on them is the one kept.
        """
        cfg = self.cfg
        t0 = time.time()
        self.pretrain(torch.cat([synth[k] for k in train]))
        self.warmup_head(synth, labels, x_cand, train, say)
        best = self.evaluate(synth, labels, x_cand, held)
        best_state, best_step = self.state(), 0
        self.history = [{"step": 0, "held_auc": best, "seconds": time.time() - t0}]
        say(f"    [e2e] frozen module (pretrained init, head only): held-out AUC {best:.4f}")
        if not joint or cfg.outer_steps == 0:
            return {"frozen_auc": best, "best_auc": best, "best_step": 0}
        frozen = best

        # A zero learning rate freezes that group (lr_init = lr_inner_lr = 0 is
        # the control in which only the head keeps training).
        opt = torch.optim.Adam([
            {"params": list(self.p0.values()), "lr": cfg.lr_init},
            {"params": list(self.log_lr.values()), "lr": cfg.lr_inner_lr},
            {"params": list(self.head.parameters()), "lr": cfg.lr_head,
             "weight_decay": cfg.head_weight_decay},
        ])
        params = list(self.p0.values()) + list(self.log_lr.values()) + list(self.head.parameters())
        rng = np.random.RandomState(cfg.seed + 6)
        y_t = {k: torch.tensor(labels[k], dtype=torch.float32, device=self.device) for k in train}
        running = []
        for step in range(1, cfg.outer_steps + 1):
            self.head.train()
            opt.zero_grad()
            batch = rng.choice(train, size=min(cfg.shadows_per_step, len(train)), replace=False)
            for k in batch:
                xs = synth[k]
                if cfg.train_subsample < 1.0:
                    keep = rng.permutation(len(xs))[:int(len(xs) * cfg.train_subsample)]
                    xs = xs[torch.as_tensor(keep, device=self.device)]
                f = self.features(xs, x_cand, create_graph=True)
                loss = self._bce(self.head(f), y_t[k]) / len(batch)
                loss.backward()
                running.append(float(loss) * len(batch))
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()
            if step % cfg.eval_every == 0 or step == cfg.outer_steps:
                a = self.evaluate(synth, labels, x_cand, held)
                lrs = np.exp([float(v) for v in self.log_lr.values()])
                self.history.append({"step": step, "held_auc": a, "train_loss": float(np.mean(running)),
                                     "seconds": time.time() - t0})
                say(f"    [e2e] step {step:4d}  train loss {np.mean(running):.4f}  held-out AUC {a:.4f}"
                    f"  inner lr {lrs.min():.1e}..{lrs.max():.1e}  ({time.time() - t0:.0f}s)")
                running = []
                if a > best:
                    best, best_state, best_step = a, self.state(), step
        self.load_state(best_state)
        return {"frozen_auc": frozen, "best_auc": best, "best_step": best_step}
