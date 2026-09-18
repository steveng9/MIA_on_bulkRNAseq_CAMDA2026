"""NoisyDiffusion target generator: training, DDPM sampling, checkpointing.

Wraps the vendored `EmbeddedDiffusion` / `DiffusionTrainer` with the blue
team's training recipe: SMOTE-balance the classes, quantile-normalise, then
200 epochs of one-cycle AdamW with early stopping.

Two details are load-bearing for the attack:

* `dp_noise_multiplier` defaults to the blue team's 1e-5.  At that scale the
  Gaussian perturbation is far below gradient noise and buys no meaningful
  privacy -- there is no epsilon accounting upstream either -- so the "DP" in
  this generator should not be read as a formal guarantee.  Shadow models
  default to 0 because matching an inconsequential noise term costs compute for
  nothing.
* `unconditional=True` collapses the label space to a single dummy class.
  MeLoMIA trains its shadows that way: the adversary has no reliable subtype
  labels for the released synthetic data, and conditioning on guessed labels
  injects noise into the very losses the attack is measuring.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .. import preprocessing as pp
from .base import Generator, register
from .nd_model import DiffusionTrainer, EmbeddedDiffusion, generate_samples

DUMMY_LABEL = 0


@dataclass
class NDGenerator(Generator):
    # architecture
    input_dim: int = 978
    num_timesteps: int = 1000
    hidden_dims: tuple = (2048, 2048)
    dropout: float = 0.2
    time_embedding_dim: int = 128
    label_embedding_dim: int = 64
    attn_num_heads: int = 0
    attn_num_tokens: int = 64
    num_groups: int = 8

    # noise schedule
    beta_schedule: str = "linear"
    linear_beta_start: float = 0.001
    linear_beta_end: float = 0.02
    cosine_s: float = 0.008
    power_sigma_max: float = 1.0
    power_sigma_min: float = 0.005
    power_rho_expo: float = 7.0

    # optimisation
    norm_method: str = "quantile"
    epochs: int = 200
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-3
    lr_pct_start: float = 0.2
    lr_div_factor: float = 25
    lr_final_div_factor: float = 25
    lr_anneal_strategy: str = "cos"
    early_stopping: bool = True
    patience: int = 30
    min_delta: float = 1e-4

    # privacy knobs (see module docstring)
    dp_noise_multiplier: float = 1e-5
    max_grad_norm: float = 1.0

    # class handling
    smote_upsample_to: int | None = 3000
    unconditional: bool = False

    verbose: bool = True

    name = "nd"

    def __post_init__(self):
        self.model: EmbeddedDiffusion | None = None
        self.diffusion: DiffusionTrainer | None = None
        self.scaler = None
        self.n_classes: int | None = None
        self._label_counts: dict | None = None

    # ── Presets ─────────────────────────────────────────────────────────────

    @staticmethod
    def preset(kind: str, **overrides) -> "NDGenerator":
        base = dict(
            # faithful to the blue team; used for targets and base shadows
            target=dict(epochs=200, dp_noise_multiplier=1e-5, smote_upsample_to=3000,
                        unconditional=False),
            # unconditional, no DP noise; used for synth-shadows and the proxy
            shadow=dict(epochs=200, dp_noise_multiplier=0.0, smote_upsample_to=None,
                        unconditional=True),
        )
        if kind not in base:
            raise KeyError(f"Unknown ND preset {kind!r}. Known: {sorted(base)}")
        return NDGenerator(**{**base[kind], **overrides})

    # ── Model construction ──────────────────────────────────────────────────

    def _build(self, n_classes: int) -> EmbeddedDiffusion:
        return EmbeddedDiffusion(
            input_dim=self.input_dim,
            num_classes=n_classes,
            num_timesteps=self.num_timesteps,
            hidden_dims=list(self.hidden_dims),
            dropout=self.dropout,
            attn_num_tokens=self.attn_num_tokens,
            attn_num_heads=self.attn_num_heads,
            time_embedding_dim=self.time_embedding_dim,
            label_embedding_dim=self.label_embedding_dim,
            num_groups=self.num_groups,
        ).to(self.device)

    def _build_diffusion(self) -> DiffusionTrainer:
        return DiffusionTrainer(
            num_timesteps=self.num_timesteps,
            beta_schedule=self.beta_schedule,
            linear_beta_start=self.linear_beta_start,
            linear_beta_end=self.linear_beta_end,
            cosine_s=self.cosine_s,
            power_sigma_max=self.power_sigma_max,
            power_sigma_min=self.power_sigma_min,
            power_rho_expo=self.power_rho_expo,
            device=self.device,
        )

    # ── Training ────────────────────────────────────────────────────────────

    def _smote(self, X, y):
        """Upsample minority classes before scaling.

        Order matters: SMOTE interpolates between nearest neighbours, which is
        only meaningful while the axes are still raw expression values.  Fitting
        the quantile transform first would warp those distances.
        """
        if self.smote_upsample_to is None or self.unconditional:
            return X, y
        from imblearn.over_sampling import SMOTE

        counts = Counter(y.tolist())
        strategy = {c: self.smote_upsample_to for c, n in counts.items()
                    if n < self.smote_upsample_to}
        if not strategy:
            return X, y
        k = min(5, min(counts.values()) - 1)
        if k < 1:
            return X, y
        smote = SMOTE(sampling_strategy=strategy, random_state=self.seed, k_neighbors=k)
        X_res, y_res = smote.fit_resample(X, y)
        if self.verbose:
            print(f"    [nd] SMOTE {X.shape} -> {X_res.shape}", flush=True)
        return X_res.astype(np.float32), y_res.astype(np.int64)

    def fit(self, X: np.ndarray, y: np.ndarray, n_classes: int) -> "NDGenerator":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        self._label_counts = dict(Counter(y.tolist()))
        self.n_classes = 1 if self.unconditional else int(n_classes)

        X_fit, y_fit = self._smote(X, y)
        if self.unconditional:
            y_fit = np.full(len(X_fit), DUMMY_LABEL, dtype=np.int64)

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.scaler, X_scaled = pp.fit_scaler(self.norm_method, X_fit)
        loader = DataLoader(
            TensorDataset(torch.tensor(X_scaled), torch.tensor(y_fit)),
            batch_size=self.batch_size, shuffle=True,
        )

        self.model = self._build(self.n_classes)
        self.diffusion = self._build_diffusion()
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=self.lr, epochs=self.epochs, steps_per_epoch=len(loader),
            pct_start=self.lr_pct_start, anneal_strategy=self.lr_anneal_strategy,
            div_factor=self.lr_div_factor, final_div_factor=self.lr_final_div_factor,
        )

        best, no_improve, best_state = float("inf"), 0, None
        for epoch in range(self.epochs):
            total = 0.0
            for xb, yb in loader:
                total += self.diffusion.train_step(
                    self.model, opt, xb, yb, self.device,
                    dp_noise_multiplier=self.dp_noise_multiplier,
                    max_grad_norm=self.max_grad_norm,
                )
                sched.step()
            avg = total / len(loader)

            if self.verbose and (epoch % 25 == 0 or epoch == self.epochs - 1):
                print(f"    [nd] epoch {epoch:4d}/{self.epochs}  loss={avg:.6f}", flush=True)

            if self.early_stopping:
                if avg < best - self.min_delta:
                    best, no_improve = avg, 0
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in self.model.state_dict().items()}
                else:
                    no_improve += 1
                if no_improve >= self.patience:
                    if self.verbose:
                        print(f"    [nd] early stop @ epoch {epoch}", flush=True)
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)
        self.model.eval()
        return self

    # ── Sampling ────────────────────────────────────────────────────────────

    def sample(self, n: int, chunk: int = 600) -> tuple:
        """Reverse-diffuse `n` samples, preserving the training class mix."""
        counts = self._label_counts or {DUMMY_LABEL: n}
        total = sum(counts.values())
        parts_X, parts_y = [], []
        for label, count in counts.items():
            want = max(1, int(round(n * count / total)))
            gen_label = DUMMY_LABEL if self.unconditional else int(label)
            while want > 0:
                k = min(want, chunk)
                X = generate_samples(
                    model=self.model, diffusion_trainer=self.diffusion,
                    num_samples=k, input_dim=self.input_dim, label=gen_label,
                    device=self.device, scaler=self.scaler,
                )
                parts_X.append(np.asarray(X, dtype=np.float32))
                parts_y.append(np.full(k, int(label), dtype=np.int64))
                want -= k

        X = np.concatenate(parts_X)
        y = np.concatenate(parts_y)
        rng = np.random.default_rng(self.seed)
        order = rng.permutation(len(X))[:n]
        return X[order], y[order]

    # ── Persistence ─────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": self.model.state_dict(), "n_classes": self.n_classes,
             "label_counts": self._label_counts, "unconditional": self.unconditional},
            path,
        )
        pp.save_scaler(self.scaler, pp.scaler_path(path))

    def load(self, path: Path) -> "NDGenerator":
        path = Path(path)
        ckpt = torch.load(path, map_location=self.device)
        self.n_classes = ckpt["n_classes"]
        self._label_counts = ckpt.get("label_counts")
        self.unconditional = ckpt.get("unconditional", self.unconditional)
        self.model = self._build(self.n_classes)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.diffusion = self._build_diffusion()
        self.scaler = pp.load_scaler(pp.scaler_path(path))
        return self


register(NDGenerator)
