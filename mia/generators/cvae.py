"""CVAE target generator: training, sampling, checkpointing.

Two hyperparameter roles, both expressed through this one class:

  target / base shadow -- must mirror the blue team exactly, or the synthetic
      data a shadow produces will not look like the data the real target
      produced.  That is the `target` preset: StandardScaler, plain Adam at a
      constant 1e-3, and an epoch count derived from `num_iters`.

  synth-shadow / proxy -- free to fit the synthetic data as tightly as
      possible, since the goal is a sharp membership signal rather than a
      faithful generator.  That is the `sharp` preset: quantile scaling,
      AdamW with one-cycle annealing, and early stopping.

The distinction is the footnote in the CAMDA abstract made concrete.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .. import preprocessing as pp
from .base import Generator, register
from .cvae_model import CVAE


@dataclass
class CVAEGenerator(Generator):
    # architecture
    z_dim: int = 128
    beta: float = 0.001
    transform: str = "none"
    condition_type: str = "embedding"
    disease_embed_dim: int = 20

    # optimisation
    preprocess: str = "standard"
    batch_size: int = 64
    num_iters: int = 10000        # blue team expresses budget in steps, not epochs
    epochs: int | None = None     # set to override num_iters
    lr: float = 1e-3
    weight_decay: float = 0.0
    optimizer: str = "adam"       # "adam" | "adamw"
    scheduler: str = "none"       # "none" | "onecycle"
    early_stopping: bool = False
    patience: int = 50
    min_delta: float = 1e-4

    # conditioning: None/"real" uses the labels given, "none" disables conditioning
    condition_mode: str | None = None


    name = "cvae"

    def __post_init__(self):
        self.model: CVAE | None = None
        self.scaler = None
        self.n_classes: int | None = None
        self._train_labels: np.ndarray | None = None

    # ── Presets ─────────────────────────────────────────────────────────────

    @staticmethod
    def preset(kind: str, **overrides) -> "CVAEGenerator":
        base = dict(
            target=dict(preprocess="standard", optimizer="adam", scheduler="none",
                        lr=1e-3, weight_decay=0.0, early_stopping=False,
                        batch_size=64, num_iters=10000),
            sharp=dict(preprocess="quantile", optimizer="adamw", scheduler="onecycle",
                       lr=1e-3, weight_decay=1e-3, early_stopping=True,
                       patience=50, batch_size=32, epochs=500),
        )
        if kind not in base:
            raise KeyError(f"Unknown CVAE preset {kind!r}. Known: {sorted(base)}")
        return CVAEGenerator(**{**base[kind], **overrides})

    # ── Training ────────────────────────────────────────────────────────────

    def _n_epochs(self, n_samples: int) -> int:
        if self.epochs is not None:
            return int(self.epochs)
        return max(1, (self.num_iters * self.batch_size) // max(n_samples, 1))

    def fit(self, X: np.ndarray, y: np.ndarray, n_classes: int) -> "CVAEGenerator":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        self.n_classes = int(n_classes)
        self._train_labels = y.copy()

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.scaler, X_scaled = pp.fit_scaler(self.preprocess, X)
        loader = DataLoader(
            TensorDataset(torch.tensor(X_scaled), torch.tensor(y)),
            batch_size=self.batch_size, shuffle=True,
        )

        self.model = CVAE(
            x_dim=X.shape[1], y_dim=self.n_classes, z_dim=self.z_dim,
            beta=self.beta, transform=self.transform,
            condition_type=self.condition_type,
            disease_embed_dim=self.disease_embed_dim,
        ).to(self.device)

        if self.optimizer == "adamw":
            opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr,
                                    weight_decay=self.weight_decay)
        else:
            opt = torch.optim.Adam(self.model.parameters(), lr=self.lr,
                                   weight_decay=self.weight_decay)

        epochs = self._n_epochs(len(X))
        sched = None
        if self.scheduler == "onecycle":
            sched = torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=self.lr, epochs=epochs, steps_per_epoch=len(loader),
                pct_start=0.2, anneal_strategy="cos", div_factor=25,
                final_div_factor=25,
            )

        best, no_improve, best_state = float("inf"), 0, None
        for epoch in range(epochs):
            self.model.train()
            total = 0.0
            for xb, yb in loader:
                xb = xb.to(self.device)
                y_vec = self.model.encode_condition(
                    yb.to(self.device), condition_mode=self.condition_mode
                )
                loss = self.model.compute_loss(xb, y_vec)["loss"]
                opt.zero_grad()
                loss.backward()
                opt.step()
                if sched is not None:
                    sched.step()
                total += loss.item() * len(xb)

            avg = total / len(X)
            if self.verbose and (epoch % 50 == 0 or epoch == epochs - 1):
                print(f"    [cvae] epoch {epoch:4d}/{epochs}  loss={avg:.6f}", flush=True)

            if self.early_stopping:
                if avg < best - self.min_delta:
                    best, no_improve = avg, 0
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in self.model.state_dict().items()}
                else:
                    no_improve += 1
                if no_improve >= self.patience:
                    if self.verbose:
                        print(f"    [cvae] early stop @ epoch {epoch}", flush=True)
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)
        self.model.eval()
        return self

    # ── Sampling ────────────────────────────────────────────────────────────

    def sample(self, n: int) -> tuple:
        """Draw `n` samples, reusing the training label distribution."""
        rng = np.random.default_rng(self.seed)
        pool = self._train_labels
        y_syn = pool[rng.integers(0, len(pool), size=n)] if pool is not None \
            else rng.integers(0, self.n_classes, size=n)
        X_scaled = self.model.sample_labels(
            y_syn, condition_mode=self.condition_mode, device=self.device
        )
        return pp.invert_scaler(self.scaler, X_scaled), y_syn.astype(np.int64)

    # ── Persistence ─────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "n_classes": self.n_classes,
                "train_labels": self._train_labels,
            },
            path,
        )
        pp.save_scaler(self.scaler, pp.scaler_path(path))

    def load(self, path: Path) -> "CVAEGenerator":
        path = Path(path)
        ckpt = torch.load(path, map_location=self.device)
        self.n_classes = ckpt["n_classes"]
        self._train_labels = ckpt.get("train_labels")
        self.model = CVAE(
            x_dim=978, y_dim=self.n_classes, z_dim=self.z_dim, beta=self.beta,
            transform=self.transform, condition_type=self.condition_type,
            disease_embed_dim=self.disease_embed_dim,
        ).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.scaler = pp.load_scaler(pp.scaler_path(path))
        return self


register(CVAEGenerator)
