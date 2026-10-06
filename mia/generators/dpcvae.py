"""DP-CVAE target generator: the CAMDA Health Privacy Challenge's `dpcvae`
baseline -- the same CVAE as `cvae`, trained with DP-SGD through Opacus.

Mirrors `src/generators/models/cvae.py` of the challenge starter package with
`generator_name == "dpcvae"`, and takes its defaults from that package's
`dpcvae_config`:

    z_dim 64 (the non-private CVAE uses 128), one-hot condition, beta 0.001,
    Adam 1e-3, batch 64, 10,000 iterations, StandardScaler,
    target epsilon 10, delta 1e-5, per-sample gradient clip (max_norm) 0.1

`PrivacyEngine.make_private_with_epsilon` picks the noise multiplier that
spends exactly (epsilon, delta) over the planned epochs under Poisson
sampling; the multiplier and the epsilon actually spent are in `report()`.

Accounting.  Opacus's default accountant is PRV, which is what the baseline
gets.  PRV's cost grows as the noise shrinks -- calibrating epsilon = 100 takes
a minute and epsilon = 1000 does not finish -- so `accountant="auto"` uses PRV
up to epsilon = 100 and RDP beyond (RDP is slightly conservative: at
epsilon = 10 it asks for noise 3.91 where PRV asks for 3.70).  The accountant
actually used is in `report()`.

What the guarantee does not cover.  The baseline standardises with a
StandardScaler fitted on the training rows and never privatised (its config
has a `preprocessor_eps` entry that the code does not read), and it draws the
synthetic labels from the exact training label counts.  Both are released in
the clear through the synthetic data, so with the default `preprocess` the
generator is DP-SGD-trained but not DP end to end -- the same defect as the
challenge's DP-PGM (`pgg.py`).  `preprocess="fixed:0:24"` scales by public
bounds instead and `private_labels=True` replaces the label counts with a
Gaussian-mechanism histogram paid for out of the budget (`label_budget`, as a
fraction of epsilon, basic composition); with both, `report()` marks the
release end to end.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .. import preprocessing as pp
from .base import register
from .cvae import CVAEGenerator
from .cvae_model import CVAE


@dataclass
class DPCVAEGenerator(CVAEGenerator):
    z_dim: int = 64
    condition_type: str = "onehot"

    epsilon: float = 10.0
    delta: float = 1e-5
    max_grad_norm: float = 0.1
    accountant: str = "auto"        # "prv" | "rdp" | "auto" (see module docstring)
    private_labels: bool = False
    label_budget: float = 0.05

    name = "dpcvae"
    requires = {"opacus": "1.0", "torch": "2.0"}
    env = "sota"

    def __post_init__(self):
        super().__post_init__()
        self._report = {}

    def fit(self, X: np.ndarray, y: np.ndarray, n_classes: int) -> "DPCVAEGenerator":
        from opacus import PrivacyEngine

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        self.n_classes = int(n_classes)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        eps_train = self.epsilon
        if self.private_labels:
            eps_lab = self.label_budget * self.epsilon
            eps_train = self.epsilon - eps_lab
            # Gaussian mechanism on the label histogram; add/remove-one
            # neighbours change one count by 1, so L2 sensitivity is 1.
            from opacus.accountants.utils import get_noise_multiplier
            sigma = get_noise_multiplier(target_epsilon=eps_lab, target_delta=self.delta / 2,
                                         sample_rate=1.0, steps=1, accountant="rdp")
            rng = np.random.default_rng(self.seed + 7)
            counts = np.bincount(y, minlength=self.n_classes) + rng.normal(0, sigma, self.n_classes)
            p = np.clip(counts, 0, None)
            p = p / p.sum() if p.sum() > 0 else np.full(self.n_classes, 1 / self.n_classes)
            self._train_labels = rng.choice(self.n_classes, size=len(y), p=p)
            delta_train = self.delta / 2
        else:
            self._train_labels = y.copy()
            delta_train = self.delta

        self.scaler, X_scaled = pp.fit_scaler(self.preprocess, X, y)
        loader = DataLoader(TensorDataset(torch.tensor(X_scaled), torch.tensor(y)),
                            batch_size=self.batch_size, shuffle=True)
        model = CVAE(x_dim=X_scaled.shape[1], y_dim=self.n_classes, z_dim=self.z_dim,
                     beta=self.beta, transform=self.transform,
                     condition_type=self.condition_type,
                     disease_embed_dim=self.disease_embed_dim).to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        epochs = self._n_epochs(len(X))

        accountant = self.accountant
        if accountant == "auto":
            accountant = "prv" if self.epsilon <= 100 else "rdp"
        engine = PrivacyEngine(accountant=accountant)
        model, opt, loader = engine.make_private_with_epsilon(
            module=model, optimizer=opt, data_loader=loader,
            target_epsilon=eps_train, target_delta=delta_train,
            max_grad_norm=self.max_grad_norm, epochs=epochs)

        for epoch in range(epochs):
            total, seen = 0.0, 0
            for xb, yb in loader:
                if len(xb) == 0:        # Poisson sampling can draw an empty batch
                    continue
                xb = xb.to(self.device)
                y_vec = model._module.encode_condition(
                    yb.to(self.device), condition_mode=self.condition_mode)
                model.train()
                model.zero_grad()
                loss = model._module.compute_loss(xb, y_vec)["loss"]
                loss.backward()
                opt.step()
                opt.zero_grad()
                total += loss.item() * len(xb)
                seen += len(xb)
            if self.verbose and (epoch % 50 == 0 or epoch == epochs - 1):
                print(f"    [dpcvae] epoch {epoch:4d}/{epochs}  "
                      f"loss={total / max(seen, 1):.6f}", flush=True)

        self.model = model._module
        self.model.eval()
        data_indep = pp.describe(self.preprocess)["data_independent"]
        self._report = {
            "mechanism": "DP-SGD (Opacus), Poisson sampling",
            "accountant": accountant,
            "noise_multiplier": float(opt.noise_multiplier),
            "epsilon_spent_training": float(engine.get_epsilon(delta_train)),
            "epsilon_target": float(self.epsilon), "delta": float(self.delta),
            "epochs": int(epochs), "sample_rate": float(1 / len(loader)),
            "preprocessing_private": bool(data_indep),
            "labels_private": bool(self.private_labels),
            "dp_end_to_end": bool(data_indep and self.private_labels),
        }
        return self

    def report(self) -> dict:
        return dict(self._report)


register(DPCVAEGenerator)
