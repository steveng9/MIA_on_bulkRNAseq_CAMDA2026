"""TabSyn target generator (Zhang et al., ICLR 2024): a latent diffusion model
for tables.  A transformer VAE embeds each column as a `d_token`-dimensional
token; an EDM score model is then trained on the flattened latents.

The model code is vendored verbatim from amazon-science/tabsyn
(`tabsyn_vae.py`, `tabsyn_diffusion.py`); this file is the training and
sampling recipe of `tabsyn/vae/main.py`, `tabsyn/main.py` and
`tabsyn/sample.py`, with their constants as the dataclass defaults:

    VAE        2 layers, d_token 4, 1 head, FFN factor 32, Adam 1e-3,
               4000 epochs, beta annealed 1e-2 -> 1e-5 by x0.7 on plateau,
               LR x0.95 on plateau (patience 10), best-validation checkpoint
    diffusion  MLP denoiser of width 1024 on (z - mean) / 2, Adam 1e-3,
               LR x0.9 on plateau (patience 20), up to 10,001 epochs,
               early stop after 500 without improvement, best-loss checkpoint
    sampling   50-step stochastic 2nd-order EDM sampler
    scaling    quantile-normal with n // 30 knots ("quantile_n30")

The class label is the table's one categorical column, exactly as TabSyn
treats a classification target, so labels are *generated* with the expression
rather than drawn from the training frequencies as the CVAE does.

Where this departs from upstream, and why:

  batch_size   upstream uses 4096, i.e. the whole cohort in one batch.  Each
               gene is a token, so self-attention is over 980 tokens and one
               attention map costs batch x 980^2 floats; six transformer layers
               at batch 871 need ~40 GB.  256 fits comfortably.  This raises
               the number of optimiser steps per epoch from 1 to n / 256.
  val_frac     upstream schedules the VAE (LR, beta, checkpoint choice) on the
               dataset's *test* split.  Here that split is the non-member half,
               which a target generator must never see.  With val_frac = 0 the
               schedule runs on the training rows themselves, so that every
               member is trained on; val_frac > 0 holds out that share of the
               members instead, which is closer to upstream but leaves members
               the model never fitted and so dilutes the membership signal.

Two settings beyond upstream (off by default, so `tabsyn` is the upstream
recipe; `tabsyn@diff_schedule=steps,latent_scale=std` is the tuned one).  On
BRCA the upstream VAE reconstructs the data exactly, and all the quality is
lost in the diffusion stage, for two reasons that follow from the cohort being
small (871 rows) and wide (978 genes):

  latent_scale   upstream divides the centred latents by 2, which presumes
                 latents of about unit scale; EDM's preconditioning assumes the
                 data has standard deviation sigma_data = 0.5.  Here the KL
                 weight has been annealed to 1e-5 and the latents have standard
                 deviation ~0.3 per dimension after the division, so the
                 denoiser is asked for the wrong scale and samples come out
                 with 4x the variance.  "std" divides by the measured standard
                 deviation instead, as latent diffusion models do (Rombach et
                 al. 2022, scale factor).
  diff_schedule  upstream halves patience on the per-epoch loss: LR x0.9 after
                 20 epochs without improvement, stop after 500.  With 871 rows
                 and batch 4096 an epoch is ONE gradient step on a loss that is
                 noisy by construction (random noise level per row), so the LR
                 decays to ~1e-7 and training stops after ~1,900 steps.
                 "steps" trains for `diff_steps` steps with cosine decay and
                 exponential weight averaging (Karras et al. 2022).
  class_freq     TabSyn generates the label jointly and under-samples the rare
                 subtypes (4% -> 2%), which costs macro-F1.  "train" keeps
                 generated rows until the release has the training class
                 proportions, as label-conditional generators have by design.
                 Like theirs, the release then reveals the class counts.

  preprocess     upstream's quantile-normal scaling stretches a few genes to
                 about twice their real spread on the way back; clipping at the
                 0.1% / 99.9% quantiles and standardising cuts the per-gene
                 error by a third at the same utility.

`TUNED` below is the four together.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .. import preprocessing as pp
from .base import Generator, register
from .tabsyn_diffusion import MLPDiffusion, Model, sample as edm_sample
from .tabsyn_vae import Decoder_model, Encoder_model, Model_VAE


def vae_loss(X_num, X_cat, Recon_X_num, Recon_X_cat, mu_z, logvar_z):
    """`compute_loss` of tabsyn/vae/main.py (accuracy bookkeeping dropped)."""
    ce_loss_fn = nn.CrossEntropyLoss()
    mse_loss = (X_num - Recon_X_num).pow(2).mean()
    ce_loss = 0
    for idx, x_cat in enumerate(Recon_X_cat):
        ce_loss += ce_loss_fn(x_cat, X_cat[:, idx])
    ce_loss /= max(len(Recon_X_cat), 1)
    temp = 1 + logvar_z - mu_z.pow(2) - logvar_z.exp()
    loss_kld = -0.5 * torch.mean(temp.mean(-1).mean())
    return mse_loss, ce_loss, loss_kld


@dataclass
class TabSynGenerator(Generator):
    preprocess: str = "quantile_n30"

    # VAE (tabsyn/vae/main.py)
    d_token: int = 4
    n_head: int = 1
    factor: int = 32
    num_layers: int = 2
    vae_lr: float = 1e-3
    vae_epochs: int = 4000
    max_beta: float = 1e-2
    min_beta: float = 1e-5
    lambd: float = 0.7
    batch_size: int = 256
    val_frac: float = 0.0

    # diffusion (tabsyn/main.py, tabsyn/sample.py)
    dim_t: int = 1024
    diff_lr: float = 1e-3
    diff_epochs: int = 10001
    diff_batch_size: int = 4096
    diff_patience: int = 500
    sample_steps: int = 50

    # The two diffusion-stage settings that upstream's recipe gets wrong for a
    # small, wide cohort (see the module docstring); "upstream" reproduces it.
    latent_scale: str = "upstream"    # "upstream": (z - mean) / 2 | "std": to sigma_data
    diff_schedule: str = "upstream"   # "upstream": plateau LR + early stop | "steps"
    diff_steps: int = 20000           # gradient steps when diff_schedule = "steps"
    diff_ema: float = 0.999           # weight averaging when diff_schedule = "steps"
    class_freq: str = "generated"     # "generated": as sampled | "train": training proportions
    unconditional: bool = False       # ignore the labels (MeLoMIA probes): one constant class

    name = "tabsyn"

    def __post_init__(self):
        self.vae = self.encoder = self.decoder = self.diffusion = None
        self.scaler = None
        self.n_classes = self.n_genes = None
        self.z_mean = None
        self.z_scale = 2.0
        self.class_counts = None
        self.train_z = None

    # ── Model construction ──────────────────────────────────────────────────

    def _modules(self):
        a = (self.num_layers, self.n_genes, [self.n_classes], self.d_token)
        kw = dict(n_head=self.n_head, factor=self.factor)
        self.vae = Model_VAE(*a, bias=True, **kw).to(self.device)
        self.encoder = Encoder_model(*a, **kw).to(self.device).eval()
        self.decoder = Decoder_model(*a, **kw).to(self.device).eval()
        in_dim = (self.n_genes + 1) * self.d_token
        self.diffusion = Model(denoise_fn=MLPDiffusion(in_dim, self.dim_t),
                               hid_dim=in_dim).to(self.device)

    @torch.no_grad()
    def encode(self, X_scaled: np.ndarray, y: np.ndarray) -> torch.Tensor:
        """Deterministic latents (encoder mean, [CLS] dropped, flattened)."""
        out = []
        for i in range(0, len(X_scaled), self.batch_size):
            xb = torch.as_tensor(X_scaled[i:i + self.batch_size]).float().to(self.device)
            yb = torch.as_tensor(y[i:i + self.batch_size]).long().view(-1, 1).to(self.device)
            out.append(self.encoder(xb, yb)[:, 1:, :].reshape(len(xb), -1).cpu())
        return torch.cat(out)

    # ── Training ────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray, n_classes: int) -> "TabSynGenerator":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        self.n_classes = int(n_classes)
        if self.unconditional:
            y, self.n_classes = np.zeros_like(y), 1

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        rng = np.random.default_rng(self.seed)

        self.class_counts = np.bincount(y, minlength=self.n_classes)
        self.scaler, Xs = pp.fit_scaler(self.preprocess, X, y)
        self.n_genes = Xs.shape[1]
        self._modules()

        if self.val_frac > 0:
            perm = rng.permutation(len(Xs))
            n_val = max(1, int(round(self.val_frac * len(Xs))))
            val_idx, tr_idx = perm[:n_val], perm[n_val:]
        else:
            tr_idx = val_idx = np.arange(len(Xs))
        self._fit_vae(Xs[tr_idx], y[tr_idx], Xs[val_idx], y[val_idx])

        self.train_z = self.encode(Xs[tr_idx], y[tr_idx])
        self._fit_diffusion(self.train_z)
        return self

    def _fit_vae(self, X, y, X_val, y_val):
        dev = self.device
        Xt, yt = torch.tensor(X).float(), torch.tensor(y).long().view(-1, 1)
        Xv, yv = torch.tensor(X_val).float(), torch.tensor(y_val).long().view(-1, 1)
        opt = torch.optim.Adam(self.vae.parameters(), lr=self.vae_lr, weight_decay=0)
        sched = ReduceLROnPlateau(opt, mode="min", factor=0.95, patience=10)

        best, best_state, patience, beta = float("inf"), None, 0, self.max_beta
        n = len(Xt)
        for epoch in range(self.vae_epochs):
            self.vae.train()
            perm = torch.randperm(n)
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                xb, yb = Xt[idx].to(dev), yt[idx].to(dev)
                opt.zero_grad()
                rec_num, rec_cat, mu, logvar = self.vae(xb, yb)
                mse, ce, kld = vae_loss(xb, yb, rec_num, rec_cat, mu, logvar)
                (mse + ce + beta * kld).backward()
                opt.step()

            # Upstream's schedule: everything keys on validation CE alone
            # (`val_mse * 0 + val_ce`), i.e. on how well the label is rebuilt.
            self.vae.eval()
            with torch.no_grad():
                v_mse = v_ce = 0.0
                for i in range(0, len(Xv), self.batch_size):
                    xb, yb = Xv[i:i + self.batch_size].to(dev), yv[i:i + self.batch_size].to(dev)
                    rec_num, rec_cat, mu, logvar = self.vae(xb, yb)
                    m, c, _ = vae_loss(xb, yb, rec_num, rec_cat, mu, logvar)
                    v_mse += m.item() * len(xb)
                    v_ce += c.item() * len(xb)
                v_mse, v_ce = v_mse / len(Xv), v_ce / len(Xv)
            # With one constant class CE is identically ~0, so a label-free
            # model is scheduled on the reconstruction error instead.
            monitor = v_mse if self.unconditional else v_ce
            sched.step(monitor)
            if monitor < best:
                best, patience = monitor, 0
                best_state = {k: v.detach().clone() for k, v in self.vae.state_dict().items()}
            else:
                patience += 1
                if patience == 10 and beta > self.min_beta:
                    beta = beta * self.lambd
            if self.verbose and (epoch % 200 == 0 or epoch == self.vae_epochs - 1):
                print(f"    [tabsyn vae] epoch {epoch:5d}/{self.vae_epochs}  beta={beta:.5f}  "
                      f"val mse={v_mse:.5f}  ce={v_ce:.5f}", flush=True)

        self.vae.load_state_dict(best_state)
        self.vae.eval()
        self.encoder.load_weights(self.vae)
        self.decoder.load_weights(self.vae)

    def _fit_diffusion(self, train_z: torch.Tensor):
        dev = self.device
        self.z_mean = train_z.mean(0)
        if self.latent_scale == "upstream":
            self.z_scale = 2.0
        elif self.latent_scale == "std":      # EDM's preconditioning assumes sigma_data = 0.5
            self.z_scale = float((train_z - self.z_mean).std()) / 0.5
        else:
            raise ValueError(f"latent_scale must be upstream or std, not {self.latent_scale!r}")
        data = ((train_z - self.z_mean) / self.z_scale).to(dev)
        opt = torch.optim.Adam(self.diffusion.parameters(), lr=self.diff_lr, weight_decay=0)
        if self.diff_schedule == "steps":
            return self._fit_diffusion_steps(data, opt)
        if self.diff_schedule != "upstream":
            raise ValueError(f"diff_schedule must be upstream or steps, not {self.diff_schedule!r}")
        sched = ReduceLROnPlateau(opt, mode="min", factor=0.9, patience=20)

        self.diffusion.train()
        best, best_state, patience, n = float("inf"), None, 0, len(data)
        for epoch in range(self.diff_epochs):
            perm = torch.randperm(n, device=dev)
            total = 0.0
            for i in range(0, n, self.diff_batch_size):
                xb = data[perm[i:i + self.diff_batch_size]]
                loss = self.diffusion(xb).mean()
                total += loss.item() * len(xb)
                opt.zero_grad()
                loss.backward()
                opt.step()
            cur = total / n
            sched.step(cur)
            if cur < best:
                best, patience = cur, 0
                best_state = {k: v.detach().clone()
                              for k, v in self.diffusion.state_dict().items()}
            else:
                patience += 1
                if patience == self.diff_patience:
                    if self.verbose:
                        print(f"    [tabsyn diff] early stop @ epoch {epoch}", flush=True)
                    break
            if self.verbose and epoch % 1000 == 0:
                print(f"    [tabsyn diff] epoch {epoch:5d}/{self.diff_epochs}  "
                      f"loss={cur:.5f}", flush=True)
        self.diffusion.load_state_dict(best_state)
        self.diffusion.eval()

    def _fit_diffusion_steps(self, data, opt):
        """A fixed number of gradient steps, cosine LR decay, averaged weights.

        The EDM loss draws a fresh noise level per row, so one epoch's loss is a
        noisy estimate; nothing here reacts to it.
        """
        import copy
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.diff_steps)
        avg = copy.deepcopy(self.diffusion)
        self.diffusion.train()
        n, step = len(data), 0
        while step < self.diff_steps:
            perm = torch.randperm(n, device=data.device)
            for i in range(0, n, self.diff_batch_size):
                loss = self.diffusion(data[perm[i:i + self.diff_batch_size]]).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                with torch.no_grad():
                    d = min(self.diff_ema, (1 + step) / (10 + step))
                    for pa, pn in zip(avg.parameters(), self.diffusion.parameters()):
                        pa.mul_(d).add_(pn, alpha=1 - d)
                step += 1
                if self.verbose and step % 5000 == 0:
                    print(f"    [tabsyn diff] step {step:6d}/{self.diff_steps}  "
                          f"loss={loss.item():.5f}", flush=True)
                if step >= self.diff_steps:
                    break
        self.diffusion.load_state_dict(avg.state_dict())
        self.diffusion.eval()

    # ── Sampling ────────────────────────────────────────────────────────────

    def sample(self, n: int) -> tuple:
        torch.manual_seed(self.seed + 1)
        if self.class_freq == "generated":
            return self._sample(n)
        if self.class_freq != "train":
            raise ValueError(f"class_freq must be generated or train, not {self.class_freq!r}")
        # Keep generated rows class by class until the release has the training
        # class proportions -- what the label-conditional generators (CVAE,
        # NoisyDiffusion) get by construction.  Rows are never altered.
        want = np.floor(self.class_counts / self.class_counts.sum() * n).astype(int)
        frac = self.class_counts / self.class_counts.sum() * n - want
        want[np.argsort(-frac)[:n - want.sum()]] += 1
        X_keep, y_keep, have = [], [], np.zeros_like(want)
        for _ in range(50):
            X, y = self._sample(2 * n)
            for c in np.flatnonzero(have < want):
                idx = np.flatnonzero(y == c)[:want[c] - have[c]]
                X_keep.append(X[idx]); y_keep.append(y[idx]); have[c] += len(idx)
            if (have >= want).all():
                break
        else:
            raise RuntimeError(f"tabsyn: classes {np.flatnonzero(have < want).tolist()} are "
                               "(almost) never generated; cannot match training proportions")
        X, y = np.concatenate(X_keep), np.concatenate(y_keep)
        perm = np.random.default_rng(self.seed + 1).permutation(len(y))
        return X[perm], y[perm]

    @torch.no_grad()
    def _sample(self, n: int) -> tuple:
        dim = (self.n_genes + 1) * self.d_token
        X_out, y_out = [], []
        for i in range(0, n, self.diff_batch_size):
            m = min(self.diff_batch_size, n - i)
            z = edm_sample(self.diffusion.denoise_fn_D, m, dim,
                           num_steps=self.sample_steps, device=self.device)
            z = (z * self.z_scale + self.z_mean.to(self.device)).float().reshape(m, -1, self.d_token)
            for j in range(0, m, self.batch_size):
                x_num, x_cat = self.decoder(z[j:j + self.batch_size])
                X_out.append(x_num.cpu().numpy())
                y_out.append(x_cat[0].argmax(dim=-1).cpu().numpy())
        y_syn = np.concatenate(y_out).astype(np.int64)
        X_syn = pp.invert_scaler(self.scaler, np.concatenate(X_out), y_syn)
        return X_syn, y_syn

    # ── Persistence ─────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"vae": self.vae.state_dict(),
                    "diffusion": self.diffusion.state_dict(),
                    "z_mean": self.z_mean, "z_scale": self.z_scale, "class_counts": self.class_counts, "n_classes": self.n_classes,
                    "n_genes": self.n_genes}, path)
        pp.save_scaler(self.scaler, pp.scaler_path(path))

    def load(self, path: Path) -> "TabSynGenerator":
        path = Path(path)
        ckpt = torch.load(path, map_location=self.device)
        self.n_classes, self.n_genes = ckpt["n_classes"], ckpt["n_genes"]
        self._modules()
        self.vae.load_state_dict(ckpt["vae"])
        self.vae.eval()
        self.encoder.load_weights(self.vae)
        self.decoder.load_weights(self.vae)
        self.diffusion.load_state_dict(ckpt["diffusion"])
        self.diffusion.eval()
        self.z_mean = ckpt["z_mean"]
        self.z_scale = ckpt.get("z_scale", 2.0)
        self.class_counts = ckpt.get("class_counts")
        self.scaler = pp.load_scaler(pp.scaler_path(path))
        return self


# target name: tabsyn@class_freq=train,diff_schedule=steps,latent_scale=std,preprocess=clip:0.001:0.999+standard
TUNED = dict(class_freq="train", diff_schedule="steps", latent_scale="std",
             preprocess="clip:0.001:0.999+standard")

register(TabSynGenerator)
