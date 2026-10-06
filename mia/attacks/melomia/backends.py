"""Generator-specific halves of MeLoMIA.

A backend answers three questions for one generator family:

  * what does a *base shadow* look like -- a generator configured exactly like
    the target, used to turn a controlled real-data split into an internal
    synthetic dataset;
  * what does a *synth-shadow / proxy* look like -- the same family, free to
    overfit its synthetic training set, since its job is to expose membership
    signal rather than to generate well;
  * how do you read a per-sample loss grid out of one of those models.

Everything above this file is generator-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from ... import datasets as D
from ... import generators as G
from ... import preprocessing as pp

#: The diffusion timesteps MeLoMIA-ND sweeps.  Spans the whole trajectory from
#: near-clean (t=1, where only a memorised sample can be denoised exactly) to
#: heavily corrupted (t=750, where nothing is recoverable and the loss is a
#: baseline).  Optuna picks the informative window out of this superset.
ND_TIMESTEPS = (1, 2, 5, 10, 20, 30, 40, 50, 75, 100, 150, 200, 300, 500, 750)

#: Posterior temperatures MeLoMIA-CVAE sweeps.  alpha=0 decodes the posterior
#: mean (pure reconstruction quality); larger alpha walks away from it, and how
#: fast the reconstruction degrades is what distinguishes a sharply-fitted
#: member posterior from a diffuse non-member one.
CVAE_TEMPERATURES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)


@dataclass
class Backend(ABC):
    dataset: str
    device: str = "cuda"
    seed: int = 42
    verbose: bool = True

    name: str = field(init=False, default="base")

    # ── Sweep axis ──────────────────────────────────────────────────────────

    @property
    @abstractmethod
    def sweep_points(self) -> tuple:
        """The superset of sweep values (timesteps or temperatures)."""

    @property
    @abstractmethod
    def n_noise(self) -> int:
        """Number of frozen draws per sweep point."""

    # ── Model construction ──────────────────────────────────────────────────

    @abstractmethod
    def base_shadow(self) -> G.Generator:
        """Target-faithful generator, for turning a real split into synthetic data."""

    @abstractmethod
    def probe(self) -> G.Generator:
        """Sharp generator used for synth-shadows and the inference-time proxy."""

    @abstractmethod
    def extract(self, gen: G.Generator, X_raw: np.ndarray, y=None) -> tuple:
        """Return (losses (n, n_sweep, n_noise), extra (n, d) or None).

        `y` (class per record) is passed only in the white-box setting, where
        the model read is a label-conditional target or shadow; probes are
        label-free and ignore it.
        """

    def load_probe(self, path: Path) -> G.Generator:
        return self.probe().load(path)

    def load_base(self, path: Path) -> G.Generator:
        return self.base_shadow().load(path)


# ─────────────────────────────────────────────────────────────────────────────
# NoisyDiffusion
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NDBackend(Backend):
    timesteps: tuple = ND_TIMESTEPS
    n_noise_vectors: int = 600
    shadow_epochs: int = 200
    probe_epochs: int = 300
    #: rows per forward pass during extraction.  Each sample is expanded by
    #: n_noise, so the sample batch is derived from this rather than fixed --
    #: otherwise the working set scales with the noise budget and a 600-vector
    #: run allocates tens of GB, which matters when several workers share a GPU.
    rows_per_forward: int = 40_000

    name = "nd"

    @property
    def sweep_points(self) -> tuple:
        return tuple(self.timesteps)

    @property
    def n_noise(self) -> int:
        return self.n_noise_vectors

    def base_shadow(self) -> G.Generator:
        return G.build("nd", seed=self.seed, device=self.device,
                       epochs=self.shadow_epochs, dp_noise_multiplier=1e-5,
                       smote_upsample_to=3000, unconditional=False, verbose=False)

    def probe(self) -> G.Generator:
        # Unconditional: the adversary has no trustworthy subtype labels for a
        # released synthetic dataset, and conditioning on guesses would add
        # label-prediction error straight into the loss features.
        return G.build("nd", seed=self.seed, device=self.device,
                       epochs=self.probe_epochs, dp_noise_multiplier=0.0,
                       smote_upsample_to=None, unconditional=True, verbose=False)

    @torch.no_grad()
    def extract(self, gen, X_raw: np.ndarray, y=None) -> tuple:
        """Denoising error at each (timestep, frozen epsilon) pair.

        For every sweep point t the sample is corrupted exactly as training
        would corrupt it,
            x_t = sqrt(abar_t) x_0 + sqrt(1 - abar_t) eps,
        the noise-prediction head is queried, and the squared error against the
        epsilon that was actually injected is averaged over the 978 genes.  A
        model that memorised x_0 predicts eps unusually well.

        The epsilon bank is drawn once from a fixed seed and reused for every
        sample and every shadow, so differences between samples are differences
        in the model, not in which noise they happened to get.
        """
        model, diff = gen.model, gen.diffusion
        model.eval()
        X = pp.apply_scaler(gen.scaler, X_raw)

        n, dim = X.shape
        n_sweep, n_noise = len(self.sweep_points), self.n_noise
        out = np.empty((n, n_sweep, n_noise), dtype=np.float32)

        rng = torch.Generator(device="cpu").manual_seed(self.seed)
        bank = torch.randn(n_noise, dim, generator=rng).to(self.device)
        X_gpu = torch.tensor(X, dtype=torch.float32, device=self.device)
        labels = torch.zeros(len(X), dtype=torch.long, device=self.device)
        batch = max(1, self.rows_per_forward // n_noise)

        for si, t_val in enumerate(self.sweep_points):
            a = diff.sqrt_alpha_bar[t_val].item()
            b = diff.sqrt_one_minus_alpha_bar[t_val].item()
            t_scalar = torch.tensor(float(t_val), device=self.device)
            for start in range(0, n, batch):
                end = min(start + batch, n)
                x0 = X_gpu[start:end]
                B = end - start
                noisy = (a * x0.unsqueeze(1) + b * bank.unsqueeze(0)).reshape(B * n_noise, dim)
                eps_pred = model(
                    noisy,
                    t_scalar.expand(B * n_noise),
                    labels[start:end].unsqueeze(1).expand(-1, n_noise).reshape(-1),
                )
                eps_true = bank.unsqueeze(0).expand(B, -1, -1).reshape(B * n_noise, dim)
                mse = ((eps_pred - eps_true) ** 2).mean(dim=1).reshape(B, n_noise)
                out[start:end, si, :] = mse.cpu().numpy()
        return out, None


# ─────────────────────────────────────────────────────────────────────────────
# Conditional VAE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CVAEBackend(Backend):
    temperatures: tuple = CVAE_TEMPERATURES
    n_draws: int = 50
    shadow_num_iters: int = 10000
    probe_epochs: int = 500
    batch_size: int = 64
    include_kl: bool = True
    include_latent_norms: bool = True

    name = "cvae"

    @property
    def sweep_points(self) -> tuple:
        return tuple(self.temperatures)

    @property
    def n_noise(self) -> int:
        return self.n_draws

    def base_shadow(self) -> G.Generator:
        return G.CVAEGenerator.preset(
            "target", seed=self.seed, device=self.device,
            num_iters=self.shadow_num_iters, verbose=False,
        )

    def probe(self) -> G.Generator:
        # Unconditional, for the same reason as ND: guessed subtype labels would
        # corrupt the very reconstruction losses being measured.
        return G.CVAEGenerator.preset(
            "sharp", seed=self.seed, device=self.device,
            epochs=self.probe_epochs, condition_mode="none", verbose=False,
        )

    @torch.no_grad()
    def extract(self, gen, X_raw: np.ndarray, y=None) -> tuple:
        """Reconstruction error across a sweep of posterior temperatures.

        One encoder pass gives (mu, sigma).  At temperature alpha the latent is
        perturbed as z = mu + alpha * sigma * eps for each frozen eps, decoded,
        and scored against the input.  alpha=0 measures pure reconstruction of
        the posterior mean; increasing alpha probes how wide the basin around
        that mean is.

        The per-dimension KL contribution of the posterior to the standard
        normal prior is appended as extra features: members tend to get sharper,
        more individually-shaped posteriors, and the *pattern* across latent
        dimensions carries more of that than the scalar total does.
        """
        model = gen.model
        model.eval()
        X = pp.apply_scaler(gen.scaler, X_raw)

        n = len(X)
        n_sweep, n_draw = len(self.sweep_points), self.n_draws
        z_dim = model.z_dim

        rng = torch.Generator(device="cpu").manual_seed(self.seed)
        bank = torch.randn(n_draw, z_dim, generator=rng).to(self.device)

        losses = np.empty((n, n_sweep, n_draw), dtype=np.float32)
        kl = np.empty((n, z_dim), dtype=np.float32)
        norms = np.empty((n, 2), dtype=np.float32)

        X_gpu = torch.tensor(X, dtype=torch.float32, device=self.device)
        labels = torch.zeros(n, dtype=torch.long, device=self.device)

        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            xb = X_gpu[start:end]
            B = end - start
            y_vec = model.encode_condition(labels[start:end], condition_mode="none")

            mu, logvar = model.encode(xb, y_vec)
            kl[start:end] = (
                -0.5 * (1 + logvar - mu ** 2 - logvar.exp())
            ).cpu().numpy()
            sigma = torch.exp(0.5 * logvar)
            norms[start:end, 0] = mu.norm(dim=1).cpu().numpy()
            norms[start:end, 1] = sigma.norm(dim=1).cpu().numpy()

            y_flat = y_vec.unsqueeze(1).expand(-1, n_draw, -1).reshape(B * n_draw, -1)
            x_flat = xb.unsqueeze(1).expand(-1, n_draw, -1).reshape(B * n_draw, -1)
            for si, alpha in enumerate(self.sweep_points):
                z = mu.unsqueeze(1) + alpha * sigma.unsqueeze(1) * bank.unsqueeze(0)
                rec = model.decode(z.reshape(B * n_draw, z_dim), y_flat)
                mse = ((rec - x_flat) ** 2).mean(dim=1).reshape(B, n_draw)
                losses[start:end, si, :] = mse.cpu().numpy()

        blocks = []
        if self.include_kl:
            blocks.append(kl)
        if self.include_latent_norms:
            blocks.append(norms)
        extra = np.concatenate(blocks, axis=1).astype(np.float32) if blocks else None
        return losses, extra


# ─────────────────────────────────────────────────────────────────────────────
# TabSyn (latent diffusion)
# ─────────────────────────────────────────────────────────────────────────────

#: EDM noise levels MeLoMIA-TabSyn sweeps, in units of the normalised latent
#: (data SD 0.5).  Training draws ln(sigma) ~ N(-1.2, 1.2), i.e. mostly
#: 0.03-3; the sweep runs from near-clean, where only a memorised latent is
#: denoised exactly, to far above the data scale, where the loss is a baseline.
TABSYN_SIGMAS = (0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.5, 5.0, 10.0, 40.0)


@dataclass
class TabSynBackend(Backend):
    """The model roles, following the rule of the TimeDiff audit
    (~/ansons_capstone, commits ccb82c0 and 1c01066):

      base shadow   the target's recipe exactly (`tabsyn.TUNED`, tuned for
                    fidelity), releasing as many rows as it was trained on, so
                    its synthetic data is distributed like the target's;
      probe         synth-shadows and the final proxy share ONE recipe, chosen
                    to memorise the synthetic data they are fitted to: 4x the
                    denoiser's capacity (width 1024 -> 2048) and 2x its steps,
                    label-free, same learning rate, batch and weight averaging.
                    The VAE is left at the target's recipe: it already
                    reconstructs its training rows to an MSE of ~2e-5.

    Both scale their inputs with their own training data's statistics and score
    real records through that same scaler.
    """
    sigmas: tuple = TABSYN_SIGMAS
    n_noise_vectors: int = 200
    vae_temperatures: tuple = CVAE_TEMPERATURES   # the VAE half: MeLoMIA-CVAE's sweep
    vae_draws: int = 16
    probe_dim_t: int = 2048
    probe_diff_steps: int = 40000
    probe_vae_epochs: int = 4000
    shadow_vae_epochs: int = 4000         # the target's; lower only for tests
    shadow_diff_steps: int = 20000        # likewise
    rows_per_forward: int = 20_000

    name = "tabsyn"

    @property
    def sweep_points(self) -> tuple:
        return tuple(self.sigmas)

    @property
    def n_noise(self) -> int:
        return self.n_noise_vectors

    def base_shadow(self) -> G.Generator:
        from ...generators.tabsyn import TUNED
        return G.build("tabsyn", seed=self.seed, device=self.device, verbose=False,
                       vae_epochs=self.shadow_vae_epochs,
                       diff_steps=self.shadow_diff_steps, **TUNED)

    def probe(self) -> G.Generator:
        from ...generators.tabsyn import TUNED
        kw = dict(TUNED, class_freq="generated", unconditional=True,
                  dim_t=self.probe_dim_t, diff_steps=self.probe_diff_steps,
                  vae_epochs=self.probe_vae_epochs)
        return G.build("tabsyn", seed=self.seed, device=self.device, verbose=False, **kw)

    @torch.no_grad()
    def extract(self, gen, X_raw: np.ndarray, y=None) -> tuple:
        """Loss signatures from both halves of the model.

        Diffusion half (the grid, as in MeLoMIA-ND).  A record is scaled and
        encoded to its latent z (encoder mean), normalised exactly as in
        training.  For every sweep level sigma it is corrupted as
        z + sigma * eps and the EDM denoiser D(., sigma) is asked for z back;
        the squared error is averaged over the latent dimensions.  A model that
        memorised z denoises it unusually well.

        VAE half (extra features, as in MeLoMIA-CVAE).  Reconstruction error
        across a sweep of posterior temperatures, plus KL and norm statistics
        of the posterior; see the comments below.

        Both epsilon banks are drawn once from fixed seeds and shared by all
        records and models.
        """
        X = np.asarray(pp.apply_scaler(gen.scaler, X_raw), dtype=np.float32)
        n = len(X)
        labels = (np.zeros(n, dtype=np.int64) if gen.n_classes == 1 or y is None
                  else np.asarray(y, dtype=np.int64))
        if gen.n_classes > 1 and y is None:
            raise ValueError("a label-conditional TabSyn model needs the records' classes")

        # ── VAE half: the MeLoMIA-CVAE signature, on TabSyn's VAE ───────────
        # One encoder pass gives the posterior (mu, sigma) of every latent
        # coordinate.  At temperature alpha the latent is perturbed as
        # mu + alpha * sigma * eps for each frozen eps, decoded, and scored
        # against the input, exactly as CVAEBackend.extract does.
        core = gen.vae.VAE
        temps, n_draw = self.vae_temperatures, self.vae_draws
        lat = (gen.n_genes + 1, gen.d_token)
        rng = torch.Generator(device="cpu").manual_seed(self.seed + 1)
        vbank = torch.randn(n_draw, *lat, generator=rng).to(self.device)
        z = torch.empty(n, lat[0] * lat[1])
        vae_grid = np.empty((n, len(temps), n_draw), dtype=np.float32)
        kl_stats = np.empty((n, 5), dtype=np.float32)
        vb = max(1, gen.batch_size // n_draw)
        for i in range(0, n, vb):
            xb = torch.as_tensor(X[i:i + vb], device=self.device)
            yb = torch.as_tensor(labels[i:i + vb], device=self.device).view(-1, 1)
            tok = core.Tokenizer(xb, yb)
            mu, logvar = core.encoder_mu(tok)[:, 1:], core.encoder_logvar(tok)[:, 1:]
            sigma = torch.exp(0.5 * logvar)
            B = len(xb)
            z[i:i + B] = mu.reshape(B, -1).cpu()
            kl = (-0.5 * (1 + logvar - mu ** 2 - logvar.exp())).reshape(B, -1)
            kl_stats[i:i + B] = torch.stack(
                [kl.mean(1), kl.std(1), kl.max(1).values,
                 mu.reshape(B, -1).norm(dim=1), sigma.reshape(B, -1).norm(dim=1)], dim=1).cpu().numpy()
            x_rep = xb.unsqueeze(1).expand(-1, n_draw, -1).reshape(B * n_draw, -1)
            for ti, alpha in enumerate(temps):
                zz = mu.unsqueeze(1) + alpha * sigma.unsqueeze(1) * vbank.unsqueeze(0)
                x_num, _ = gen.decoder(zz.reshape(B * n_draw, *lat))
                vae_grid[i:i + B, ti] = ((x_num - x_rep) ** 2).mean(dim=1).reshape(B, n_draw).cpu().numpy()
        zn = ((z - gen.z_mean.cpu()) / gen.z_scale).to(self.device)
        dim = zn.shape[1]

        # ── diffusion half ──────────────────────────────────────────────────
        n_sweep, n_noise = len(self.sweep_points), self.n_noise
        out = np.empty((n, n_sweep, n_noise), dtype=np.float32)
        rng = torch.Generator(device="cpu").manual_seed(self.seed)
        bank = torch.randn(n_noise, dim, generator=rng).to(self.device)
        denoise = gen.diffusion.denoise_fn_D
        batch = max(1, self.rows_per_forward // n_noise)
        for si, sigma in enumerate(self.sweep_points):
            for start in range(0, n, batch):
                z0 = zn[start:start + batch]
                B = len(z0)
                noisy = (z0.unsqueeze(1) + sigma * bank.unsqueeze(0)).reshape(B * n_noise, dim)
                pred = denoise(noisy, torch.full((B * n_noise,), float(sigma), device=self.device))
                target = z0.unsqueeze(1).expand(-1, n_noise, -1).reshape(B * n_noise, dim)
                mse = ((pred - target) ** 2).mean(dim=1).reshape(B, n_noise)
                out[start:start + B, si, :] = mse.cpu().numpy()
        out = np.maximum(out, 1e-12)            # the grid is log-transformed downstream
        # The VAE grid has its own sweep axis, so it enters as summaries over the
        # draws rather than through the diffusion grid: per temperature the
        # mean, SD, minimum and median of the log error (alpha = 0 is
        # deterministic, so one value), then the posterior's KL statistics and
        # norms, then the norm of the normalised latent.
        lg = np.log(np.maximum(vae_grid, 1e-12))
        blocks = []
        for ti, alpha in enumerate(temps):
            if alpha == 0:
                blocks.append(lg[:, ti, :1])
            else:
                blocks.append(np.stack([lg[:, ti].mean(1), lg[:, ti].std(1), lg[:, ti].min(1),
                                        np.median(lg[:, ti], axis=1)], axis=1))
        blocks += [np.log(kl_stats[:, :3] + 1e-12), kl_stats[:, 3:],
                   zn.norm(dim=1, keepdim=True).cpu().numpy()]
        return out, np.concatenate(blocks, axis=1).astype(np.float32)


BACKENDS = {"nd": NDBackend, "cvae": CVAEBackend, "tabsyn": TabSynBackend}


def build_backend(name: str, dataset: str, **kw) -> Backend:
    if name not in BACKENDS:
        raise KeyError(f"Unknown MeLoMIA backend {name!r}. Known: {sorted(BACKENDS)}")
    return BACKENDS[name](dataset=dataset, **kw)
