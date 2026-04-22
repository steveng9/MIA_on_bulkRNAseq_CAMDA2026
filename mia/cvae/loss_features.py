"""Extract CVAE loss features for membership inference.

Feature design
--------------
For each real sample x (with class condition y_vec):

1. Encode x → μ, logvar  (deterministic, one encoder pass)
2. Per-dimension KL contribution:
     kl_d = -0.5 * (1 + logvar_d - μ_d² - exp(logvar_d))   ∈ R^z_dim
   This quantifies how much each latent dimension is used for this sample.
   Members' posteriors may differ systematically from non-members'.

3. For each temperature α ∈ TEMP_LIST:
   For each of N_SAMPLES fixed noise vectors ε ~ N(0,I):
     z = μ + α·exp(0.5·logvar)·ε
     rec_loss = MSE(decode(z, y_vec), x)   (scalar)
   This gives a (N_SAMPLES × len(TEMP_LIST)) matrix of reconstruction losses
   analogous to the (N_NOISE × len(T_LIST)) diffusion loss matrix.

   At α=0: z=μ for all noise draws → N_SAMPLES identical values (deterministic).
   At higher α: more stochastic z → higher variance in rec_loss.
   Members are expected to have lower rec_loss and more graceful degradation
   across temperatures because the decoder has been trained on them.

Raw feature vector per sample (shape):
  rec_losses : (N_SAMPLES * len(TEMP_LIST),)   — ordered temp-major
  per_dim_kl : (z_dim,)
  concatenated: (N_SAMPLES * len(TEMP_LIST) + z_dim,)

Summary feature vector (prepare_cvae_features with FEATURE_MODE='summary'):
  per_temp stats (mean,std,min,max) : (4 * len(TEMP_LIST),)
  per_dim_kl                        : (z_dim,)
  concatenated: (4*len(TEMP_LIST) + z_dim,)
"""

import numpy as np
import torch

from .. import config


@torch.no_grad()
def extract_cvae_features(
    model,
    X,
    y_int,
    temp_list=None,
    n_samples=None,
    device=None,
    batch_size=512,
):
    """Compute CVAE loss-feature vectors for every sample in X.

    Parameters
    ----------
    model      : CVAE, on *device*, in eval mode
    X          : np.ndarray (n_data, x_dim) — already scaled
    y_int      : np.ndarray (n_data,) int64 — integer class labels
                 (ignored when CVAE_LABEL_MODE == 'none')
    temp_list  : list[float] — temperature values for z-sampling
    n_samples  : int — K noise samples per temperature
    device     : str
    batch_size : int — max z-vectors decoded at once per sample

    Returns
    -------
    rec_losses  : np.ndarray (n_data, len(temp_list) * n_samples)
    per_dim_kl  : np.ndarray (n_data, z_dim)
    """
    temp_list  = temp_list  or config.CVAE_TEMP_LIST
    n_samples  = n_samples  or config.CVAE_N_SAMPLES
    device     = device     or config.DEVICE
    condition_mode = config.CVAE_LABEL_MODE

    model.eval()
    n_data, x_dim = X.shape
    n_t = len(temp_list)
    z_dim = model.z_dim

    # Fixed noise bank: (n_samples, z_dim)
    rng = torch.Generator(device="cpu").manual_seed(config.SEED)
    noise_bank = torch.randn(n_samples, z_dim, generator=rng)  # CPU, fixed

    rec_losses = np.empty((n_data, n_t * n_samples), dtype=np.float32)
    per_dim_kl = np.empty((n_data, z_dim), dtype=np.float32)

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y_int, dtype=torch.long)

    for i in range(n_data):
        x_i = X_t[i].unsqueeze(0).to(device)  # (1, x_dim)
        y_vec_i = model._encode_condition(
            y_t[i].unsqueeze(0).to(device), condition_mode=condition_mode
        )  # (1, y_input_dim)

        # ── Encode once (deterministic) ───────────────────────────────────
        mu, logvar = model.encode(x_i, y_vec_i)   # (1, z_dim) each
        mu_np = mu.squeeze(0).cpu().numpy()
        lv_np = logvar.squeeze(0).cpu().numpy()

        # Per-dim KL: -0.5*(1 + logvar - μ² - exp(logvar))
        per_dim_kl[i] = -0.5 * (1 + lv_np - mu_np ** 2 - np.exp(lv_np))

        # ── Stochastic reconstruction losses at each temperature ──────────
        mu_dev   = mu.squeeze(0)                          # (z_dim,)
        sigma_dev = torch.exp(0.5 * logvar.squeeze(0))   # (z_dim,)
        x_exp = x_i.expand(batch_size, -1)               # reused below

        # y_vec expanded for decoder batching
        # (will be sliced to actual batch size)

        for ti, alpha in enumerate(temp_list):
            # z = mu + alpha * sigma * noise_bank  — shape (n_samples, z_dim)
            noise_dev = noise_bank.to(device)
            z_all = mu_dev.unsqueeze(0) + alpha * sigma_dev.unsqueeze(0) * noise_dev
            # z_all: (n_samples, z_dim)

            losses_t = []
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                bs = end - start
                z_b = z_all[start:end]                             # (bs, z_dim)
                y_b = y_vec_i.expand(bs, -1)                      # (bs, y_input_dim)
                rec_b = model.decode(z_b, y_b)                    # (bs, x_dim)
                x_b = x_i.expand(bs, -1)                          # (bs, x_dim)
                mse = ((rec_b - x_b) ** 2).mean(dim=1)            # (bs,)
                losses_t.append(mse.cpu().numpy())

            rec_losses[i, ti * n_samples:(ti + 1) * n_samples] = np.concatenate(losses_t)

        if (i + 1) % 100 == 0 or i == n_data - 1:
            print(f"  [CVAE features] processed {i + 1}/{n_data} samples")

    return rec_losses, per_dim_kl


def summarize_rec_losses(rec_losses, n_samples, n_temps):
    """Summarize stochastic reconstruction losses into per-temperature statistics.

    Parameters
    ----------
    rec_losses : (n_data, n_temps * n_samples) — temp-major ordering

    Returns
    -------
    summary : (n_data, n_temps * 4)  — mean,std,min,max per temperature
    """
    n_data = rec_losses.shape[0]
    # Reshape: (n_data, n_temps, n_samples)
    reshaped = rec_losses.reshape(n_data, n_temps, n_samples)
    mean = reshaped.mean(axis=2)
    std  = reshaped.std(axis=2)
    mn   = reshaped.min(axis=2)
    mx   = reshaped.max(axis=2)
    # Stack → (n_data, n_temps, 4) → (n_data, n_temps*4)
    return np.stack([mean, std, mn, mx], axis=2).reshape(n_data, n_temps * 4)


def prepare_cvae_features(rec_losses, per_dim_kl):
    """Combine and optionally summarize CVAE features based on CVAE_FEATURE_MODE.

    Parameters
    ----------
    rec_losses  : (n_data, n_temps * N_SAMPLES)
    per_dim_kl  : (n_data, z_dim)

    Returns
    -------
    features : (n_data, total_dim)
    """
    if config.CVAE_FEATURE_MODE == "raw":
        return np.concatenate([rec_losses, per_dim_kl], axis=1)
    elif config.CVAE_FEATURE_MODE == "summary":
        n_temps = len(config.CVAE_TEMP_LIST)
        n_samp  = config.CVAE_N_SAMPLES
        summary = summarize_rec_losses(rec_losses, n_samp, n_temps)
        return np.concatenate([summary, per_dim_kl], axis=1)
    else:
        raise ValueError(f"Unknown CVAE_FEATURE_MODE: {config.CVAE_FEATURE_MODE!r}")
