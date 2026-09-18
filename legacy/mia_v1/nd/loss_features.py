"""Extract per-sample loss trajectories across timesteps and noise vectors."""

import numpy as np
import torch

from .. import config


@torch.no_grad()
def extract_loss_features(
    model,
    diffusion_trainer,
    X,
    y_int,
    t_list=None,
    n_noise=None,
    device=None,
    batch_size=128,
):
    """Compute the loss-trajectory feature vector for every sample in X.

    For each sample x₀:
      1. Draw n_noise fixed noise vectors ε ~ N(0, I).
      2. For each timestep t in t_list:
           x_t = sqrt(ᾱ_t) · x₀ + sqrt(1 − ᾱ_t) · ε
           ε_pred = model(x_t, t, label)
           loss = ‖ε_pred − ε‖²   (per noise vector, not averaged)

    Returns
    -------
    features : np.ndarray (n_samples, n_noise * len(t_list))
    """
    t_list  = t_list  or config.T_LIST
    n_noise = n_noise or config.N_NOISE
    device  = device  or config.DEVICE
    model.eval()

    n_samples, input_dim = X.shape
    n_t = len(t_list)

    rng = torch.Generator(device="cpu").manual_seed(config.SEED)
    noise_bank = torch.randn(n_noise, input_dim, generator=rng)

    all_losses = np.empty((n_samples, n_t, n_noise), dtype=np.float32)

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y_int, dtype=torch.long)

    for i in range(n_samples):
        x0 = X_t[i]
        label_i = y_t[i].item()

        x0_exp     = x0.unsqueeze(0).expand(n_noise, -1)
        labels_exp = torch.full((n_noise,), label_i, dtype=torch.long)

        for ti, t_val in enumerate(t_list):
            sqrt_abar    = diffusion_trainer.sqrt_alpha_bar[t_val].item()
            sqrt_1m_abar = diffusion_trainer.sqrt_one_minus_alpha_bar[t_val].item()

            x_noisy  = (sqrt_abar * x0_exp + sqrt_1m_abar * noise_bank).to(device)
            t_tensor = torch.full((n_noise,), t_val, dtype=torch.float32, device=device)
            labels_dev = labels_exp.to(device)

            losses_t = []
            for start in range(0, n_noise, batch_size):
                end = min(start + batch_size, n_noise)
                eps_pred = model(x_noisy[start:end], t_tensor[start:end], labels_dev[start:end])
                eps_true = noise_bank[start:end].to(device)
                mse = ((eps_pred - eps_true) ** 2).mean(dim=1)
                losses_t.append(mse.cpu().numpy())

            all_losses[i, ti, :] = np.concatenate(losses_t)

        if (i + 1) % 100 == 0 or i == n_samples - 1:
            print(f"  [loss features] processed {i + 1}/{n_samples} samples")

    return all_losses.reshape(n_samples, n_t * n_noise)


def summarize_features(raw, n_noise, n_t):
    """Summarize raw loss features into per-timestep statistics.

    Returns
    -------
    summary : np.ndarray (n_samples, n_t * 4)  — mean, std, min, max per timestep
    """
    n_samples = raw.shape[0]
    reshaped = raw.reshape(n_samples, n_t, n_noise)
    mean = reshaped.mean(axis=2)
    std  = reshaped.std(axis=2)
    mn   = reshaped.min(axis=2)
    mx   = reshaped.max(axis=2)
    return np.stack([mean, std, mn, mx], axis=2).reshape(n_samples, n_t * 4)


def prepare_features(raw):
    """Dispatch on config.FEATURE_MODE: 'raw' returns as-is, 'summary' summarizes."""
    if config.FEATURE_MODE == "raw":
        return raw
    elif config.FEATURE_MODE == "summary":
        return summarize_features(raw, config.N_NOISE, len(config.T_LIST))
    else:
        raise ValueError(f"Unknown FEATURE_MODE: {config.FEATURE_MODE!r}")
