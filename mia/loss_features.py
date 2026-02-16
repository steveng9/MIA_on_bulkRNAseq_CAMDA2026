"""Extract per-sample loss trajectories across timesteps and noise vectors."""

import numpy as np
import torch

from . import config


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

    For each sample x₀ we:
      1. Draw *n_noise* fixed noise vectors ε ~ N(0, I).
      2. For each timestep t in *t_list*:
           x_t = sqrt(ᾱ_t) · x₀ + sqrt(1 − ᾱ_t) · ε
           ε_pred = model(x_t, t, label)
           loss = ‖ε_pred − ε‖²          (per-noise-vector, not averaged)

    The result is a matrix of shape (n_samples, n_noise × len(t_list)).

    Parameters
    ----------
    model : EmbeddedDiffusion (already on *device*, in eval mode)
    diffusion_trainer : DiffusionTrainer (holds the noise schedule)
    X : np.ndarray, shape (n_samples, input_dim) – already scaled
    y_int : np.ndarray, shape (n_samples,) – integer class labels
    t_list : list[int]
    n_noise : int
    device : str
    batch_size : int  – how many (sample × noise) pairs to process at once

    Returns
    -------
    features : np.ndarray, shape (n_samples, n_noise * len(t_list))
    """
    t_list = t_list or config.T_LIST
    n_noise = n_noise or config.N_NOISE
    device = device or config.DEVICE
    model.eval()

    n_samples, input_dim = X.shape
    n_t = len(t_list)

    # Pre-draw fixed noise vectors: (n_noise, input_dim)
    rng = torch.Generator(device="cpu").manual_seed(config.SEED)
    noise_bank = torch.randn(n_noise, input_dim, generator=rng)  # on CPU

    # We'll fill this: (n_samples, n_t, n_noise)
    all_losses = np.empty((n_samples, n_t, n_noise), dtype=np.float32)

    # Process one sample at a time, batching over noise vectors
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y_int, dtype=torch.long)

    for i in range(n_samples):
        x0 = X_t[i]                          # (input_dim,)
        label_i = y_t[i].item()

        # Expand x0 to (n_noise, input_dim)
        x0_exp = x0.unsqueeze(0).expand(n_noise, -1)       # (n_noise, input_dim)
        labels_exp = torch.full((n_noise,), label_i, dtype=torch.long)

        for ti, t_val in enumerate(t_list):
            # Compute x_t for all noise vectors at once
            sqrt_abar = diffusion_trainer.sqrt_alpha_bar[t_val].item()
            sqrt_1m_abar = diffusion_trainer.sqrt_one_minus_alpha_bar[t_val].item()

            x_noisy = (sqrt_abar * x0_exp + sqrt_1m_abar * noise_bank).to(device)
            t_tensor = torch.full((n_noise,), t_val, dtype=torch.float32, device=device)
            labels_dev = labels_exp.to(device)

            # Process in sub-batches if n_noise is large
            losses_t = []
            for start in range(0, n_noise, batch_size):
                end = min(start + batch_size, n_noise)
                eps_pred = model(
                    x_noisy[start:end],
                    t_tensor[start:end],
                    labels_dev[start:end],
                )
                eps_true = noise_bank[start:end].to(device)
                # Per-noise-vector MSE: (batch,)
                mse = ((eps_pred - eps_true) ** 2).mean(dim=1)
                losses_t.append(mse.cpu().numpy())

            all_losses[i, ti, :] = np.concatenate(losses_t)

        if (i + 1) % 100 == 0 or i == n_samples - 1:
            print(f"  [loss features] processed {i + 1}/{n_samples} samples")

    # Reshape to (n_samples, n_t * n_noise)
    features = all_losses.reshape(n_samples, n_t * n_noise)
    return features
