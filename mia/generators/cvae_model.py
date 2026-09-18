"""Conditional VAE architecture (target generator "cvae").

Layer-for-layer identical to the CAMDA 2025 blue-team baseline
(src/generators/models/cvae.py in the challenge starter package): a 3-layer
encoder to a 512-wide joint representation, Gaussian latent of `z_dim`, and a
mirrored decoder.  The class condition enters both halves through its own
branch, either one-hot or a learned embedding.

MeLoMIA reaches into `encode`/`decode` directly, so the split between them
matters as much as the weights do.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class CVAE(nn.Module):
    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        z_dim: int,
        beta: float = 1.0,
        transform: str = "none",
        condition_type: str = "embedding",
        disease_embed_dim: int = 20,
    ):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.beta = beta
        self.transform = transform
        self.condition_type = condition_type
        self.disease_embed_dim = disease_embed_dim

        if condition_type == "embedding":
            self.disease_embedding = nn.Embedding(y_dim, disease_embed_dim)
            y_input_dim = disease_embed_dim
        else:
            y_input_dim = y_dim
        self.y_input_dim = y_input_dim

        self.fc_feat_x = nn.Sequential(
            nn.Linear(x_dim, 1000), nn.ReLU(),
            nn.Linear(1000, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
        )
        self.fc_feat_y = nn.Sequential(nn.Linear(y_input_dim, 256), nn.ReLU())
        self.fc_feat_all = nn.Sequential(nn.Linear(512, 512), nn.ReLU())
        self.fc_mu = nn.Linear(512, z_dim)
        self.fc_logvar = nn.Linear(512, z_dim)

        self.dec_z = nn.Sequential(nn.Linear(z_dim, 256), nn.ReLU())
        self.dec_y = nn.Sequential(nn.Linear(y_input_dim, 256), nn.ReLU())
        self.dec = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 1000), nn.ReLU(),
            nn.Linear(1000, x_dim),
        )
        self.rec_crit = nn.MSELoss()

    # ── Core ────────────────────────────────────────────────────────────────

    def encode(self, x, y_vec):
        feat = self.fc_feat_all(
            torch.cat([self.fc_feat_x(x), self.fc_feat_y(y_vec)], dim=1)
        )
        return self.fc_mu(feat), self.fc_logvar(feat)

    def decode(self, z, y_vec):
        out = self.dec(torch.cat([self.dec_z(z), self.dec_y(y_vec)], dim=1))
        if self.transform == "exp":
            out = out.exp()
        elif self.transform == "sigmoid":
            out = torch.sigmoid(out)
        elif self.transform == "relu":
            out = torch.relu(out)
        return out

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x, y_vec):
        mu, logvar = self.encode(x, y_vec)
        return mu, logvar, self.decode(self.reparameterize(mu, logvar), y_vec)

    def compute_loss(self, x, y_vec) -> dict:
        mu, logvar, rec = self.forward(x, y_vec)
        rec_loss = self.rec_crit(rec, x)
        kl_loss = torch.mean(
            -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0
        )
        return {
            "loss": rec_loss + self.beta * kl_loss,
            "rec_loss": rec_loss,
            "kl_loss": kl_loss,
            "mu": mu,
            "logvar": logvar,
            "rec": rec,
        }

    # ── Conditioning ────────────────────────────────────────────────────────

    def encode_condition(self, y, condition_mode: str | None = None):
        """Integer labels -> the conditioning vector both halves consume.

        `condition_mode="none"` returns zeros, which switches the model to an
        unconditional VAE.  MeLoMIA uses that when it has no trustworthy class
        labels for the target cohort, rather than conditioning on guesses.
        """
        dev = next(self.parameters()).device
        if condition_mode == "none":
            batch = y.shape[0] if hasattr(y, "shape") and y.dim() > 0 else 1
            return torch.zeros(batch, self.y_input_dim, device=dev)

        if hasattr(y, "dim") and y.dim() == 2 and y.size(1) == self.y_dim:
            y = y.argmax(dim=1)
        y = y.long().to(dev)

        if self.condition_type == "embedding":
            return self.disease_embedding(y)
        out = torch.zeros(len(y), self.y_dim, device=dev)
        out.scatter_(1, y.unsqueeze(1), 1)
        return out.float()

    # ── Sampling ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def sample_labels(self, y_int: np.ndarray, condition_mode=None, device="cpu",
                      batch_size: int = 256):
        """Decode one sample per entry of `y_int` from the standard-normal prior."""
        self.eval()
        outs = []
        y_t = torch.as_tensor(y_int, dtype=torch.long)
        for start in range(0, len(y_t), batch_size):
            yb = y_t[start:start + batch_size].to(device)
            z = torch.randn(len(yb), self.z_dim, device=device)
            y_vec = self.encode_condition(yb, condition_mode=condition_mode)
            outs.append(self.decode(z, y_vec).cpu().numpy())
        return np.concatenate(outs).astype(np.float32)
