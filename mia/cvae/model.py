"""CVAE model for membership inference shadow modelling.

Adapted from Health-Privacy-Challenge/src/generators/models/cvae.py.
Key additions:
  - condition_mode="none" in _encode_condition: returns a true zero tensor,
    bypassing all embedding/linear layers, for unconditional-equivalent training.
  - Fixed generate_for_subtype bug (y_vec was undefined in the original).
  - Removed CVAEDataGenerationPipeline and CLI code (not needed here).
"""

import numpy as np
import torch
import torch.nn as nn


def _one_hot(labels: torch.Tensor, num_classes: int, device) -> torch.Tensor:
    """Integer labels → float one-hot, shape (batch, num_classes)."""
    out = torch.zeros(len(labels), num_classes, device=device)
    out.scatter_(1, labels.long().unsqueeze(1).to(device), 1)
    return out.float()


class CVAE(nn.Module):
    def __init__(self, x_dim, y_dim, z_dim, beta=1.0,
                 transform="none", condition_type="onehot", disease_embed_dim=20):
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

        self.fc_feat_x = nn.Sequential(
            nn.Linear(x_dim, 1000), nn.ReLU(),
            nn.Linear(1000, 512),  nn.ReLU(),
            nn.Linear(512, 256),   nn.ReLU(),
        )
        self.fc_feat_y = nn.Sequential(nn.Linear(y_input_dim, 256), nn.ReLU())
        self.fc_feat_all = nn.Sequential(nn.Linear(512, 512), nn.ReLU())
        self.fc_mu = nn.Linear(512, z_dim)
        self.fc_logvar = nn.Linear(512, z_dim)

        self.dec_z = nn.Sequential(nn.Linear(z_dim, 256), nn.ReLU())
        self.dec_y = nn.Sequential(nn.Linear(y_input_dim, 256), nn.ReLU())
        self.dec = nn.Sequential(
            nn.Linear(512, 512),  nn.ReLU(),
            nn.Linear(512, 1000), nn.ReLU(),
            nn.Linear(1000, x_dim),
        )
        self.rec_crit = nn.MSELoss()

    # ── Core operations ───────────────────────────────────────────────────────

    def forward(self, x, y_vec):
        """x: (B, x_dim), y_vec: (B, y_input_dim) — already encoded."""
        mu, logvar = self.encode(x, y_vec)
        rec = self.decode(self.reparameterize(mu, logvar), y_vec)
        return mu, logvar, rec

    def encode(self, x, y_vec):
        feat_x = self.fc_feat_x(x)
        feat_y = self.fc_feat_y(y_vec)
        feat = self.fc_feat_all(torch.cat([feat_x, feat_y], dim=1))
        return self.fc_mu(feat), self.fc_logvar(feat)

    def decode(self, z, y_vec):
        out = self.dec(torch.cat([self.dec_z(z), self.dec_y(y_vec)], dim=1))
        if self.transform == "exp":
            out = out.exp()
        elif self.transform == "sigmoid":
            out = torch.sigmoid(out)
        elif self.transform == "relu":
            out = torch.nn.functional.relu(out)
        return out

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def compute_loss(self, x, y_vec):
        """Returns dict with loss, rec_loss, kl_loss, mu, logvar, rec."""
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

    # ── Condition encoding ────────────────────────────────────────────────────

    def _encode_condition(self, y, condition_mode=None):
        """Convert raw integer (or one-hot) labels to the y_vec the model expects.

        condition_mode:
          None / "label" — normal encoding (one-hot or embedding)
          "none"         — return a true zero tensor, effectively disabling
                           class conditioning for both encoder and decoder.
                           The zero tensor has no dependence on y values.
        """
        dev = next(self.parameters()).device
        batch_size = y.shape[0] if (hasattr(y, "shape") and y.dim() > 0) else 1

        if condition_mode == "none":
            y_input_dim = (self.disease_embed_dim
                           if self.condition_type == "embedding"
                           else self.y_dim)
            return torch.zeros(batch_size, y_input_dim, device=dev)

        # Handle 2-D one-hot input: convert to integer indices
        if hasattr(y, "dim") and y.dim() == 2 and y.size(1) == self.y_dim:
            y = y.argmax(dim=1)

        y = y.long().to(dev)
        if self.condition_type == "embedding":
            return self.disease_embedding(y)
        else:
            return _one_hot(y, self.y_dim, dev)

    # ── Generation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def sample(self, n, data_loader, condition_mode=None, device="cpu"):
        """Draw n samples. Returns (fake_data np, fake_label np)."""
        self.eval()
        fake_data, fake_label = [], []
        total = 0
        while total <= n:
            for data_x, data_y in data_loader:
                if total > n:
                    break
                z = torch.randn([data_x.shape[0], self.z_dim], device=device)
                y_vec = self._encode_condition(data_y.to(device), condition_mode=condition_mode)
                fake_data.append(self.decode(z, y_vec).cpu().numpy())
                fake_label.append(data_y.cpu().numpy())
                total += len(data_x)
        return np.concatenate(fake_data)[:n], np.concatenate(fake_label)[:n]

    @torch.no_grad()
    def generate_for_subtype(self, subtype_label: int, num_samples: int = 100,
                              condition_mode=None, device="cpu"):
        """Generate num_samples from a specific class label."""
        self.eval()
        y_raw = torch.tensor([subtype_label] * num_samples, dtype=torch.long)
        y_vec = self._encode_condition(y_raw.to(device), condition_mode=condition_mode)
        z = torch.randn([num_samples, self.z_dim], device=device)
        fake_data = self.decode(z, y_vec).cpu().numpy()
        fake_label = np.array([subtype_label] * num_samples)
        return fake_data, fake_label
