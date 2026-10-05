"""The differentiable pieces of MeLoMIA-E2E: a functional CVAE probe, its
adaptation to one synthetic dataset, and the loss readout the head consumes.

Everything here takes the probe's weights as an explicit dict of tensors, so
the weights after adaptation are a differentiable function of the shared
initialisation and of the learned step sizes.  The architecture and the
parameter names are `generators.cvae_model.CVAE`'s, run unconditionally (the
condition vector is zero, as in MeLoMIA-CVAE's probe), so the two attacks read
the same kind of model.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as Fn

from ...generators.cvae_model import CVAE

#: Posterior temperatures of the readout (MeLoMIA-CVAE's sweep).
TEMPERATURES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)

_UNUSED = ("disease_embedding.weight", "fc_feat_y.0.weight", "dec_y.0.weight")


def init_params(x_dim: int, z_dim: int, seed: int, device: str) -> dict:
    """Fresh probe weights, with the CVAE's own initialisation and names."""
    torch.manual_seed(seed)
    net = CVAE(x_dim=x_dim, y_dim=1, z_dim=z_dim)
    return {k: v.detach().clone().to(device).requires_grad_(True)
            for k, v in net.named_parameters() if k not in _UNUSED}


def _lin(p: dict, name: str, x):
    return Fn.linear(x, p[name + ".weight"], p[name + ".bias"])


def encode(p: dict, x):
    h = torch.relu(_lin(p, "fc_feat_x.0", x))
    h = torch.relu(_lin(p, "fc_feat_x.2", h))
    h = torch.relu(_lin(p, "fc_feat_x.4", h))
    hy = torch.relu(p["fc_feat_y.0.bias"]).expand(len(x), -1)   # zero condition
    feat = torch.relu(_lin(p, "fc_feat_all.0", torch.cat([h, hy], dim=1)))
    return _lin(p, "fc_mu", feat), _lin(p, "fc_logvar", feat).clamp(-12.0, 6.0)


def decode(p: dict, z):
    hz = torch.relu(_lin(p, "dec_z.0", z))
    hy = torch.relu(p["dec_y.0.bias"]).expand(len(z), -1)
    h = torch.relu(_lin(p, "dec.0", torch.cat([hz, hy], dim=1)))
    return _lin(p, "dec.4", torch.relu(_lin(p, "dec.2", h)))


def generative_loss(p: dict, x, eps, beta: float):
    """The CVAE's training loss: reconstruction MSE + beta * KL."""
    mu, logvar = encode(p, x)
    rec = decode(p, mu + torch.exp(0.5 * logvar) * eps)
    kl = (-0.5 * (1 + logvar - mu ** 2 - logvar.exp())).sum(dim=1).mean()
    return ((rec - x) ** 2).mean() + beta * kl


def noise_bank(n_steps: int, n_rows: int, z_dim: int, seed: int, device: str):
    """Reparameterisation noise of the adaptation steps, fixed by seed.

    The same bank serves every shadow and the proxy, so two adaptations differ
    only in the synthetic data they were fitted to.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(n_steps, n_rows, z_dim, generator=g).to(device)


def adapt(p0: dict, log_lr: dict, x_syn, eps, beta: float,
          create_graph: bool, batch: int | None = None, optimizer: str = "adam",
          grad_from: int = 0) -> dict:
    """Fit the probe to one synthetic dataset, starting from the shared weights.

    `len(eps)` gradient steps on the generative loss, each tensor with its own
    learned step size.  With `create_graph` the result stays differentiable
    with respect to `p0` and `log_lr` (second order); without it the steps are
    constants and only the identity path to `p0` carries gradient.

    `optimizer`  "adam" (bias-corrected, betas 0.9 / 0.999) or "sgd".
    `batch`      rows per step, taken in a fixed cyclic order; None is full batch.
    `grad_from`  steps before this index are taken without a graph (truncated
                 backpropagation), which bounds memory for long adaptations.
    """
    p = p0
    names = list(p0)
    n = len(x_syn)
    m = {k: torch.zeros_like(p0[k]) for k in names}
    v = {k: torch.zeros_like(p0[k]) for k in names}
    b1, b2 = 0.9, 0.999
    for t in range(len(eps)):
        graph = create_graph and t >= grad_from
        if batch is None or batch >= n:
            xb, eb = x_syn, eps[t]
        else:
            idx = (torch.arange(batch, device=x_syn.device) + t * batch) % n
            xb, eb = x_syn[idx], eps[t, :batch]
        loss = generative_loss(p, xb, eb, beta)
        grads = torch.autograd.grad(loss, [p[k] for k in names], create_graph=graph)
        new = {}
        for k, g in zip(names, grads):
            if optimizer == "sgd":
                step = g
            else:
                m[k] = b1 * m[k] + (1 - b1) * g
                v[k] = b2 * v[k] + (1 - b2) * g * g
                step = (m[k] / (1 - b1 ** (t + 1))) / ((v[k] / (1 - b2 ** (t + 1)) + 1e-12).sqrt() + 1e-8)
            new[k] = p[k] - log_lr[k].exp() * step
        p = new
        if create_graph and not graph:
            # Truncated: carry values only, but keep the identity path to p0 so
            # the shared weights still receive first-order gradient.
            p = {k: p0[k] + (p[k] - p0[k]).detach() for k in names}
            m = {k: m[k].detach() for k in names}
            v = {k: v[k].detach() for k in names}
    return p


def readout(p: dict, x, bank, temperatures=TEMPERATURES) -> torch.Tensor:
    """Loss signature of every record under one model, differentiable in `p`.

    MeLoMIA-CVAE's evidence, kept differentiable: the log reconstruction error
    at each posterior temperature (its mean, spread, minimum and maximum over
    the frozen latent draws), the KL of each latent coordinate, and the norms
    of the posterior mean and scale.
    """
    mu, logvar = encode(p, x)
    sigma = torch.exp(0.5 * logvar)
    n, n_draw = len(x), len(bank)
    cols = []
    for alpha in temperatures:
        if alpha == 0:
            cols.append(((decode(p, mu) - x) ** 2).mean(dim=1, keepdim=True).add(1e-12).log())
            continue
        z = mu.unsqueeze(1) + alpha * sigma.unsqueeze(1) * bank.unsqueeze(0)
        rec = decode(p, z.reshape(n * n_draw, -1)).reshape(n, n_draw, -1)
        lg = ((rec - x.unsqueeze(1)) ** 2).mean(dim=2).add(1e-12).log()
        cols.append(torch.stack([lg.mean(1), lg.std(1), lg.min(1).values, lg.max(1).values], dim=1))
    kl = -0.5 * (1 + logvar - mu ** 2 - logvar.exp())
    cols += [kl, mu.norm(dim=1, keepdim=True), sigma.norm(dim=1, keepdim=True)]
    return torch.cat(cols, dim=1)


def readout_dim(z_dim: int, temperatures=TEMPERATURES) -> int:
    return sum(1 if a == 0 else 4 for a in temperatures) + z_dim + 2


def standardise(F: torch.Tensor) -> torch.Tensor:
    """Standardise each feature over the candidate records of one model."""
    return (F - F.mean(dim=0, keepdim=True)) / (F.std(dim=0, keepdim=True) + 1e-6)


def auc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))
