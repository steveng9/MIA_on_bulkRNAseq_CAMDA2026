"""3-layer MLP for membership inference classification."""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from . import config


class MembershipMLP(nn.Module):
    """MLP: input → hidden → hidden → 1 (tanh activations, sigmoid output)."""

    def __init__(self, input_dim=None, hidden_dim=None, dropout=0.0):
        super().__init__()
        input_dim = input_dim or config.MLP_INPUT_DIM
        hidden_dim = hidden_dim or config.MLP_HIDDEN_DIM
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return torch.sigmoid(self.net(x))


def _tpr_at_fpr(y_true, y_score, fpr_target=0.10):
    """Compute TPR @ a given FPR threshold."""
    # Sort by descending score
    desc = np.argsort(-y_score)
    y_sorted = y_true[desc]

    n_pos = y_sorted.sum()
    n_neg = len(y_sorted) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0

    fp = 0
    tp = 0
    best_tpr = 0.0
    for label in y_sorted:
        if label == 1:
            tp += 1
        else:
            fp += 1
        fpr = fp / n_neg
        if fpr > fpr_target:
            break
        best_tpr = tp / n_pos
    return best_tpr


def train_classifier(
    X_train, y_train, X_val, y_val,
    epochs=None, lr=None, device=None, save_dir=None,
):
    """Train a MembershipMLP on loss features.

    Parameters
    ----------
    X_train, y_train : np.ndarray – training loss features and membership labels
    X_val, y_val     : np.ndarray – validation set
    epochs : int
    lr : float
    device : str

    Returns
    -------
    model : trained MembershipMLP (on CPU)
    history : dict with 'train_loss', 'val_loss', 'val_tpr_at_10fpr' per epoch
    """
    epochs = epochs or config.MLP_EPOCHS
    lr = lr or config.MLP_LR
    device = device or config.DEVICE

    model = MembershipMLP(input_dim=X_train.shape[1], dropout=config.MLP_DROPOUT).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=config.MLP_WEIGHT_DECAY)
    criterion = nn.BCELoss()

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=config.MLP_BATCH_SIZE, shuffle=True)

    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_np = y_val.astype(np.float32)

    history = {"train_loss": [], "val_loss": [], "val_tpr_at_10fpr": []}
    best_metric = -1.0
    best_state = None

    for epoch in range(epochs):
        # ── Train ────────────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).squeeze(1)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= len(train_ds)

        # ── Validate ─────────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t).squeeze(1).cpu().numpy()
        val_loss = float(nn.BCELoss()(
            torch.tensor(val_pred), torch.tensor(y_val_np),
        ))
        tpr = _tpr_at_fpr(y_val.astype(int), val_pred, fpr_target=0.10)

        history["train_loss"].append(epoch_loss)
        history["val_loss"].append(val_loss)
        history["val_tpr_at_10fpr"].append(tpr)

        if tpr > best_metric:
            best_metric = tpr
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"  [MLP] epoch {epoch + 1:4d}/{epochs}  "
                  f"train_loss={epoch_loss:.4f}  val_loss={val_loss:.4f}  "
                  f"TPR@10%FPR={tpr:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.cpu()

    # ── Save ─────────────────────────────────────────────────────────────
    save_dir = save_dir or config.CLASSIFIER_DIR
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, "mlp_best.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"  [MLP] best TPR@10%FPR = {best_metric:.4f}  →  saved to {ckpt_path}")

    return model, history


def load_classifier(device=None):
    """Load a trained MembershipMLP from disk."""
    device = device or "cpu"
    model = MembershipMLP(dropout=config.MLP_DROPOUT)
    ckpt_path = os.path.join(config.CLASSIFIER_DIR, "mlp_best.pt")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model
