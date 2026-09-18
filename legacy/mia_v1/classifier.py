"""MLP and Random Forest classifiers for membership inference."""

import os
import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from . import config

RF_PARAMS = {
    "n_estimators":     500,
    "max_depth":        8,
    "max_features":     "sqrt",
    "min_samples_leaf": 20,
    "max_samples":      0.7,
}


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
    dropout=None, hidden_dim=None, weight_decay=None, batch_size=None,
):
    """Train a MembershipMLP on loss features.

    Parameters
    ----------
    X_train, y_train : np.ndarray – training loss features and membership labels
    X_val, y_val     : np.ndarray – validation set
    epochs, lr, device, save_dir : training knobs (default to config values)
    dropout, hidden_dim, weight_decay, batch_size : model/optimizer overrides
        (default to config.MLP_* values; pass explicitly for CVAE attack)

    Returns
    -------
    model : trained MembershipMLP (on CPU)
    history : dict with 'train_loss', 'val_loss', 'val_tpr_at_10fpr' per epoch
    """
    epochs       = epochs       or config.MLP_EPOCHS
    lr           = lr           or config.MLP_LR
    device       = device       or config.DEVICE
    dropout      = config.MLP_DROPOUT      if dropout      is None else dropout
    hidden_dim   = config.MLP_HIDDEN_DIM   if hidden_dim   is None else hidden_dim
    weight_decay = config.MLP_WEIGHT_DECAY if weight_decay is None else weight_decay
    batch_size   = batch_size   or config.MLP_BATCH_SIZE

    model = MembershipMLP(input_dim=X_train.shape[1],
                          hidden_dim=hidden_dim,
                          dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCELoss()

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

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
    """Load the default ND MembershipMLP from CLASSIFIER_DIR."""
    device = device or "cpu"
    model = MembershipMLP(dropout=config.MLP_DROPOUT)
    ckpt_path = os.path.join(config.CLASSIFIER_DIR, "mlp_best.pt")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model


def load_classifier_from_dir(save_dir, input_dim, hidden_dim=None, dropout=None, device=None):
    """Load a MembershipMLP checkpoint from an arbitrary directory.

    Parameters
    ----------
    save_dir   : str  — directory containing mlp_best.pt
    input_dim  : int  — must match the checkpoint's first layer width
    hidden_dim : int  — defaults to config.MLP_HIDDEN_DIM
    dropout    : float — defaults to config.MLP_DROPOUT
    device     : str
    """
    device     = device     or "cpu"
    hidden_dim = hidden_dim or config.MLP_HIDDEN_DIM
    dropout    = config.MLP_DROPOUT if dropout is None else dropout

    model = MembershipMLP(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
    ckpt_path = os.path.join(save_dir, "mlp_best.pt")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model


# ── Random Forest ─────────────────────────────────────────────────────────────

def train_rf_classifier(X_train, y_train, X_val, y_val, save_dir=None, params=None):
    """Train a RandomForestClassifier and save it with joblib.

    Returns (rf_model, metrics_dict) where metrics_dict has 'val_tpr_at_10fpr'.
    """
    from sklearn.ensemble import RandomForestClassifier

    params   = params   or RF_PARAMS
    save_dir = save_dir or config.CLASSIFIER_DIR
    os.makedirs(save_dir, exist_ok=True)

    print(f"  [RF] fitting RandomForestClassifier  params={params}  "
          f"train={len(X_train)}  val={len(X_val)}  dim={X_train.shape[1]}")
    rf = RandomForestClassifier(**params, random_state=config.SEED, n_jobs=-1)
    rf.fit(X_train, y_train.astype(int))

    val_scores = rf.predict_proba(X_val)[:, 1]
    tpr = _tpr_at_fpr(y_val.astype(int), val_scores, fpr_target=0.10)
    acc = ((val_scores > 0.5).astype(int) == y_val.astype(int)).mean()
    print(f"  [RF] val TPR@10%FPR={tpr:.4f}  ACC={acc:.4f}")

    ckpt_path = os.path.join(save_dir, "rf_best.pkl")
    joblib.dump(rf, ckpt_path)
    print(f"  [RF] saved → {ckpt_path}")

    return rf, {"val_tpr_at_10fpr": tpr, "val_acc": acc}


def load_rf_classifier(save_dir):
    ckpt_path = os.path.join(save_dir, "rf_best.pkl")
    return joblib.load(ckpt_path)


# ── Unified scorer ────────────────────────────────────────────────────────────

def predict_scores(clf, X):
    """Return a 1-D score array for X, dispatching on config.CLASSIFIER_TYPE."""
    if config.CLASSIFIER_TYPE == "rf":
        return clf.predict_proba(X)[:, 1]
    clf.eval()
    with torch.no_grad():
        return clf(torch.tensor(X, dtype=torch.float32)).squeeze(1).numpy()
