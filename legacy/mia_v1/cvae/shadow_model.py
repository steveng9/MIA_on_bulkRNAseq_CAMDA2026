"""Train and load CVAE shadow models."""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .. import config
from ..data_utils import fit_quantile_scaler, build_label_predictor, load_real_data
from .model import CVAE


def _y_input_dim(y_dim):
    """y vector dimensionality for the configured condition_type."""
    if config.CVAE_CONDITION_TYPE == "embedding":
        return config.CVAE_DISEASE_EMBED_DIM
    return y_dim


def _get_y_int(X_raw, dataset_name, condition_mode=None):
    """Predict integer labels for X_raw using KNN on ND synthetic data.

    Returns np.ndarray (n,) int64. Falls back to zeros for 'none' mode
    (labels are ignored anyway in that mode, but we still need a placeholder).
    """
    condition_mode = condition_mode or config.CVAE_LABEL_MODE
    if condition_mode == "none":
        return np.zeros(len(X_raw), dtype=np.int64)
    # KNN predictor trained on ND synthetic data (split 1 as reference)
    predict = build_label_predictor(dataset_name, split_no=1)
    return predict(X_raw).astype(np.int64)


def build_cvae(dataset_name, device=None):
    """Instantiate a fresh CVAE matching the project config."""
    device = device or config.DEVICE
    num_classes = config.DATASETS[dataset_name]["num_classes"]
    model = CVAE(
        x_dim=config.CVAE_INPUT_DIM,
        y_dim=num_classes,
        z_dim=config.CVAE_Z_DIM,
        beta=config.CVAE_BETA,
        transform=config.CVAE_TRANSFORM,
        condition_type=config.CVAE_CONDITION_TYPE,
        disease_embed_dim=config.CVAE_DISEASE_EMBED_DIM,
    )
    return model.to(device)


def train_cvae_shadow(X_train, y_int, dataset_name, save_path, split_no, device=None):
    """Train a CVAE shadow on the provided (scaled) data subset.

    Parameters
    ----------
    X_train  : np.ndarray (n_train, INPUT_DIM) — raw (unscaled)
    y_int    : np.ndarray (n_train,) int64      — class labels (ignored if CVAE_LABEL_MODE='none')
    dataset_name : str
    save_path    : str — checkpoint path (.pt)
    split_no     : int — used as seed offset
    device       : str

    Returns
    -------
    (model, scaler)  — model on device, scaler fitted on X_train
    """
    device = device or config.DEVICE
    condition_mode = config.CVAE_LABEL_MODE

    print(f"[CVAE shadow split {split_no}] n={len(X_train)}  "
          f"label_mode={condition_mode}  device={device}")

    # Normalize
    scaler, X_scaled = fit_quantile_scaler(X_train)
    X_t = torch.tensor(X_scaled, dtype=torch.float32)
    y_t = torch.tensor(y_int, dtype=torch.long)

    ds = TensorDataset(X_t, y_t)
    loader = DataLoader(ds, batch_size=config.CVAE_BATCH_SIZE, shuffle=True)

    torch.manual_seed(config.SEED + split_no)
    np.random.seed(config.SEED + split_no)

    model = build_cvae(dataset_name, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.CVAE_LR, weight_decay=config.CVAE_LR_WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.CVAE_LR, epochs=config.CVAE_EPOCHS,
        steps_per_epoch=len(loader), pct_start=config.CVAE_LR_PCT_START,
        anneal_strategy=config.CVAE_LR_ANNEAL_STRATEGY,
        div_factor=config.CVAE_LR_DIV_FACTOR,
        final_div_factor=config.CVAE_LR_FINAL_DIV_FACTOR,
    )

    best_loss, no_improve, best_state = float("inf"), 0, None

    for epoch in range(config.CVAE_EPOCHS):
        model.train()
        total_loss = 0.0
        for X_b, y_b in loader:
            X_b = X_b.to(device)
            y_vec = model._encode_condition(y_b.to(device), condition_mode=condition_mode)
            out = model.compute_loss(X_b, y_vec)
            loss = out["loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * len(X_b)

        avg_loss = total_loss / len(ds)
        if epoch % 10 == 0 or epoch == config.CVAE_EPOCHS - 1:
            print(f"  [CVAE split {split_no}] epoch {epoch:3d}  loss={avg_loss:.6f}")

        if config.CVAE_EARLY_STOPPING:
            if avg_loss < best_loss - config.CVAE_EARLY_STOPPING_MIN_DELTA:
                best_loss = avg_loss
                no_improve = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                no_improve += 1
            if no_improve >= config.CVAE_EARLY_STOPPING_PATIENCE:
                print(f"  [CVAE split {split_no}] early stop @ epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"  [CVAE split {split_no}] saved → {save_path}")

    return model, scaler


def load_cvae_shadow(dataset_name, split_no, device=None):
    """Load a saved CVAE shadow model. Returns model (on device)."""
    device = device or config.DEVICE
    model = build_cvae(dataset_name, device)
    ckpt = os.path.join(config.CVAE_SHADOW_MODEL_DIR, dataset_name, f"shadow_split_{split_no}.pt")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model


def load_cvae_target_proxy(dataset_name, device=None):
    """Load CVAE target proxy (trained on challenge synthetic data)."""
    device = device or config.DEVICE
    model = build_cvae(dataset_name, device)
    ckpt = os.path.join(config.CVAE_SHADOW_MODEL_DIR, dataset_name, "target_proxy.pt")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model


def generate_cvae_synthetic(model, dataset_name, X_train, y_int, n_samples, device=None):
    """Generate n_samples from a trained CVAE shadow.

    Returns (X_syn np float32, y_syn np int64).
    Uses the training label distribution (cycling through the loader).
    """
    device = device or config.DEVICE
    condition_mode = config.CVAE_LABEL_MODE

    X_t = torch.tensor(X_train.astype(np.float32))
    y_t = torch.tensor(y_int, dtype=torch.long)
    loader = DataLoader(TensorDataset(X_t, y_t),
                        batch_size=config.CVAE_BATCH_SIZE, shuffle=True)

    fake_X, fake_y = model.sample(n_samples, loader,
                                  condition_mode=condition_mode, device=device)
    return fake_X.astype(np.float32), fake_y.astype(np.int64)
