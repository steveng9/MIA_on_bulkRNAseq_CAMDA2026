"""Train shadow NoisyDiffusion models on synthetic data."""

import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import QuantileTransformer
from imblearn.over_sampling import SMOTE
from collections import Counter

# Add NoisyDiffusion source to path so we can import the model classes.
_nd_brca = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..", "CAMDA25_NoisyDiffusion", "TCGA-BRCA",
)
if _nd_brca not in sys.path:
    sys.path.insert(0, _nd_brca)
from model import EmbeddedDiffusion, DiffusionTrainer

from . import config
from .data_utils import load_nd_synthetic, encode_labels, fit_quantile_scaler


def build_model(num_classes, device):
    """Instantiate an EmbeddedDiffusion matching the target config."""
    model = EmbeddedDiffusion(
        input_dim=config.INPUT_DIM,
        num_classes=num_classes,
        num_timesteps=config.NUM_TIMESTEPS,
        hidden_dims=list(config.HIDDEN_DIMS),
        dropout=config.DROPOUT,
        attn_num_tokens=config.ATTN_NUM_TOKENS,
        attn_num_heads=config.ATTN_NUM_HEADS,
        time_embedding_dim=config.TIME_EMBEDDING_DIM,
        label_embedding_dim=config.LABEL_EMBEDDING_DIM,
        num_groups=config.NUM_GROUPS,
    )
    return model.to(device)


def build_diffusion_trainer(device):
    """Instantiate a DiffusionTrainer matching the target noise schedule."""
    return DiffusionTrainer(
        num_timesteps=config.NUM_TIMESTEPS,
        beta_schedule=config.BETA_SCHEDULE,
        linear_beta_start=config.LINEAR_BETA_START,
        linear_beta_end=config.LINEAR_BETA_END,
        cosine_s=0.008,
        power_sigma_max=1.0,
        power_sigma_min=0.005,
        power_rho_expo=7.0,
        device=device,
    )


def train_shadow_model(dataset_name, split_no, device=None):
    """Train a shadow model on the NoisyDiffusion repo's synthetic data for one split.

    Replicates the target training pipeline: SMOTE upsampling → QuantileTransformer
    → OneCycleLR → DP noise.

    Returns (model, diffusion_trainer, scaler).
    """
    device = device or config.DEVICE
    ds = config.DATASETS[dataset_name]
    num_classes = ds["num_classes"]
    label_list = ds["label_list"]

    # ── Load labeled synthetic data ──────────────────────────────────────
    X_syn, y_syn_str = load_nd_synthetic(dataset_name, split_no)
    y_int = encode_labels(y_syn_str, label_list)

    # ── SMOTE upsampling ─────────────────────────────────────────────────
    print(f"[shadow {dataset_name} split {split_no}] Before SMOTE: {Counter(y_int)}")
    sampling_strategy = {i: config.SMOTE_UPSAMPLE_TO for i in range(num_classes)}
    smote = SMOTE(sampling_strategy=sampling_strategy)
    X_syn, y_int = smote.fit_resample(X_syn.astype(np.float64), y_int)
    print(f"[shadow {dataset_name} split {split_no}] After SMOTE:  {Counter(y_int)}")

    # ── QuantileTransformer ──────────────────────────────────────────────
    scaler, X_scaled = fit_quantile_scaler(X_syn)

    # ── DataLoader ───────────────────────────────────────────────────────
    ds_torch = TensorDataset(
        torch.tensor(X_scaled, dtype=torch.float32),
        torch.tensor(y_int, dtype=torch.long),
    )
    loader = DataLoader(ds_torch, batch_size=config.SHADOW_BATCH_SIZE, shuffle=True)

    # ── Model / optimizer / scheduler ────────────────────────────────────
    model = build_model(num_classes, device)
    diff_trainer = build_diffusion_trainer(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.SHADOW_LR, weight_decay=config.SHADOW_LR_WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.SHADOW_LR, epochs=config.SHADOW_EPOCHS,
        steps_per_epoch=len(loader), pct_start=config.SHADOW_LR_PCT_START,
        anneal_strategy=config.SHADOW_LR_ANNEAL_STRATEGY,
        div_factor=config.SHADOW_LR_DIV_FACTOR,
        final_div_factor=config.SHADOW_LR_FINAL_DIV_FACTOR,
    )

    # ── Training loop with early stopping ────────────────────────────────
    best_loss, no_improve, best_state = float("inf"), 0, None

    for epoch in range(config.SHADOW_EPOCHS):
        model.train()
        total_loss = 0.0
        for X_b, y_b in loader:
            loss = diff_trainer.train_step(
                model, optimizer, X_b, y_b, device,
                dp_noise_multiplier=config.SHADOW_DP_NOISE_MULTIPLIER,
                max_grad_norm=config.SHADOW_MAX_GRAD_NORM,
            )
            scheduler.step()
            total_loss += loss

        avg_loss = total_loss / len(loader)
        if epoch % 10 == 0 or epoch == config.SHADOW_EPOCHS - 1:
            print(f"  [shadow {dataset_name} split {split_no}] epoch {epoch:3d}  loss={avg_loss:.6f}")

        if config.SHADOW_EARLY_STOPPING:
            if avg_loss < best_loss - config.SHADOW_EARLY_STOPPING_MIN_DELTA:
                best_loss = avg_loss
                no_improve = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                no_improve += 1
            if no_improve >= config.SHADOW_EARLY_STOPPING_PATIENCE:
                print(f"  [shadow {dataset_name} split {split_no}] early stop @ epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    # ── Save ─────────────────────────────────────────────────────────────
    save_dir = os.path.join(config.SHADOW_MODEL_DIR, dataset_name)
    os.makedirs(save_dir, exist_ok=True)
    ckpt = os.path.join(save_dir, f"shadow_split_{split_no}.pt")
    torch.save(model.state_dict(), ckpt)
    print(f"  [shadow {dataset_name} split {split_no}] saved → {ckpt}")

    return model, diff_trainer, scaler


def load_shadow_model(dataset_name, split_no, device=None):
    """Load a saved shadow model. Returns (model, diffusion_trainer)."""
    device = device or config.DEVICE
    ds = config.DATASETS[dataset_name]
    model = build_model(ds["num_classes"], device)
    ckpt = os.path.join(config.SHADOW_MODEL_DIR, dataset_name, f"shadow_split_{split_no}.pt")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model, build_diffusion_trainer(device)
