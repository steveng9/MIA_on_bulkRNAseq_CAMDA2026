"""CVAE backend for the unified MIA pipeline."""

import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .. import config
from ..data_utils import (
    fit_quantile_scaler,
    build_label_predictor,
    get_real_labels_all,
    get_real_labels_for_split,
)
from ..cvae.shadow_model import build_cvae, train_cvae_shadow, generate_cvae_synthetic
from ..cvae.loss_features import extract_cvae_features, prepare_cvae_features
from .base import MIABackend


class CVAEBackend(MIABackend):
    """CVAE backend.

    By default uses ground-truth subtype labels (CVAE_LABEL_MODE='real').
    For submission inference on challenge synthetic data (no real labels),
    falls back to KNN automatically.
    """

    @property
    def name(self) -> str:
        return "cvae"

    # ── Label helpers (override base-class dummy defaults) ────────────────────

    def get_train_labels(self, split_no, X_train):
        if config.CVAE_LABEL_MODE == "real":
            return get_real_labels_for_split(self.dataset_name, split_no)
        if config.CVAE_LABEL_MODE == "none":
            return np.zeros(len(X_train), dtype=np.int64)
        # knn fallback
        return build_label_predictor(self.dataset_name)(X_train).astype(np.int64)

    def get_all_labels(self, X_real_np):
        if config.CVAE_LABEL_MODE == "real":
            return get_real_labels_all(self.dataset_name)
        if config.CVAE_LABEL_MODE == "none":
            return np.zeros(len(X_real_np), dtype=np.int64)
        return build_label_predictor(self.dataset_name)(X_real_np).astype(np.int64)

    def get_submission_labels(self, X_syn):
        # Real labels are never available for challenge synthetic data → KNN or none.
        if config.CVAE_LABEL_MODE == "none":
            return np.zeros(len(X_syn), dtype=np.int64)
        return build_label_predictor(self.dataset_name)(X_syn).astype(np.int64)

    # ── Shadow training / loading ─────────────────────────────────────────────

    def train_shadow(self, X_train, y_int, save_path, split_no, device):
        model, scaler = train_cvae_shadow(
            X_train, y_int, self.dataset_name, save_path,
            split_no=split_no, device=device,
        )
        return model, scaler

    def load_shadow(self, save_path, device):
        model = build_cvae(self.dataset_name, device)
        model.load_state_dict(torch.load(save_path, map_location=device))
        model.eval()
        return model

    # ── Synthetic generation ──────────────────────────────────────────────────

    def generate_synthetic(self, model, X_train_scaled, y_int, n_samples, scaler,
                           split_no, device):
        X_syn_scaled, y_syn = generate_cvae_synthetic(
            model, self.dataset_name, X_train_scaled, y_int,
            n_samples=n_samples, device=device,
        )
        X_syn_raw = scaler.inverse_transform(
            X_syn_scaled.astype(np.float64)
        ).astype(np.float32)
        return X_syn_raw, y_syn

    # ── Feature extraction ────────────────────────────────────────────────────

    def extract_features(self, model, X_scaled, y_int, device):
        rec_losses, per_dim_kl = extract_cvae_features(
            model, X_scaled, y_int, device=device
        )
        return rec_losses, per_dim_kl

    # ── Feature serialisation ─────────────────────────────────────────────────

    def save_raw_features(self, raw_features, path, y_member, sample_ids):
        rec_losses, per_dim_kl = raw_features
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path,
                 rec_losses=rec_losses, per_dim_kl=per_dim_kl,
                 y_member=y_member,
                 sample_ids=np.array(sample_ids, dtype=object))

    def load_raw_features(self, path):
        d = np.load(path, allow_pickle=True)
        return (d["rec_losses"], d["per_dim_kl"]), d["y_member"]

    def features_valid(self, path: str) -> bool:
        try:
            d = np.load(path, allow_pickle=True)
            expected = len(config.CVAE_TEMP_LIST) * config.CVAE_N_SAMPLES
            return int(d["rec_losses"].shape[1]) == expected
        except Exception:
            return False

    def prepare_features(self, raw_features) -> np.ndarray:
        rec_losses, per_dim_kl = raw_features
        return prepare_cvae_features(rec_losses, per_dim_kl)

    # ── MLP config ────────────────────────────────────────────────────────────

    @property
    def mlp_kwargs(self) -> dict:
        return {
            "hidden_dim":   config.CVAE_MLP_HIDDEN_DIM,
            "dropout":      config.CVAE_MLP_DROPOUT,
            "weight_decay": config.CVAE_MLP_WEIGHT_DECAY,
            "epochs":       config.CVAE_MLP_EPOCHS,
            "lr":           config.CVAE_MLP_LR,
            "batch_size":   config.CVAE_MLP_BATCH_SIZE,
        }
