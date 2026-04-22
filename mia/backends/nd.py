"""NoisyDiffusion backend for the unified MIA pipeline."""

import os
from dataclasses import dataclass

import numpy as np
import torch

from .. import config
from ..data_utils import load_nd_synthetic, encode_labels, fit_quantile_scaler
from ..nd.shadow_model import (
    build_model, build_diffusion_trainer, train_shadow_model,
)
from ..nd.loss_features import extract_loss_features, prepare_features
from .base import MIABackend

# Maximum number of synthetic splits available from the Blue Team repo.
_ND_MAX_SYNTH_SPLITS = 5


@dataclass
class NDShadowModel:
    """Bundles model + diffusion_trainer so the pipeline can treat them as one object."""
    model: object
    diff_trainer: object


class NDBackend(MIABackend):
    """NoisyDiffusion backend.

    Shadow models are unconditional (y_int is always a dummy label).
    Synthetic data comes from the Blue Team repo (splits 1-5 only).
    """

    @property
    def name(self) -> str:
        return "nd"

    # ── Shadow training / loading ─────────────────────────────────────────────

    def train_shadow(self, X_train, y_int, save_path, split_no, device):
        # y_int is ignored — ND shadows are unconditional
        model, diff_trainer, scaler = train_shadow_model(
            X_train, save_path, split_no=split_no, device=device
        )
        return NDShadowModel(model, diff_trainer), scaler

    def load_shadow(self, save_path, device):
        model = build_model(config.UNCONDITIONAL_NUM_CLASSES, device)
        model.load_state_dict(torch.load(save_path, map_location=device))
        model.eval()
        return NDShadowModel(model, build_diffusion_trainer(device))

    # ── Synthetic generation ──────────────────────────────────────────────────

    def generate_synthetic(self, model, X_train_scaled, y_int, n_samples, scaler,
                           split_no, device):
        if split_no > _ND_MAX_SYNTH_SPLITS:
            raise NotImplementedError(
                f"ND synthetic data is only available for splits 1-{_ND_MAX_SYNTH_SPLITS} "
                f"(Blue Team repo).  Requested split {split_no}."
            )
        ds = config.DATASETS[self.dataset_name]
        X_syn, y_str = load_nd_synthetic(self.dataset_name, split_no)
        y_syn = encode_labels(y_str, ds["label_list"])
        return X_syn, y_syn

    # ── Feature extraction ────────────────────────────────────────────────────

    def extract_features(self, shadow: NDShadowModel, X_scaled, y_int, device):
        # y_int is the dummy label array (unused inside extract_loss_features)
        return extract_loss_features(
            shadow.model, shadow.diff_trainer, X_scaled, y_int, device=device
        )

    # ── Feature serialisation ─────────────────────────────────────────────────

    def save_raw_features(self, raw_features, path, y_member, sample_ids):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, features=raw_features, y_member=y_member,
                 sample_ids=np.array(sample_ids, dtype=object))

    def load_raw_features(self, path):
        d = np.load(path, allow_pickle=True)
        return d["features"], d["y_member"]

    def features_valid(self, path: str) -> bool:
        try:
            d = np.load(path, allow_pickle=True)
            expected = len(config.T_LIST) * config.N_NOISE
            return int(d["features"].shape[1]) == expected
        except Exception:
            return False

    def prepare_features(self, raw_features) -> np.ndarray:
        return prepare_features(raw_features)

    # ── MLP config ────────────────────────────────────────────────────────────

    @property
    def mlp_kwargs(self) -> dict:
        return {
            "hidden_dim":   config.MLP_HIDDEN_DIM,
            "dropout":      config.MLP_DROPOUT,
            "weight_decay": config.MLP_WEIGHT_DECAY,
            "epochs":       config.MLP_EPOCHS,
            "lr":           config.MLP_LR,
            "batch_size":   config.MLP_BATCH_SIZE,
        }
