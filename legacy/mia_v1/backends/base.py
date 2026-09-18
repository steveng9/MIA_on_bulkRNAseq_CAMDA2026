"""Abstract base class for MIA attack backends (ND and CVAE)."""

import os
from abc import ABC, abstractmethod

import numpy as np

from .. import config


class MIABackend(ABC):
    """One instance per (model_type, dataset) pair.

    Encapsulates everything that differs between NoisyDiffusion and CVAE:
    shadow training, synthetic generation, feature extraction, and feature
    serialisation.  All path construction and label resolution also live here
    so the shared pipeline.py stays completely model-agnostic.
    """

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name

    # ── Identity ──────────────────────────────────────────────────────────────

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in path construction: 'nd' or 'cvae'."""

    # ── Path helpers (all relative to PIPELINE_BASE) ─────────────────────────

    def real_shadow_path(self, split_no: int) -> str:
        return os.path.join(config.PIPELINE_REAL_SHADOW, self.name,
                            self.dataset_name, f"split_{split_no}.pt")

    def synth_shadow_path(self, split_no: int) -> str:
        return os.path.join(config.PIPELINE_SYNTH_SHADOW, self.name,
                            self.dataset_name, f"split_{split_no}.pt")

    def synthetic_data_path(self, split_no: int) -> str:
        return os.path.join(config.PIPELINE_SYNTHETIC, self.name,
                            self.dataset_name, f"split_{split_no}.npz")

    def features_path(self, split_no: int, source: str) -> str:
        """source: 'real' (real-shadow features) or 'synth' (synth-shadow features)."""
        return os.path.join(config.PIPELINE_FEATURES, source, self.name,
                            self.dataset_name, f"split_{split_no}.npz")

    def classifier_dir(self, source: str) -> str:
        return os.path.join(config.PIPELINE_CLASSIFIERS, source, self.name,
                            self.dataset_name)

    def target_proxy_path(self) -> str:
        return os.path.join(config.PIPELINE_TARGET_PROXY, self.name,
                            self.dataset_name, "proxy.pt")

    # ── Label helpers (default = unconditional dummy label, correct for ND) ──

    def get_train_labels(self, split_no: int, X_train: np.ndarray) -> np.ndarray:
        """Labels for real-data shadow training on a given split's training subset."""
        return np.full(len(X_train), config.DUMMY_LABEL, dtype=np.int64)

    def get_all_labels(self, X_real_np: np.ndarray) -> np.ndarray:
        """Labels for all real samples (used in feature extraction)."""
        return np.full(len(X_real_np), config.DUMMY_LABEL, dtype=np.int64)

    def get_submission_labels(self, X_syn: np.ndarray) -> np.ndarray:
        """Labels for challenge proxy training on the challenge synthetic data."""
        return np.full(len(X_syn), config.DUMMY_LABEL, dtype=np.int64)

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def train_shadow(self, X_train: np.ndarray, y_int: np.ndarray,
                     save_path: str, split_no: int, device: str):
        """Train a shadow model on X_train (raw, unscaled).

        Returns (model, scaler) where model is an opaque backend-specific object
        and scaler was fitted on X_train.
        """

    @abstractmethod
    def load_shadow(self, save_path: str, device: str):
        """Load a saved shadow model.  Returns the same opaque object as train_shadow."""

    @abstractmethod
    def generate_synthetic(self, model, X_train_scaled: np.ndarray,
                           y_int: np.ndarray, n_samples: int,
                           scaler, split_no: int, device: str):
        """Produce synthetic gene-expression samples.

        Parameters
        ----------
        model          : opaque object from load_shadow / train_shadow
        X_train_scaled : normalised training data used to train model
        y_int          : integer class labels for training data
        n_samples      : how many samples to generate
        scaler         : QuantileTransformer fitted on X_train (for inverse-transform)
        split_no       : split index (used by ND backend to read Blue Team repo)
        device         : torch device string

        Returns
        -------
        X_syn : np.ndarray (n_samples, 978) — in ORIGINAL (unscaled) space
        y_syn : np.ndarray (n_samples,) int64 — class labels
        """

    @abstractmethod
    def extract_features(self, model, X_scaled: np.ndarray,
                         y_int: np.ndarray, device: str):
        """Extract raw MIA features for every sample in X_scaled.

        Returns a backend-specific raw-feature object (opaque to the pipeline).
        """

    @abstractmethod
    def save_raw_features(self, raw_features, path: str,
                          y_member: np.ndarray, sample_ids: list):
        """Serialise raw_features + metadata to an .npz file."""

    @abstractmethod
    def load_raw_features(self, path: str):
        """Load raw features from disk.  Returns (raw_features, y_member)."""

    @abstractmethod
    def prepare_features(self, raw_features) -> np.ndarray:
        """Convert backend-specific raw features to a 2-D MLP input array."""

    def features_valid(self, path: str) -> bool:
        """Return True if the saved feature file matches the current config.

        Override in backends where the raw feature shape depends on config params
        that can change across runs (e.g. N_SAMPLES, TEMP_LIST for CVAE).
        The default returns True (assume compatible).
        """
        return True

    @property
    @abstractmethod
    def mlp_kwargs(self) -> dict:
        """Keyword arguments forwarded verbatim to train_classifier."""
