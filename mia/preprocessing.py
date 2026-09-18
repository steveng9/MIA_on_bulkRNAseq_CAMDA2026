"""Feature scaling shared by generators and attacks.

Which scaler a model was trained under is part of its identity: loss features
extracted under different quantile boundaries are not comparable, so scalers
are always persisted next to the weights rather than refitted at inference.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, StandardScaler


def make_scaler(kind: str):
    if kind == "quantile":
        return QuantileTransformer(output_distribution="normal")
    if kind == "standard":
        return StandardScaler()
    if kind == "minmax":
        return MinMaxScaler()
    if kind == "none":
        return None
    raise ValueError(f"Unknown scaler {kind!r}")


def fit_scaler(kind: str, X: np.ndarray):
    """Returns (scaler_or_None, X_transformed float32)."""
    scaler = make_scaler(kind)
    if scaler is None:
        return None, np.asarray(X, dtype=np.float32)
    Xt = scaler.fit_transform(np.asarray(X, dtype=np.float64)).astype(np.float32)
    return scaler, Xt


def apply_scaler(scaler, X: np.ndarray) -> np.ndarray:
    if scaler is None:
        return np.asarray(X, dtype=np.float32)
    return scaler.transform(np.asarray(X, dtype=np.float64)).astype(np.float32)


def invert_scaler(scaler, X: np.ndarray) -> np.ndarray:
    if scaler is None:
        return np.asarray(X, dtype=np.float32)
    return scaler.inverse_transform(np.asarray(X, dtype=np.float64)).astype(np.float32)


def save_scaler(scaler, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(scaler, f)


def load_scaler(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def scaler_path(model_path: Path) -> Path:
    """Convention: weights at foo.pt keep their scaler at foo_scaler.pkl."""
    return Path(model_path).with_suffix("").with_name(Path(model_path).stem + "_scaler.pkl")
