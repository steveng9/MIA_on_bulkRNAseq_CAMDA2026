"""Data loading and preprocessing for the MIA pipeline.

The challenge provides:
  - Real data TSV: genes × samples (rows=978 genes, columns=sample IDs). Must transpose.
  - Synthetic data CSV: samples × genes (rows=samples, columns=gene IDs). No labels.

The NoisyDiffusion repo provides:
  - Labeled synthetic data: synthetic_data_split_{k}.csv + synthetic_labels_split_{k}.csv
  - Splits YAML: which sample IDs are train/test per fold → ground truth membership.
"""

import os
import yaml
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
from sklearn.neighbors import KNeighborsClassifier

from . import config


# ─────────────────────────────────────────────────────────────────────────────
# Loading challenge data
# ─────────────────────────────────────────────────────────────────────────────

def load_real_data(dataset_name):
    """Load the full real-data TSV (genes × samples), transpose to (samples × genes).

    Returns:
        X : pd.DataFrame, shape (n_samples, 978), index = sample IDs
        gene_names : list[str], the 978 gene ENSEMBL IDs
    """
    ds = config.DATASETS[dataset_name]
    df = pd.read_csv(ds["real_tsv"], sep="\t", index_col=0)
    # df is (978 genes × n_samples) → transpose to (n_samples × 978)
    X = df.T
    gene_names = list(df.index)
    return X, gene_names


def load_challenge_synthetic(dataset_name):
    """Load the challenge's synthetic CSV (samples × genes, no labels).

    Returns:
        X : np.ndarray, shape (n_syn, 978)
    """
    ds = config.DATASETS[dataset_name]
    df = pd.read_csv(ds["synthetic_csv"])
    return df.values.astype(np.float32)


def load_reference_data(dataset_name):
    """Load the reference dataset (COMBINED only). Same format as real data TSV.

    Returns:
        X : pd.DataFrame (n_ref, 978), index = sample IDs
    """
    ds = config.DATASETS[dataset_name]
    path = ds.get("reference_tsv")
    if path is None or not os.path.exists(path):
        return None
    df = pd.read_csv(path, sep="\t", index_col=0)
    return df.T


# ─────────────────────────────────────────────────────────────────────────────
# Loading NoisyDiffusion repo data (for training / ground truth)
# ─────────────────────────────────────────────────────────────────────────────

def load_splits_yaml(dataset_name):
    """Load the splits YAML → dict mapping split name to train/test sample IDs."""
    ds = config.DATASETS[dataset_name]
    with open(ds["splits_yaml"]) as f:
        return yaml.safe_load(f)["splits"]


def load_nd_synthetic(dataset_name, split_no):
    """Load NoisyDiffusion repo's labeled synthetic data for a split.

    Returns:
        X : np.ndarray (n_syn, 978)
        y_str : np.ndarray (n_syn,) string labels
    """
    ds = config.DATASETS[dataset_name]
    d = ds["nd_synthetic_dir"]
    X = pd.read_csv(os.path.join(d, f"synthetic_data_split_{split_no}.csv")).values.astype(np.float32)
    y = pd.read_csv(os.path.join(d, f"synthetic_labels_split_{split_no}.csv")).values.ravel()
    return X, y


def get_membership_labels(dataset_name, split_no):
    """Determine ground-truth membership for one CV split.

    Members (1) = samples in the training set for this split.
    Non-members (0) = samples in the test set for this split.

    Returns:
        sample_ids : list[str]  – all sample IDs in the real data
        y_member : np.ndarray (n_samples,) – 0/1 membership labels
    """
    X_real, _ = load_real_data(dataset_name)
    all_ids = list(X_real.index)

    splits = load_splits_yaml(dataset_name)
    split_key = f"split_{split_no}"
    test_ids = set(splits[split_key]["test_index"])

    y_member = np.array([0 if sid in test_ids else 1 for sid in all_ids], dtype=np.int64)
    return all_ids, y_member


# ─────────────────────────────────────────────────────────────────────────────
# Label encoding / inference
# ─────────────────────────────────────────────────────────────────────────────

def encode_labels(y_str, label_list):
    """Map string labels to integer indices."""
    mapping = {c: i for i, c in enumerate(label_list)}
    return np.array([mapping[l] for l in y_str], dtype=np.int64)


def infer_labels_knn(X_unlabeled, X_labeled, y_labeled_int, k=5):
    """Predict integer labels for unlabeled samples using KNN on labeled data.

    Both X arrays should be in the same feature space (raw gene expression).
    """
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_labeled, y_labeled_int)
    return knn.predict(X_unlabeled)


def build_label_predictor(dataset_name, split_no=1):
    """Train a KNN label predictor on the NoisyDiffusion repo's labeled synthetic data.

    Returns a function: predict(X_raw) → y_int
    """
    ds = config.DATASETS[dataset_name]
    X_syn, y_syn_str = load_nd_synthetic(dataset_name, split_no)
    y_syn_int = encode_labels(y_syn_str, ds["label_list"])

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_syn, y_syn_int)
    return knn.predict


# ─────────────────────────────────────────────────────────────────────────────
# Scaler utilities
# ─────────────────────────────────────────────────────────────────────────────

def fit_quantile_scaler(X):
    """Fit QuantileTransformer on X, return (scaler, X_transformed)."""
    scaler = QuantileTransformer(output_distribution="normal")
    X_t = scaler.fit_transform(X.astype(np.float64)).astype(np.float32)
    return scaler, X_t
