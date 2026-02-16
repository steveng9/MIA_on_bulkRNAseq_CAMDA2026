"""End-to-end MIA pipeline orchestration.

Pipeline overview:
  1. Train shadow models on NoisyDiffusion repo's labeled synthetic data (5 splits).
  2. Extract loss-trajectory features for all real samples using each shadow model.
     - Real sample labels (subtypes) are inferred via KNN on labeled synthetic data.
     - Scaler is fitted on the same SMOTE-upsampled synthetic data used for training.
  3. Train MLP classifier on features from splits 1-3 (using ground-truth membership
     from the splits YAML), validate on splits 4-5.
  4. For the challenge: extract features using the shadow model and predict membership
     for all real samples.
"""

import os
import numpy as np
import pandas as pd
import torch

from . import config
from .data_utils import (
    load_real_data,
    load_nd_synthetic,
    load_splits_yaml,
    get_membership_labels,
    encode_labels,
    fit_quantile_scaler,
    infer_labels_knn,
)
from .shadow_model import train_shadow_model, load_shadow_model
from .loss_features import extract_loss_features
from .classifier import train_classifier, load_classifier, _tpr_at_fpr

from imblearn.over_sampling import SMOTE
from collections import Counter


def _reconstruct_scaler(dataset_name, split_no):
    """Reconstruct the QuantileTransformer fitted during shadow model training.

    Must replicate the exact same SMOTE + scaler pipeline.
    """
    ds = config.DATASETS[dataset_name]
    X_syn, y_syn_str = load_nd_synthetic(dataset_name, split_no)
    y_int = encode_labels(y_syn_str, ds["label_list"])

    sampling_strategy = {i: config.SMOTE_UPSAMPLE_TO for i in range(ds["num_classes"])}
    smote = SMOTE(sampling_strategy=sampling_strategy)
    X_syn, _ = smote.fit_resample(X_syn.astype(np.float64), y_int)

    scaler, _ = fit_quantile_scaler(X_syn)
    return scaler


def _get_real_labels(dataset_name, X_real_np, split_no=1):
    """Predict integer subtype labels for real samples using KNN on labeled synthetic data."""
    ds = config.DATASETS[dataset_name]
    X_syn, y_syn_str = load_nd_synthetic(dataset_name, split_no)
    y_syn_int = encode_labels(y_syn_str, ds["label_list"])
    return infer_labels_knn(X_real_np, X_syn, y_syn_int, k=5)


# ── Step 1 ───────────────────────────────────────────────────────────────────

def step_train_shadows(dataset_name, splits=None, device=None):
    """Train shadow models for the requested splits."""
    splits = splits or list(range(1, config.NUM_SPLITS + 1))
    device = device or config.DEVICE
    for s in splits:
        print(f"\n{'='*60}")
        print(f"Training shadow model: {dataset_name} split {s}")
        print(f"{'='*60}")
        train_shadow_model(dataset_name, s, device=device)


# ── Step 2 ───────────────────────────────────────────────────────────────────

def step_extract_features(dataset_name, splits=None, device=None):
    """Extract loss features for all real samples, using each split's shadow model.

    For each split, we:
      - Load the shadow model
      - Reconstruct the scaler (same SMOTE + QuantileTransformer)
      - Predict subtype labels for real samples via KNN
      - Scale real data with that scaler
      - Extract loss-trajectory features
      - Save alongside ground-truth membership labels from the splits YAML
    """
    splits = splits or list(range(1, config.NUM_SPLITS + 1))
    device = device or config.DEVICE
    os.makedirs(os.path.join(config.FEATURES_DIR, dataset_name), exist_ok=True)

    # Load real data once
    X_real_df, _ = load_real_data(dataset_name)
    X_real_np = X_real_df.values.astype(np.float32)
    sample_ids = list(X_real_df.index)

    for s in splits:
        print(f"\n{'='*60}")
        print(f"Extracting features: {dataset_name} split {s}")
        print(f"{'='*60}")

        model, diff_trainer = load_shadow_model(dataset_name, s, device=device)
        scaler = _reconstruct_scaler(dataset_name, s)

        # Predict subtype labels for real samples
        y_int = _get_real_labels(dataset_name, X_real_np, split_no=s)
        print(f"  Predicted label distribution: {Counter(y_int)}")

        # Scale real data
        X_scaled = scaler.transform(X_real_np.astype(np.float64)).astype(np.float32)

        # Extract features
        features = extract_loss_features(model, diff_trainer, X_scaled, y_int, device=device)

        # Ground-truth membership
        _, y_member = get_membership_labels(dataset_name, s)

        out_path = os.path.join(config.FEATURES_DIR, dataset_name, f"features_split_{s}.npz")
        np.savez(out_path, features=features, y_member=y_member,
                 y_label_int=y_int, sample_ids=sample_ids)
        print(f"  Saved {out_path}  shape={features.shape}")


# ── Step 3 ───────────────────────────────────────────────────────────────────

def step_train_classifier(dataset_name, train_splits=None, val_splits=None, device=None):
    """Train the MLP on extracted features from some splits, validate on others."""
    train_splits = train_splits or [1, 2, 3]
    val_splits = val_splits or [4, 5]
    device = device or config.DEVICE
    feat_dir = os.path.join(config.FEATURES_DIR, dataset_name)

    def _load(split_list):
        Xs, ys = [], []
        for s in split_list:
            d = np.load(os.path.join(feat_dir, f"features_split_{s}.npz"))
            Xs.append(d["features"])
            ys.append(d["y_member"])
        return np.concatenate(Xs), np.concatenate(ys)

    X_train, y_train = _load(train_splits)
    X_val, y_val = _load(val_splits)

    print(f"\nMLP train: {len(X_train)} samples "
          f"(members={y_train.sum()}, non-members={len(y_train)-y_train.sum()})")
    print(f"MLP val:   {len(X_val)} samples "
          f"(members={y_val.sum()}, non-members={len(y_val)-y_val.sum()})")

    model, history = train_classifier(X_train, y_train, X_val, y_val, device=device)
    return model, history


# ── Step 4: Challenge inference ──────────────────────────────────────────────

def step_predict_challenge(dataset_name, shadow_split=1, device=None):
    """Produce membership predictions for the challenge's real data.

    Uses a trained shadow model + trained MLP classifier.
    Outputs a CSV with sample_id and membership score.
    """
    device = device or config.DEVICE

    # Load real data
    X_real_df, _ = load_real_data(dataset_name)
    X_real_np = X_real_df.values.astype(np.float32)
    sample_ids = list(X_real_df.index)

    # Shadow model + scaler
    model, diff_trainer = load_shadow_model(dataset_name, shadow_split, device=device)
    scaler = _reconstruct_scaler(dataset_name, shadow_split)

    # Predict labels + scale
    y_int = _get_real_labels(dataset_name, X_real_np, split_no=shadow_split)
    X_scaled = scaler.transform(X_real_np.astype(np.float64)).astype(np.float32)

    # Extract features
    features = extract_loss_features(model, diff_trainer, X_scaled, y_int, device=device)

    # Classify
    clf = load_classifier()
    clf.eval()
    with torch.no_grad():
        scores = clf(torch.tensor(features, dtype=torch.float32)).squeeze(1).numpy()

    # Save predictions
    ds = config.DATASETS[dataset_name]
    out_path = os.path.join(ds["challenge_dir"], "synthetic_data_1_predictions.csv")
    pd.DataFrame({"sample_id": sample_ids, "score": scores}).to_csv(out_path, index=False)
    print(f"  Predictions saved → {out_path}")

    return sample_ids, scores


# ── Full pipeline ────────────────────────────────────────────────────────────

def run_full_pipeline(dataset_name, device=None):
    """Run the complete MIA pipeline for one dataset."""
    device = device or config.DEVICE

    print("\n" + "=" * 70)
    print(f"STEP 1: Training shadow models ({dataset_name})")
    print("=" * 70)
    step_train_shadows(dataset_name, device=device)

    print("\n" + "=" * 70)
    print(f"STEP 2: Extracting loss features ({dataset_name})")
    print("=" * 70)
    step_extract_features(dataset_name, device=device)

    print("\n" + "=" * 70)
    print(f"STEP 3: Training MLP classifier ({dataset_name})")
    print("=" * 70)
    model, history = step_train_classifier(dataset_name, device=device)

    # ── Per-split evaluation ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"STEP 4: Per-split evaluation ({dataset_name})")
    print("=" * 70)
    feat_dir = os.path.join(config.FEATURES_DIR, dataset_name)

    for s in range(1, config.NUM_SPLITS + 1):
        d = np.load(os.path.join(feat_dir, f"features_split_{s}.npz"))
        features, y_member = d["features"], d["y_member"]

        model.eval()
        with torch.no_grad():
            scores = model(torch.tensor(features, dtype=torch.float32)).squeeze(1).numpy()

        tpr = _tpr_at_fpr(y_member, scores, fpr_target=0.10)
        acc = ((scores > 0.5).astype(int) == y_member).mean()
        print(f"  Split {s}: ACC={acc:.4f}  TPR@10%FPR={tpr:.4f}  "
              f"(members={y_member.sum()}, non-members={len(y_member)-y_member.sum()})")

    # ── Challenge predictions ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"STEP 5: Challenge predictions ({dataset_name})")
    print("=" * 70)
    step_predict_challenge(dataset_name, device=device)

    return model, history


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["BRCA", "COMBINED"], default="BRCA")
    parser.add_argument("--device", default=config.DEVICE)
    args = parser.parse_args()
    run_full_pipeline(args.dataset, device=args.device)
