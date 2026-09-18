"""NoisyDiffusion MIA pipeline (clean namespace entry point).

Mirrors mia.attack but lives under mia.nd for architectural clarity.
Run: python -m mia.nd.attack [--dataset BRCA] [--profile tuned] [--force all]
"""

import os
import numpy as np
import pandas as pd
import torch

from .. import config
from ..data_utils import (
    load_real_data,
    load_real_split,
    load_challenge_synthetic,
    load_nd_synthetic,
    get_membership_labels,
    get_nd_membership_labels,
    generate_custom_splits,
    fit_quantile_scaler,
)
from .shadow_model import (
    train_shadow_model,
    train_target_proxy,
    load_shadow_model,
    load_target_proxy,
)
from .loss_features import extract_loss_features, prepare_features
from ..classifier import (
    train_classifier, train_rf_classifier,
    load_classifier_from_dir, load_rf_classifier,
    predict_scores, _tpr_at_fpr,
)


def _force(stage):
    return "all" in config.FORCE_STAGES or stage in config.FORCE_STAGES


def _num_shadow_splits():
    if config.SPLIT_MODE == "custom":
        return config.NUM_CUSTOM_SPLITS
    return config.NUM_SPLITS


def _default_train_val_splits():
    N = _num_shadow_splits()
    n_train = max(1, int(N * 0.7))
    train_splits = list(range(1, n_train + 1))
    val_splits = list(range(n_train + 1, N + 1))
    return train_splits, val_splits


def step_generate_splits(dataset_name):
    generate_custom_splits(dataset_name)


def step_train_shadows(dataset_name, splits=None, device=None):
    splits = splits or list(range(1, _num_shadow_splits() + 1))
    device = device or config.DEVICE
    force = _force("shadows")
    for s in splits:
        save_dir = os.path.join(config.SHADOW_MODEL_DIR, dataset_name)
        save_path = os.path.join(save_dir, f"shadow_split_{s}.pt")
        if not force and os.path.exists(save_path):
            print(f"  Shadow split {s} exists: {save_path} (skipping)")
            continue
        print(f"\n{'='*60}\nTraining shadow model: {dataset_name} split {s}\n{'='*60}")
        X_train = load_real_split(dataset_name, s)
        train_shadow_model(X_train, save_path, split_no=s, device=device)


def step_extract_features(dataset_name, splits=None, device=None):
    splits = splits or list(range(1, _num_shadow_splits() + 1))
    device = device or config.DEVICE
    os.makedirs(os.path.join(config.FEATURES_DIR, dataset_name), exist_ok=True)

    X_real_df, _ = load_real_data(dataset_name)
    X_real_np = X_real_df.values.astype(np.float32)
    sample_ids = list(X_real_df.index)
    y_int = np.full(len(X_real_np), config.DUMMY_LABEL, dtype=np.int64)

    force = _force("features")
    for s in splits:
        out_path = os.path.join(config.FEATURES_DIR, dataset_name, f"features_split_{s}.npz")
        if not force and os.path.exists(out_path):
            print(f"  Features split {s} exist: {out_path} (skipping)")
            continue
        print(f"\n{'='*60}\nExtracting features: {dataset_name} split {s}\n{'='*60}")

        model, diff_trainer = load_shadow_model(dataset_name, s, device=device)
        X_train_split = load_real_split(dataset_name, s)
        scaler, _ = fit_quantile_scaler(X_train_split)
        X_scaled = scaler.transform(X_real_np.astype(np.float64)).astype(np.float32)
        features = extract_loss_features(model, diff_trainer, X_scaled, y_int, device=device)
        _, y_member = get_membership_labels(dataset_name, s)

        np.savez(out_path, features=features, y_member=y_member,
                 y_label_int=y_int, sample_ids=sample_ids)
        print(f"  Saved {out_path}  shape={features.shape}")


def _nd_clf_dir():
    return os.path.join(config.CLASSIFIER_DIR, config.CLASSIFIER_TYPE)


def _load_nd_clf(clf_dir, input_dim):
    if config.CLASSIFIER_TYPE == "rf":
        return load_rf_classifier(clf_dir)
    return load_classifier_from_dir(
        clf_dir, input_dim,
        hidden_dim=config.MLP_HIDDEN_DIM,
        dropout=config.MLP_DROPOUT,
    )


def step_train_classifier(dataset_name, train_splits=None, val_splits=None, device=None):
    if train_splits is None or val_splits is None:
        default_train, default_val = _default_train_val_splits()
        train_splits = train_splits or default_train
        val_splits = val_splits or default_val
    device = device or config.DEVICE
    feat_dir = os.path.join(config.FEATURES_DIR, dataset_name)

    def _load(split_list):
        Xs, ys = [], []
        for s in split_list:
            d = np.load(os.path.join(feat_dir, f"features_split_{s}.npz"))
            Xs.append(prepare_features(d["features"]))
            ys.append(d["y_member"])
        return np.concatenate(Xs), np.concatenate(ys)

    X_train, y_train = _load(train_splits)
    X_val, y_val = _load(val_splits)

    clf_type = config.CLASSIFIER_TYPE
    print(f"\nND {clf_type.upper()} train: {len(X_train)} samples (dim={X_train.shape[1]}) "
          f"(members={y_train.sum()}, non-members={len(y_train)-y_train.sum()})")
    print(f"ND {clf_type.upper()} val:   {len(X_val)} samples "
          f"(members={y_val.sum()}, non-members={len(y_val)-y_val.sum()})")

    clf_dir = _nd_clf_dir()
    if clf_type == "rf":
        clf, history = train_rf_classifier(X_train, y_train, X_val, y_val, save_dir=clf_dir)
    else:
        clf, history = train_classifier(X_train, y_train, X_val, y_val, device=device,
                                        save_dir=clf_dir)
    return clf, history


def step_validate_synthetic(dataset_name, device=None):
    device = device or config.DEVICE
    X_real_df, _ = load_real_data(dataset_name)
    X_real_np = X_real_df.values.astype(np.float32)
    y_int = np.full(len(X_real_np), config.DUMMY_LABEL, dtype=np.int64)

    os.makedirs(os.path.join(config.SYNTH_VAL_MODEL_DIR, dataset_name), exist_ok=True)
    os.makedirs(os.path.join(config.SYNTH_VAL_FEATURES_DIR, dataset_name), exist_ok=True)

    force_sv = _force("synth_val")
    for s in range(1, config.NUM_SPLITS + 1):
        print(f"\n  --- Synthetic validation: split {s} ---")
        feat_path = os.path.join(
            config.SYNTH_VAL_FEATURES_DIR, dataset_name, f"features_synth_val_split_{s}.npz"
        )
        if not force_sv and os.path.exists(feat_path):
            print(f"    Features exist: {feat_path} (loading)")
            d = np.load(feat_path)
            features, y_member_nd = d["features"], d["y_member"]
        else:
            X_syn, _ = load_nd_synthetic(dataset_name, s)
            save_path = os.path.join(
                config.SYNTH_VAL_MODEL_DIR, dataset_name, f"synth_val_split_{s}.pt"
            )
            model, diff_trainer, scaler = train_shadow_model(
                X_syn, save_path, split_no=100 + s, device=device
            )
            X_scaled = scaler.transform(X_real_np.astype(np.float64)).astype(np.float32)
            features = extract_loss_features(model, diff_trainer, X_scaled, y_int, device=device)
            _, y_member_nd = get_nd_membership_labels(dataset_name, s)
            np.savez(feat_path, features=features, y_member=y_member_nd)

        features_prep = prepare_features(features)
        clf = _load_nd_clf(_nd_clf_dir(), features_prep.shape[1])
        scores = predict_scores(clf, features_prep)

        tpr = _tpr_at_fpr(y_member_nd, scores, fpr_target=0.10)
        acc = ((scores > 0.5).astype(int) == y_member_nd).mean()
        print(f"  Synth-val split {s}: ACC={acc:.4f}  TPR@10%FPR={tpr:.4f}  "
              f"(members={y_member_nd.sum()}, non-members={len(y_member_nd)-y_member_nd.sum()})")


def step_predict_challenge(dataset_name, device=None):
    device = device or config.DEVICE
    X_real_df, _ = load_real_data(dataset_name)
    X_real_np = X_real_df.values.astype(np.float32)
    sample_ids = list(X_real_df.index)

    proxy_ckpt = os.path.join(config.SHADOW_MODEL_DIR, dataset_name, "target_proxy.pt")
    if not _force("challenge") and os.path.exists(proxy_ckpt):
        print("  Loading existing target proxy...")
        model, diff_trainer = load_target_proxy(dataset_name, device=device)
        X_syn = load_challenge_synthetic(dataset_name)
        scaler, _ = fit_quantile_scaler(X_syn)
    else:
        print("  Training target proxy on challenge synthetic data...")
        model, diff_trainer, scaler = train_target_proxy(dataset_name, device=device)

    y_int = np.full(len(X_real_np), config.DUMMY_LABEL, dtype=np.int64)
    X_scaled = scaler.transform(X_real_np.astype(np.float64)).astype(np.float32)
    features = extract_loss_features(model, diff_trainer, X_scaled, y_int, device=device)
    features = prepare_features(features)

    clf = _load_nd_clf(_nd_clf_dir(), features.shape[1])
    scores = predict_scores(clf, features)

    ds = config.DATASETS[dataset_name]
    out_path = os.path.join(ds["challenge_dir"], "synthetic_data_1_predictions.csv")
    pd.DataFrame({"sample_id": sample_ids, "score": scores}).to_csv(out_path, index=False)
    print(f"  Predictions saved → {out_path}")
    return sample_ids, scores


def run_full_pipeline(dataset_name, device=None):
    device = device or config.DEVICE
    N = _num_shadow_splits()

    if config.SPLIT_MODE == "custom":
        print("\n" + "=" * 70)
        print(f"STEP 0: Generating {N} custom splits ({dataset_name})")
        print("=" * 70)
        step_generate_splits(dataset_name)

    print(f"\n  Split mode: {config.SPLIT_MODE}, {N} splits")

    print("\n" + "=" * 70)
    print(f"STEP 1: Training shadow models on real data subsets ({dataset_name})")
    print("=" * 70)
    step_train_shadows(dataset_name, device=device)

    print("\n" + "=" * 70)
    print(f"STEP 2: Extracting loss features ({dataset_name})")
    print("=" * 70)
    step_extract_features(dataset_name, device=device)

    print("\n" + "=" * 70)
    print(f"STEP 3: Training {config.CLASSIFIER_TYPE.upper()} classifier ({dataset_name})")
    print("=" * 70)
    train_splits, val_splits = _default_train_val_splits()
    print(f"  Train splits: {train_splits}, Val splits: {val_splits}")
    clf, history = step_train_classifier(
        dataset_name, train_splits=train_splits, val_splits=val_splits, device=device
    )

    print("\n" + "=" * 70)
    print(f"STEP 4: Per-split evaluation ({dataset_name})")
    print("=" * 70)
    feat_dir = os.path.join(config.FEATURES_DIR, dataset_name)
    for s in range(1, N + 1):
        d = np.load(os.path.join(feat_dir, f"features_split_{s}.npz"))
        features, y_member = prepare_features(d["features"]), d["y_member"]
        scores = predict_scores(clf, features)
        tpr = _tpr_at_fpr(y_member, scores, fpr_target=0.10)
        acc = ((scores > 0.5).astype(int) == y_member).mean()
        print(f"  Split {s}: ACC={acc:.4f}  TPR@10%FPR={tpr:.4f}  "
              f"(members={y_member.sum()}, non-members={len(y_member)-y_member.sum()})")

    print("\n" + "=" * 70)
    print(f"STEP 5: Synthetic validation — domain gap measurement ({dataset_name})")
    print("=" * 70)
    step_validate_synthetic(dataset_name, device=device)

    print("\n" + "=" * 70)
    print(f"STEP 6: Challenge predictions ({dataset_name})")
    print("=" * 70)
    step_predict_challenge(dataset_name, device=device)

    return clf, history


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["BRCA", "COMBINED"], default="BRCA")
    parser.add_argument("--device", default=config.DEVICE)
    parser.add_argument("--profile", choices=list(config.PROFILES.keys()), default=None)
    parser.add_argument("--classifier", choices=["mlp", "rf"], default="mlp",
                        help="Classifier type: 'mlp' (default) or 'rf' (Random Forest)")
    parser.add_argument("--force", default="",
                        help="Comma-separated stages to force: shadows,features,classifier,"
                             "synth_val,challenge  or 'all'")
    args = parser.parse_args()
    if args.force:
        config.FORCE_STAGES = set(s.strip() for s in args.force.split(","))
    if args.profile:
        config.apply_profile(args.profile)
    config.CLASSIFIER_TYPE = args.classifier
    print(f"[ND attack] profile={config.ACTIVE_PROFILE}  "
          f"FEATURE_MODE={config.FEATURE_MODE}  "
          f"classifier={config.CLASSIFIER_TYPE}  "
          f"MLP_INPUT_DIM={config.MLP_INPUT_DIM}")
    run_full_pipeline(args.dataset, device=args.device)
