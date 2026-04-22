"""NoisyDiffusion synth-shadow MIA pipeline (clean namespace entry point).

Mirrors mia.attack_synth_shadow but lives under mia.nd.
Run: python -m mia.nd.attack_synth_shadow [--dataset BRCA] [--profile tuned]
"""

import os
import numpy as np
import pandas as pd
import torch

from .. import config
from ..data_utils import (
    load_real_data,
    load_nd_synthetic,
    load_challenge_synthetic,
    get_nd_membership_labels,
    fit_quantile_scaler,
)
from .shadow_model import (
    train_shadow_model,
    build_model,
    build_diffusion_trainer,
    train_target_proxy,
    load_target_proxy,
)
from .loss_features import extract_loss_features, prepare_features
from ..classifier import train_classifier, load_classifier, MembershipMLP, _tpr_at_fpr


def _force(stage):
    return "all" in config.FORCE_STAGES or stage in config.FORCE_STAGES


def _shadow_model_dir(dataset_name):
    return os.path.join(config.SYNTH_SHADOW_MODEL_DIR, dataset_name)


def _features_dir(dataset_name):
    return os.path.join(config.SYNTH_SHADOW_FEATURES_DIR, dataset_name)


def step_train_shadows(dataset_name, splits=None, device=None):
    splits = splits or list(range(1, config.NUM_SPLITS + 1))
    device = device or config.DEVICE
    save_dir = _shadow_model_dir(dataset_name)
    force = _force("shadows")

    for s in splits:
        save_path = os.path.join(save_dir, f"shadow_split_{s}.pt")
        if not force and os.path.exists(save_path):
            print(f"  Synth-shadow split {s} exists: {save_path} (skipping)")
            continue
        print(f"\n{'='*60}\nTraining synth-shadow model: {dataset_name} split {s}\n{'='*60}")
        X_syn, _ = load_nd_synthetic(dataset_name, s)
        train_shadow_model(X_syn, save_path, split_no=s, device=device)


def step_extract_features(dataset_name, splits=None, device=None):
    splits = splits or list(range(1, config.NUM_SPLITS + 1))
    device = device or config.DEVICE

    feat_dir = _features_dir(dataset_name)
    os.makedirs(feat_dir, exist_ok=True)
    model_dir = _shadow_model_dir(dataset_name)

    X_real_df, _ = load_real_data(dataset_name)
    X_real_np = X_real_df.values.astype(np.float32)
    sample_ids = list(X_real_df.index)
    y_int = np.full(len(X_real_np), config.DUMMY_LABEL, dtype=np.int64)

    force = _force("features")
    for s in splits:
        out_path = os.path.join(feat_dir, f"features_split_{s}.npz")
        if not force and os.path.exists(out_path):
            print(f"  Features split {s} exist: {out_path} (skipping)")
            continue
        print(f"\n{'='*60}\nExtracting features (synth-shadow): {dataset_name} split {s}\n{'='*60}")

        model = build_model(config.UNCONDITIONAL_NUM_CLASSES, device)
        ckpt = os.path.join(model_dir, f"shadow_split_{s}.pt")
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        diff_trainer = build_diffusion_trainer(device)

        X_syn, _ = load_nd_synthetic(dataset_name, s)
        scaler, _ = fit_quantile_scaler(X_syn)
        X_scaled = scaler.transform(X_real_np.astype(np.float64)).astype(np.float32)
        features = extract_loss_features(model, diff_trainer, X_scaled, y_int, device=device)
        _, y_member = get_nd_membership_labels(dataset_name, s)

        np.savez(out_path, features=features, y_member=y_member,
                 y_label_int=y_int, sample_ids=sample_ids)
        print(f"  Saved {out_path}  shape={features.shape}")


def step_train_classifier(dataset_name, train_splits=None, val_splits=None, device=None):
    train_splits = train_splits or [1, 2, 3]
    val_splits = val_splits or [4, 5]
    device = device or config.DEVICE
    feat_dir = _features_dir(dataset_name)

    def _load(split_list):
        Xs, ys = [], []
        for s in split_list:
            d = np.load(os.path.join(feat_dir, f"features_split_{s}.npz"))
            Xs.append(prepare_features(d["features"]))
            ys.append(d["y_member"])
        return np.concatenate(Xs), np.concatenate(ys)

    X_train, y_train = _load(train_splits)
    X_val, y_val = _load(val_splits)

    print(f"\nMLP train: {len(X_train)} samples (dim={X_train.shape[1]}) "
          f"(members={y_train.sum()}, non-members={len(y_train)-y_train.sum()})")
    print(f"MLP val:   {len(X_val)} samples "
          f"(members={y_val.sum()}, non-members={len(y_val)-y_val.sum()})")

    model, history = train_classifier(X_train, y_train, X_val, y_val, device=device,
                                      save_dir=config.SYNTH_SHADOW_CLASSIFIER_DIR)
    return model, history


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

    clf = _load_classifier()
    clf.eval()
    with torch.no_grad():
        scores = clf(torch.tensor(features, dtype=torch.float32)).squeeze(1).numpy()

    ds = config.DATASETS[dataset_name]
    out_path = os.path.join(ds["challenge_dir"], "synthetic_data_1_predictions_synth_shadow.csv")
    pd.DataFrame({"sample_id": sample_ids, "score": scores}).to_csv(out_path, index=False)
    print(f"  Predictions saved → {out_path}")
    return sample_ids, scores


def _load_classifier(device=None):
    device = device or "cpu"
    model = MembershipMLP(dropout=config.MLP_DROPOUT)
    ckpt_path = os.path.join(config.SYNTH_SHADOW_CLASSIFIER_DIR, "mlp_best.pt")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model


def run_full_pipeline(dataset_name, device=None):
    device = device or config.DEVICE

    print("\n" + "=" * 70)
    print(f"ND SYNTH-SHADOW PIPELINE: {dataset_name}")
    print("=" * 70)

    print("\n" + "=" * 70)
    print(f"STEP 1: Training shadow models on ND synthetic data ({dataset_name})")
    print("=" * 70)
    step_train_shadows(dataset_name, device=device)

    print("\n" + "=" * 70)
    print(f"STEP 2: Extracting loss features via synth-shadows ({dataset_name})")
    print("=" * 70)
    step_extract_features(dataset_name, device=device)

    print("\n" + "=" * 70)
    print(f"STEP 3: Training MLP classifier ({dataset_name})")
    print("=" * 70)
    model, history = step_train_classifier(dataset_name, device=device)

    print("\n" + "=" * 70)
    print(f"STEP 4: Per-split evaluation ({dataset_name})")
    print("=" * 70)
    feat_dir = _features_dir(dataset_name)
    for s in range(1, config.NUM_SPLITS + 1):
        d = np.load(os.path.join(feat_dir, f"features_split_{s}.npz"))
        features, y_member = prepare_features(d["features"]), d["y_member"]
        model.eval()
        with torch.no_grad():
            scores = model(torch.tensor(features, dtype=torch.float32)).squeeze(1).numpy()
        tpr = _tpr_at_fpr(y_member, scores, fpr_target=0.10)
        acc = ((scores > 0.5).astype(int) == y_member).mean()
        print(f"  Split {s}: ACC={acc:.4f}  TPR@10%FPR={tpr:.4f}  "
              f"(members={y_member.sum()}, non-members={len(y_member)-y_member.sum()})")

    print("\n" + "=" * 70)
    print(f"STEP 5: Challenge predictions ({dataset_name})")
    print("=" * 70)
    step_predict_challenge(dataset_name, device=device)

    return model, history


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["BRCA", "COMBINED"], default="BRCA")
    parser.add_argument("--device", default=config.DEVICE)
    parser.add_argument("--profile", choices=list(config.PROFILES.keys()), default=None)
    parser.add_argument("--force", default="",
                        help="Comma-separated stages: shadows,features,classifier,challenge or 'all'")
    args = parser.parse_args()
    if args.force:
        config.FORCE_STAGES = set(s.strip() for s in args.force.split(","))
    if args.profile:
        config.apply_profile(args.profile)
    print(f"[ND synth-shadow] profile={config.ACTIVE_PROFILE}  "
          f"FEATURE_MODE={config.FEATURE_MODE}  MLP_INPUT_DIM={config.MLP_INPUT_DIM}")
    run_full_pipeline(args.dataset, device=args.device)
