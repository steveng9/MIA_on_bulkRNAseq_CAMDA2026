"""Alternative MIA pipeline: shadow models trained on synthetic data.

Instead of training shadows on real data subsets (which creates a domain gap
at inference, where the target proxy is trained on synthetic data), this variant
trains shadows directly on the 5 NoisyDiffusion synthetic datasets.

The MLP thus learns "what does member vs non-member look like through a
synthetic-data-trained model?" — exactly matching the inference setting.

Pipeline:
  1. Train 5 shadow models on ND synthetic data (one per split).
  2. Extract loss features for all real samples using each shadow.
     Scaler fitted on the same synthetic data used for training that shadow.
  3. Train MLP classifier on splits 1-3, validate on 4-5.
     Labels are ND ground truth (in/out of each split's training set).
  4. Per-split evaluation.
  5. Challenge inference (same as original — target proxy on challenge synthetic).
"""

import os
import numpy as np
import pandas as pd
import torch

from . import config
from .data_utils import (
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
from .classifier import train_classifier, load_classifier, MembershipMLP, _tpr_at_fpr


def _force(stage):
    """Return True if the given stage should be forced to re-run."""
    return "all" in config.FORCE_STAGES or stage in config.FORCE_STAGES


# ── Output directory helpers ────────────────────────────────────────────────
# Use separate dirs so synth-shadow artifacts don't clobber the original ones.

def _shadow_model_dir(dataset_name):
    return os.path.join(config.SYNTH_SHADOW_MODEL_DIR, dataset_name)


def _features_dir(dataset_name):
    return os.path.join(config.SYNTH_SHADOW_FEATURES_DIR, dataset_name)


# ── Step 1 ───────────────────────────────────────────────────────────────────

def step_train_shadows(dataset_name, splits=None, device=None):
    """Train shadow models on ND synthetic data (one per split).  Skips if checkpoint exists."""
    splits = splits or list(range(1, config.NUM_SPLITS + 1))
    device = device or config.DEVICE
    save_dir = _shadow_model_dir(dataset_name)
    force = _force("shadows")

    for s in splits:
        save_path = os.path.join(save_dir, f"shadow_split_{s}.pt")
        if not force and os.path.exists(save_path):
            print(f"  Synth-shadow split {s} exists: {save_path} (skipping)")
            continue
        print(f"\n{'='*60}")
        print(f"Training synth-shadow model: {dataset_name} split {s}")
        print(f"{'='*60}")

        X_syn, _ = load_nd_synthetic(dataset_name, s)
        train_shadow_model(X_syn, save_path, split_no=s, device=device)


# ── Step 2 ───────────────────────────────────────────────────────────────────

def step_extract_features(dataset_name, splits=None, device=None):
    """Extract loss features for all real samples using synth-shadow models.

    Scaler is fitted on the same ND synthetic data used for training each shadow.
    Labels come from ND splits YAML (ground truth for the ND generative process).
    """
    splits = splits or list(range(1, config.NUM_SPLITS + 1))
    device = device or config.DEVICE

    feat_dir = _features_dir(dataset_name)
    os.makedirs(feat_dir, exist_ok=True)

    model_dir = _shadow_model_dir(dataset_name)

    # Load real data once
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
        print(f"\n{'='*60}")
        print(f"Extracting features (synth-shadow): {dataset_name} split {s}")
        print(f"{'='*60}")

        # Load shadow model
        model = build_model(config.UNCONDITIONAL_NUM_CLASSES, device)
        ckpt = os.path.join(model_dir, f"shadow_split_{s}.pt")
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        diff_trainer = build_diffusion_trainer(device)

        # Fit scaler on the same ND synthetic data this shadow was trained on
        X_syn, _ = load_nd_synthetic(dataset_name, s)
        scaler, _ = fit_quantile_scaler(X_syn)

        # Scale all real data with this scaler
        X_scaled = scaler.transform(X_real_np.astype(np.float64)).astype(np.float32)

        # Extract features
        features = extract_loss_features(model, diff_trainer, X_scaled, y_int, device=device)

        # Ground-truth membership from ND splits
        _, y_member = get_nd_membership_labels(dataset_name, s)

        np.savez(out_path, features=features, y_member=y_member,
                 y_label_int=y_int, sample_ids=sample_ids)
        print(f"  Saved {out_path}  shape={features.shape}")


# ── Step 3 ───────────────────────────────────────────────────────────────────

def step_train_classifier(dataset_name, train_splits=None, val_splits=None, device=None):
    """Train MLP on synth-shadow features. Saves to SYNTH_SHADOW_CLASSIFIER_DIR."""
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

    # Train — but save to our own classifier dir
    model, history = train_classifier(X_train, y_train, X_val, y_val, device=device,
                                      save_dir=config.SYNTH_SHADOW_CLASSIFIER_DIR)
    return model, history


# ── Step 4: Challenge inference ──────────────────────────────────────────────

def step_predict_challenge(dataset_name, device=None):
    """Train target proxy on challenge synthetic data, classify with synth-shadow MLP."""
    device = device or config.DEVICE

    X_real_df, _ = load_real_data(dataset_name)
    X_real_np = X_real_df.values.astype(np.float32)
    sample_ids = list(X_real_df.index)

    # Train or load target proxy (shared with original pipeline — same model)
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

    # Load synth-shadow MLP
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
    """Load the synth-shadow MLP from its own classifier dir."""
    device = device or "cpu"
    model = MembershipMLP(dropout=config.MLP_DROPOUT)
    ckpt_path = os.path.join(config.SYNTH_SHADOW_CLASSIFIER_DIR, "mlp_best.pt")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model


# ── Full pipeline ────────────────────────────────────────────────────────────

def run_full_pipeline(dataset_name, device=None):
    """Run the synth-shadow MIA pipeline for one dataset."""
    device = device or config.DEVICE

    print("\n" + "=" * 70)
    print(f"SYNTH-SHADOW PIPELINE: {dataset_name}")
    print("=" * 70)

    # ── Step 1 ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"STEP 1: Training shadow models on ND synthetic data ({dataset_name})")
    print("=" * 70)
    step_train_shadows(dataset_name, device=device)

    # ── Step 2 ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"STEP 2: Extracting loss features via synth-shadows ({dataset_name})")
    print("=" * 70)
    step_extract_features(dataset_name, device=device)

    # ── Step 3 ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"STEP 3: Training MLP classifier ({dataset_name})")
    print("=" * 70)
    model, history = step_train_classifier(dataset_name, device=device)

    # ── Step 4: Per-split evaluation ─────────────────────────────────────
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

    # ── Step 5: Challenge predictions ────────────────────────────────────
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
    parser.add_argument("--profile", choices=list(config.PROFILES.keys()), default=None,
                        help="Named config profile (default: use current config values)")
    parser.add_argument("--force", default="",
                        help="Force re-run of stages (comma-separated: shadows,features,classifier,"
                             "synth_val,challenge) or 'all'")
    args = parser.parse_args()
    if args.force:
        config.FORCE_STAGES = set(s.strip() for s in args.force.split(","))
    if args.profile:
        config.apply_profile(args.profile)
    print(f"Active profile: {config.ACTIVE_PROFILE}  "
          f"(FEATURE_MODE={config.FEATURE_MODE}, MLP_INPUT_DIM={config.MLP_INPUT_DIM}, "
          f"MLP_DROPOUT={config.MLP_DROPOUT}, MLP_WEIGHT_DECAY={config.MLP_WEIGHT_DECAY})")
    run_full_pipeline(args.dataset, device=args.device)
