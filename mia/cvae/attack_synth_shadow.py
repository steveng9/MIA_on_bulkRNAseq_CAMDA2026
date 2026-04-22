"""CVAE synth-shadow MIA pipeline.

Shadow CVAEs are trained on CVAE-generated synthetic data (from the main attack
pipeline's shadows), instead of real data.  This eliminates the domain gap:
both training (synthetic) and inference (challenge synthetic) use a
synthetic-data-trained CVAE, matching the actual challenge setting.

Prerequisite: run mia.cvae.attack with --generate-synthetic first to produce
CVAE synthetic data in mia_output/cvae/synthetic_data/.

Pipeline:
  1. Train N CVAE synth-shadow models on CVAE synthetic data (one per split)
  2. Extract CVAE loss features for all real samples via each synth-shadow
  3. Train MLP membership classifier (70/30 split)
  4. Per-split evaluation
  5. Challenge predictions

Run: python -m mia.cvae.attack_synth_shadow [--dataset BRCA] [--profile baseline]
"""

import os
import numpy as np
import pandas as pd
import torch

from .. import config
from ..data_utils import (
    load_real_data,
    load_challenge_synthetic,
    get_membership_labels,
    fit_quantile_scaler,
    load_cvae_synthetic,
    save_cvae_synthetic,
    build_label_predictor,
)
from ..classifier import train_classifier, load_classifier_from_dir, _tpr_at_fpr
from .shadow_model import (
    train_cvae_shadow,
    load_cvae_target_proxy,
    build_cvae,
    _get_y_int,
)
from .loss_features import extract_cvae_features, prepare_cvae_features


def _force(stage):
    return "all" in config.FORCE_STAGES or stage in config.FORCE_STAGES


def _num_splits():
    # Synth-shadow uses the same split count as the main pipeline
    if config.SPLIT_MODE == "custom":
        return config.NUM_CUSTOM_SPLITS
    return config.NUM_SPLITS


def _default_train_val_splits():
    N = _num_splits()
    n_train = max(1, int(N * 0.7))
    return list(range(1, n_train + 1)), list(range(n_train + 1, N + 1))


# ── Step 1 ────────────────────────────────────────────────────────────────────

def step_train_shadows(dataset_name, splits=None, device=None):
    """Train CVAE synth-shadow models on CVAE-generated synthetic data."""
    splits = splits or list(range(1, _num_splits() + 1))
    device = device or config.DEVICE
    force = _force("shadows")

    if config.CVAE_LABEL_MODE != "none":
        predict_label = build_label_predictor(dataset_name, split_no=1)
    else:
        predict_label = None

    for s in splits:
        save_path = os.path.join(config.CVAE_SYNTH_SHADOW_MODEL_DIR, dataset_name,
                                 f"shadow_split_{s}.pt")
        if not force and os.path.exists(save_path):
            print(f"  CVAE synth-shadow split {s} exists: {save_path} (skipping)")
            continue

        print(f"\n{'='*60}\nTraining CVAE synth-shadow: {dataset_name} split {s}\n{'='*60}")
        X_syn, y_syn = load_cvae_synthetic(dataset_name, s)

        y_int = (predict_label(X_syn).astype(np.int64)
                 if predict_label is not None
                 else np.zeros(len(X_syn), dtype=np.int64))

        train_cvae_shadow(X_syn, y_int, dataset_name, save_path, split_no=s, device=device)


# ── Step 2 ────────────────────────────────────────────────────────────────────

def step_extract_features(dataset_name, splits=None, device=None):
    """Extract CVAE features via synth-shadow models.

    Scaler is fitted on the same CVAE synthetic data used to train each shadow.
    Labels come from the original custom splits (same membership ground truth).
    """
    splits = splits or list(range(1, _num_splits() + 1))
    device = device or config.DEVICE

    feat_dir = os.path.join(config.CVAE_SYNTH_SHADOW_FEATURES_DIR, dataset_name)
    os.makedirs(feat_dir, exist_ok=True)

    X_real_df, _ = load_real_data(dataset_name)
    X_real_np = X_real_df.values.astype(np.float32)
    sample_ids = list(X_real_df.index)
    y_int_all = _get_y_int(X_real_np, dataset_name)

    force = _force("features")
    for s in splits:
        out_path = os.path.join(feat_dir, f"features_split_{s}.npz")
        if not force and os.path.exists(out_path):
            print(f"  CVAE synth-shadow features split {s} exist (skipping)")
            continue

        print(f"\n{'='*60}\nExtracting CVAE synth-shadow features: {dataset_name} split {s}\n{'='*60}")

        ckpt = os.path.join(config.CVAE_SYNTH_SHADOW_MODEL_DIR, dataset_name,
                            f"shadow_split_{s}.pt")
        model = build_cvae(dataset_name, device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()

        # Scaler fitted on same CVAE synthetic data this shadow was trained on
        X_syn, _ = load_cvae_synthetic(dataset_name, s)
        scaler, _ = fit_quantile_scaler(X_syn)
        X_scaled = scaler.transform(X_real_np.astype(np.float64)).astype(np.float32)

        rec_losses, per_dim_kl = extract_cvae_features(
            model, X_scaled, y_int_all, device=device
        )
        _, y_member = get_membership_labels(dataset_name, s)

        np.savez(out_path,
                 rec_losses=rec_losses, per_dim_kl=per_dim_kl,
                 y_member=y_member, y_label_int=y_int_all, sample_ids=sample_ids)
        print(f"  Saved {out_path}  rec_losses={rec_losses.shape}  per_dim_kl={per_dim_kl.shape}")


# ── Step 3 ────────────────────────────────────────────────────────────────────

def step_train_classifier(dataset_name, train_splits=None, val_splits=None, device=None):
    if train_splits is None or val_splits is None:
        train_splits, val_splits = _default_train_val_splits()
    device = device or config.DEVICE
    feat_dir = os.path.join(config.CVAE_SYNTH_SHADOW_FEATURES_DIR, dataset_name)

    def _load(split_list):
        Xs, ys = [], []
        for s in split_list:
            d = np.load(os.path.join(feat_dir, f"features_split_{s}.npz"))
            Xs.append(prepare_cvae_features(d["rec_losses"], d["per_dim_kl"]))
            ys.append(d["y_member"])
        return np.concatenate(Xs), np.concatenate(ys)

    X_train, y_train = _load(train_splits)
    X_val, y_val = _load(val_splits)

    print(f"\nCVAE synth-shadow MLP train: {len(X_train)} samples (dim={X_train.shape[1]})  "
          f"members={y_train.sum()}  non-members={len(y_train)-y_train.sum()}")
    print(f"CVAE synth-shadow MLP val:   {len(X_val)} samples  "
          f"members={y_val.sum()}  non-members={len(y_val)-y_val.sum()}")

    model, history = train_classifier(
        X_train, y_train, X_val, y_val, device=device,
        save_dir=config.CVAE_SYNTH_SHADOW_CLASSIFIER_DIR,
        dropout=config.CVAE_MLP_DROPOUT,
        hidden_dim=config.CVAE_MLP_HIDDEN_DIM,
        weight_decay=config.CVAE_MLP_WEIGHT_DECAY,
        epochs=config.CVAE_MLP_EPOCHS,
        lr=config.CVAE_MLP_LR,
        batch_size=config.CVAE_MLP_BATCH_SIZE,
    )
    return model, history


# ── Step 5: Challenge predictions ─────────────────────────────────────────────

def step_predict_challenge(dataset_name, device=None):
    device = device or config.DEVICE

    X_real_df, _ = load_real_data(dataset_name)
    X_real_np = X_real_df.values.astype(np.float32)
    sample_ids = list(X_real_df.index)
    y_int_all = _get_y_int(X_real_np, dataset_name)

    proxy_ckpt = os.path.join(config.CVAE_SHADOW_MODEL_DIR, dataset_name, "target_proxy.pt")
    if not _force("challenge") and os.path.exists(proxy_ckpt):
        print("  Loading existing CVAE target proxy...")
        model = load_cvae_target_proxy(dataset_name, device=device)
        X_syn = load_challenge_synthetic(dataset_name)
        scaler, _ = fit_quantile_scaler(X_syn)
    else:
        print("  Training CVAE target proxy on challenge synthetic data...")
        from .shadow_model import train_cvae_shadow
        X_syn = load_challenge_synthetic(dataset_name)
        if config.CVAE_LABEL_MODE != "none":
            predict_label = build_label_predictor(dataset_name, split_no=1)
            y_syn_int = predict_label(X_syn).astype(np.int64)
        else:
            y_syn_int = np.zeros(len(X_syn), dtype=np.int64)
        model, scaler = train_cvae_shadow(
            X_syn, y_syn_int, dataset_name, proxy_ckpt, split_no=0, device=device
        )

    X_scaled = scaler.transform(X_real_np.astype(np.float64)).astype(np.float32)
    rec_losses, per_dim_kl = extract_cvae_features(
        model, X_scaled, y_int_all, device=device
    )
    features = prepare_cvae_features(rec_losses, per_dim_kl)

    clf = load_classifier_from_dir(
        save_dir=config.CVAE_SYNTH_SHADOW_CLASSIFIER_DIR,
        input_dim=features.shape[1],
        hidden_dim=config.CVAE_MLP_HIDDEN_DIM,
        dropout=config.CVAE_MLP_DROPOUT,
        device=device,
    )
    clf.eval()
    with torch.no_grad():
        scores = clf(torch.tensor(features, dtype=torch.float32)).squeeze(1).numpy()

    ds = config.DATASETS[dataset_name]
    out_path = os.path.join(ds["challenge_dir"],
                            "synthetic_data_1_predictions_cvae_synth_shadow.csv")
    pd.DataFrame({"sample_id": sample_ids, "score": scores}).to_csv(out_path, index=False)
    print(f"  Predictions saved → {out_path}")
    return sample_ids, scores


# ── Full pipeline ──────────────────────────────────────────────────────────────

def run_full_pipeline(dataset_name, device=None):
    device = device or config.DEVICE

    print("\n" + "=" * 70)
    print(f"CVAE SYNTH-SHADOW PIPELINE: {dataset_name}  label_mode={config.CVAE_LABEL_MODE}")
    print("=" * 70)

    print("\n" + "=" * 70)
    print(f"STEP 1: Training CVAE synth-shadow models ({dataset_name})")
    print("=" * 70)
    step_train_shadows(dataset_name, device=device)

    print("\n" + "=" * 70)
    print(f"STEP 2: Extracting CVAE loss features via synth-shadows ({dataset_name})")
    print("=" * 70)
    step_extract_features(dataset_name, device=device)

    print("\n" + "=" * 70)
    print(f"STEP 3: Training MLP classifier ({dataset_name})")
    print("=" * 70)
    train_splits, val_splits = _default_train_val_splits()
    model, history = step_train_classifier(
        dataset_name, train_splits=train_splits, val_splits=val_splits, device=device
    )

    print("\n" + "=" * 70)
    print(f"STEP 4: Per-split evaluation ({dataset_name})")
    print("=" * 70)
    feat_dir = os.path.join(config.CVAE_SYNTH_SHADOW_FEATURES_DIR, dataset_name)
    N = _num_splits()
    for s in range(1, N + 1):
        d = np.load(os.path.join(feat_dir, f"features_split_{s}.npz"))
        features = prepare_cvae_features(d["rec_losses"], d["per_dim_kl"])
        y_member = d["y_member"]
        model.eval()
        with torch.no_grad():
            scores = model(torch.tensor(features, dtype=torch.float32)).squeeze(1).numpy()
        tpr = _tpr_at_fpr(y_member, scores, fpr_target=0.10)
        acc = ((scores > 0.5).astype(int) == y_member).mean()
        print(f"  Split {s}: ACC={acc:.4f}  TPR@10%FPR={tpr:.4f}  "
              f"members={y_member.sum()}  non-members={len(y_member)-y_member.sum()}")

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
    parser.add_argument("--profile", choices=list(config.CVAE_PROFILES.keys()), default=None)
    parser.add_argument("--label-mode", choices=["knn", "none"], default=None)
    parser.add_argument("--force", default="",
                        help="Comma-separated: shadows,features,classifier,challenge or 'all'")
    args = parser.parse_args()

    if args.force:
        config.FORCE_STAGES = set(s.strip() for s in args.force.split(","))
    if args.profile:
        config.apply_cvae_profile(args.profile)
    if args.label_mode:
        config.CVAE_LABEL_MODE = args.label_mode

    print(f"[CVAE synth-shadow] profile={config.CVAE_ACTIVE_PROFILE}  "
          f"feature_mode={config.CVAE_FEATURE_MODE}  label_mode={config.CVAE_LABEL_MODE}")
    run_full_pipeline(args.dataset, device=args.device)
