"""CVAE-based MIA pipeline (real-data shadow variant).

Pipeline:
  0. (Re-use) custom splits from mia_output/splits/ — same as ND attack
  1. Train N CVAE shadow models on real data subsets (one per split)
  2. Extract CVAE loss features for all real samples via each shadow
  3. Train MLP membership classifier (70/30 train/val split of shadows)
  4. Per-split evaluation
  5. (Optional) Generate and save CVAE synthetic data from each shadow
     (needed for the synth-shadow ablation pipeline)
  6. Challenge predictions: CVAE proxy trained on challenge synthetic data

Run: python -m mia.cvae.attack [--dataset BRCA] [--profile baseline|tuned]
     [--label-mode knn|none] [--force all]
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
    get_membership_labels,
    generate_custom_splits,
    fit_quantile_scaler,
    save_cvae_synthetic,
    build_label_predictor,
    get_real_labels_all,
    get_real_labels_for_split,
)
from ..classifier import train_classifier, load_classifier_from_dir, _tpr_at_fpr
from .shadow_model import (
    train_cvae_shadow,
    load_cvae_shadow,
    load_cvae_target_proxy,
    build_cvae,
    generate_cvae_synthetic,
    _get_y_int,
)
from .loss_features import extract_cvae_features, prepare_cvae_features


def _force(stage):
    return "all" in config.FORCE_STAGES or stage in config.FORCE_STAGES


def _num_shadow_splits():
    if config.SPLIT_MODE == "custom":
        return config.NUM_CUSTOM_SPLITS
    return config.NUM_SPLITS


def _default_train_val_splits():
    N = _num_shadow_splits()
    n_train = max(1, int(N * 0.7))
    return list(range(1, n_train + 1)), list(range(n_train + 1, N + 1))


def _cvae_input_dim():
    n_t = len(config.CVAE_TEMP_LIST)
    if config.CVAE_FEATURE_MODE == "raw":
        return config.CVAE_N_SAMPLES * n_t + config.CVAE_Z_DIM
    return 4 * n_t + config.CVAE_Z_DIM


# ── Step 0 ────────────────────────────────────────────────────────────────────

def step_generate_splits(dataset_name):
    generate_custom_splits(dataset_name)


# ── Step 1 ────────────────────────────────────────────────────────────────────

def step_train_shadows(dataset_name, splits=None, device=None):
    """Train CVAE shadows on real data subsets.  Skips if checkpoint exists."""
    splits = splits or list(range(1, _num_shadow_splits() + 1))
    device = device or config.DEVICE
    force = _force("shadows")

    # Build KNN predictor once if needed (not used for "real" or "none" modes)
    predict_label = None
    if config.CVAE_LABEL_MODE == "knn":
        predict_label = build_label_predictor(dataset_name, split_no=1)

    for s in splits:
        save_dir = os.path.join(config.CVAE_SHADOW_MODEL_DIR, dataset_name)
        save_path = os.path.join(save_dir, f"shadow_split_{s}.pt")
        if not force and os.path.exists(save_path):
            print(f"  CVAE shadow split {s} exists: {save_path} (skipping)")
            continue

        print(f"\n{'='*60}\nTraining CVAE shadow: {dataset_name} split {s}\n{'='*60}")
        X_train = load_real_split(dataset_name, s)

        if config.CVAE_LABEL_MODE == "real":
            y_int = get_real_labels_for_split(dataset_name, s)
        elif predict_label is not None:
            y_int = predict_label(X_train).astype(np.int64)
        else:
            y_int = np.zeros(len(X_train), dtype=np.int64)

        train_cvae_shadow(X_train, y_int, dataset_name, save_path, split_no=s, device=device)


# ── Step 2 ────────────────────────────────────────────────────────────────────

def step_extract_features(dataset_name, splits=None, device=None):
    """Extract CVAE loss features for all real samples via each shadow."""
    splits = splits or list(range(1, _num_shadow_splits() + 1))
    device = device or config.DEVICE
    feat_out_dir = os.path.join(config.CVAE_FEATURES_DIR, dataset_name)
    os.makedirs(feat_out_dir, exist_ok=True)

    X_real_df, _ = load_real_data(dataset_name)
    X_real_np = X_real_df.values.astype(np.float32)
    sample_ids = list(X_real_df.index)

    # Resolve labels for all real samples (same for all splits)
    if config.CVAE_LABEL_MODE == "real":
        y_int_all = get_real_labels_all(dataset_name)
    else:
        y_int_all = _get_y_int(X_real_np, dataset_name)

    force = _force("features")
    for s in splits:
        out_path = os.path.join(feat_out_dir, f"features_split_{s}.npz")
        if not force and os.path.exists(out_path):
            print(f"  CVAE features split {s} exist: {out_path} (skipping)")
            continue

        print(f"\n{'='*60}\nExtracting CVAE features: {dataset_name} split {s}\n{'='*60}")

        model = load_cvae_shadow(dataset_name, s, device=device)

        # Scaler fitted on same real subset used for training this shadow
        X_train_split = load_real_split(dataset_name, s)
        scaler, _ = fit_quantile_scaler(X_train_split)
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

    def _load(split_list):
        Xs, ys = [], []
        for s in split_list:
            d = np.load(os.path.join(config.CVAE_FEATURES_DIR, dataset_name,
                                     f"features_split_{s}.npz"))
            Xs.append(prepare_cvae_features(d["rec_losses"], d["per_dim_kl"]))
            ys.append(d["y_member"])
        return np.concatenate(Xs), np.concatenate(ys)

    X_train, y_train = _load(train_splits)
    X_val, y_val = _load(val_splits)

    print(f"\nCVAE MLP train: {len(X_train)} samples (dim={X_train.shape[1]})  "
          f"members={y_train.sum()}  non-members={len(y_train)-y_train.sum()}")
    print(f"CVAE MLP val:   {len(X_val)} samples  "
          f"members={y_val.sum()}  non-members={len(y_val)-y_val.sum()}")

    model, history = train_classifier(
        X_train, y_train, X_val, y_val, device=device,
        save_dir=config.CVAE_CLASSIFIER_DIR,
        dropout=config.CVAE_MLP_DROPOUT,
        hidden_dim=config.CVAE_MLP_HIDDEN_DIM,
        weight_decay=config.CVAE_MLP_WEIGHT_DECAY,
        epochs=config.CVAE_MLP_EPOCHS,
        lr=config.CVAE_MLP_LR,
        batch_size=config.CVAE_MLP_BATCH_SIZE,
    )
    return model, history


# ── Step 5 (optional): Generate and save CVAE synthetic data ─────────────────

def step_generate_synthetic(dataset_name, splits=None, device=None):
    """Generate CVAE synthetic data from each trained shadow.

    Saved to CVAE_SYNTH_DIR for use by the synth-shadow pipeline.
    Skips splits where synthetic data already exists.
    """
    splits = splits or list(range(1, _num_shadow_splits() + 1))
    device = device or config.DEVICE
    force = _force("synth_gen")

    predict_label = None
    if config.CVAE_LABEL_MODE == "knn":
        predict_label = build_label_predictor(dataset_name, split_no=1)

    for s in splits:
        synth_path = os.path.join(config.CVAE_SYNTH_DIR, dataset_name,
                                  f"synthetic_data_split_{s}.npz")
        if not force and os.path.exists(synth_path):
            print(f"  CVAE synthetic split {s} exists (skipping): {synth_path}")
            continue

        print(f"\n{'='*60}\nGenerating CVAE synthetic: {dataset_name} split {s}\n{'='*60}")
        X_train = load_real_split(dataset_name, s)
        model = load_cvae_shadow(dataset_name, s, device=device)
        scaler, X_scaled = fit_quantile_scaler(X_train)

        if config.CVAE_LABEL_MODE == "real":
            y_int = get_real_labels_for_split(dataset_name, s)
        elif predict_label is not None:
            y_int = predict_label(X_train).astype(np.int64)
        else:
            y_int = np.zeros(len(X_train), dtype=np.int64)

        X_syn, y_syn = generate_cvae_synthetic(
            model, dataset_name, X_scaled, y_int,
            n_samples=len(X_train), device=device
        )
        # Inverse-transform back to original gene-expression space
        X_syn_raw = scaler.inverse_transform(X_syn.astype(np.float64)).astype(np.float32)
        save_cvae_synthetic(X_syn_raw, y_syn, dataset_name, s)
        print(f"  Saved CVAE synthetic split {s}: shape={X_syn_raw.shape}")


# ── Step 6: Challenge predictions ─────────────────────────────────────────────

def step_predict_challenge(dataset_name, device=None):
    """Train a CVAE proxy on challenge synthetic data and predict membership."""
    device = device or config.DEVICE

    X_real_df, _ = load_real_data(dataset_name)
    X_real_np = X_real_df.values.astype(np.float32)
    sample_ids = list(X_real_df.index)

    # Labels for real samples: use ground truth when available, else KNN/none
    if config.CVAE_LABEL_MODE == "real":
        y_int_all = get_real_labels_all(dataset_name)
    else:
        y_int_all = _get_y_int(X_real_np, dataset_name)

    proxy_ckpt = os.path.join(config.CVAE_SHADOW_MODEL_DIR, dataset_name, "target_proxy.pt")
    if not _force("challenge") and os.path.exists(proxy_ckpt):
        print("  Loading existing CVAE target proxy...")
        model = load_cvae_target_proxy(dataset_name, device=device)
        X_syn = load_challenge_synthetic(dataset_name)
        scaler, _ = fit_quantile_scaler(X_syn)
    else:
        print("  Training CVAE target proxy on challenge synthetic data...")
        X_syn = load_challenge_synthetic(dataset_name)
        # Challenge synthetic data has no real labels — always use KNN (or none)
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
        save_dir=config.CVAE_CLASSIFIER_DIR,
        input_dim=features.shape[1],
        hidden_dim=config.CVAE_MLP_HIDDEN_DIM,
        dropout=config.CVAE_MLP_DROPOUT,
        device=device,
    )
    clf.eval()
    with torch.no_grad():
        scores = clf(torch.tensor(features, dtype=torch.float32)).squeeze(1).numpy()

    ds = config.DATASETS[dataset_name]
    out_path = os.path.join(ds["challenge_dir"], "synthetic_data_1_predictions_cvae.csv")
    pd.DataFrame({"sample_id": sample_ids, "score": scores}).to_csv(out_path, index=False)
    print(f"  Predictions saved → {out_path}")
    return sample_ids, scores


# ── Full pipeline ──────────────────────────────────────────────────────────────

def run_full_pipeline(dataset_name, device=None, generate_synthetic=False):
    device = device or config.DEVICE
    N = _num_shadow_splits()

    if config.SPLIT_MODE == "custom":
        print("\n" + "=" * 70)
        print(f"STEP 0: Generating {N} custom splits ({dataset_name})")
        print("=" * 70)
        step_generate_splits(dataset_name)

    print(f"\n  Split mode: {config.SPLIT_MODE}, {N} splits  "
          f"label_mode={config.CVAE_LABEL_MODE}")

    print("\n" + "=" * 70)
    print(f"STEP 1: Training CVAE shadow models ({dataset_name})")
    print("=" * 70)
    step_train_shadows(dataset_name, device=device)

    print("\n" + "=" * 70)
    print(f"STEP 2: Extracting CVAE loss features ({dataset_name})")
    print("=" * 70)
    step_extract_features(dataset_name, device=device)

    print("\n" + "=" * 70)
    print(f"STEP 3: Training MLP classifier ({dataset_name})")
    print("=" * 70)
    train_splits, val_splits = _default_train_val_splits()
    print(f"  Train splits: {train_splits}, Val splits: {val_splits}")
    model, history = step_train_classifier(
        dataset_name, train_splits=train_splits, val_splits=val_splits, device=device
    )

    print("\n" + "=" * 70)
    print(f"STEP 4: Per-split evaluation ({dataset_name})")
    print("=" * 70)
    feat_dir = os.path.join(config.CVAE_FEATURES_DIR, dataset_name)
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

    if generate_synthetic:
        print("\n" + "=" * 70)
        print(f"STEP 5: Generating CVAE synthetic data ({dataset_name})")
        print("=" * 70)
        step_generate_synthetic(dataset_name, device=device)

    print("\n" + "=" * 70)
    print(f"STEP 6: Challenge predictions ({dataset_name})")
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
    parser.add_argument("--profile", choices=list(config.CVAE_PROFILES.keys()), default=None,
                        help="Named CVAE config profile")
    parser.add_argument("--label-mode", choices=["real", "knn", "none"], default=None,
                        help="Label source for real data: 'real'=BLUE zip ground truth, "
                             "'knn'=KNN on ND synthetic, 'none'=disable conditioning")
    parser.add_argument("--generate-synthetic", action="store_true",
                        help="Also generate and save CVAE synthetic data (needed for synth-shadow)")
    parser.add_argument("--force", default="",
                        help="Comma-separated: shadows,features,classifier,synth_gen,challenge or 'all'")
    args = parser.parse_args()

    if args.force:
        config.FORCE_STAGES = set(s.strip() for s in args.force.split(","))
    if args.profile:
        config.apply_cvae_profile(args.profile)
    if args.label_mode:
        config.CVAE_LABEL_MODE = args.label_mode

    print(f"[CVAE attack] profile={config.CVAE_ACTIVE_PROFILE}  "
          f"feature_mode={config.CVAE_FEATURE_MODE}  "
          f"label_mode={config.CVAE_LABEL_MODE}  "
          f"temp_list={config.CVAE_TEMP_LIST}  "
          f"n_samples={config.CVAE_N_SAMPLES}  "
          f"z_dim={config.CVAE_Z_DIM}")
    run_full_pipeline(args.dataset, device=args.device,
                      generate_synthetic=args.generate_synthetic)
