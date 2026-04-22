"""Shared MIA pipeline — model-agnostic orchestration.

All model-specific logic lives in the backend objects (mia.backends).
This module only handles:
  - Ensuring cached artefacts exist (splits, shadows, synthetic, features)
  - Training and evaluating the MLP
  - Generating challenge predictions

Pipeline stages and default flow
---------------------------------
  0. ensure_splits      — K+Q custom train/test splits
  1. ensure_real_shadows — K+Q shadows trained on real data subsets
  2. ensure_synthetic    — K+Q synthetic datasets generated from real shadows
  3. (synth mode) ensure_synth_shadows + ensure_features("synth") + MLP
     (real  mode) ensure_features("real") + MLP + domain-gap eval

Force flags:  "splits", "real_shadows", "synthetic", "synth_shadows",
              "features", "classifier", "submission", "all"
"""

import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from . import config
from .data_utils import (
    load_real_data,
    load_real_split,
    load_challenge_synthetic,
    get_membership_labels,
    generate_custom_splits,
    fit_quantile_scaler,
)
from .classifier import train_classifier, load_classifier_from_dir, _tpr_at_fpr


# ── Helpers ───────────────────────────────────────────────────────────────────

def _force(stage: str, force_stages: set) -> bool:
    return "all" in force_stages or stage in force_stages


def _hdr(msg: str):
    print(f"\n{'='*70}\n{msg}\n{'='*70}")


def _eval_scores(clf, features: np.ndarray, y_member: np.ndarray) -> dict:
    clf.eval()
    with torch.no_grad():
        scores = clf(torch.tensor(features, dtype=torch.float32)).squeeze(1).numpy()
    tpr   = _tpr_at_fpr(y_member, scores, fpr_target=0.10)
    auc   = float(roc_auc_score(y_member, scores)) if len(np.unique(y_member)) > 1 else float("nan")
    acc   = float((((scores > 0.5).astype(int)) == y_member).mean())
    return {"tpr10fpr": tpr, "auc": auc, "acc": acc, "scores": scores}


# ── Stage 0: Splits ───────────────────────────────────────────────────────────

def ensure_splits(dataset_name: str, n_total: int, force: bool = False):
    generate_custom_splits(dataset_name, n_splits=n_total, force=force)


# ── Stage 1: Real shadows ─────────────────────────────────────────────────────

def ensure_real_shadows(backend, split_nos: list, device: str, force: bool = False):
    """Train real-data shadow models for any missing split_nos."""
    for k in split_nos:
        path = backend.real_shadow_path(k)
        if not force and os.path.exists(path):
            continue
        _hdr(f"Real shadow — {backend.name.upper()} / {backend.dataset_name} / split {k}")
        X_train = load_real_split(backend.dataset_name, k)
        y_int   = backend.get_train_labels(k, X_train)
        backend.train_shadow(X_train, y_int, path, k, device)


# ── Stage 2: Synthetic datasets ───────────────────────────────────────────────

def ensure_synthetic(backend, split_nos: list, device: str, force: bool = False):
    """Generate synthetic datasets for any missing split_nos.

    Requires real shadows to already exist (ensure_real_shadows first).
    The scaler is fitted on the real training subset (same data the shadow saw).
    """
    for k in split_nos:
        synth_path = backend.synthetic_data_path(k)
        if not force and os.path.exists(synth_path):
            continue
        _hdr(f"Synthetic generation — {backend.name.upper()} / {backend.dataset_name} / split {k}")

        real_path = backend.real_shadow_path(k)
        if not os.path.exists(real_path):
            raise FileNotFoundError(
                f"Real shadow missing for split {k}: {real_path}\n"
                "Run ensure_real_shadows first."
            )
        model = backend.load_shadow(real_path, device)

        X_train = load_real_split(backend.dataset_name, k)
        y_int   = backend.get_train_labels(k, X_train)
        scaler, X_scaled = fit_quantile_scaler(X_train)

        X_syn, y_syn = backend.generate_synthetic(
            model, X_scaled, y_int, len(X_train), scaler, k, device
        )

        os.makedirs(os.path.dirname(synth_path), exist_ok=True)
        np.savez(synth_path, X=X_syn.astype(np.float32), y=y_syn.astype(np.int64))
        print(f"  Saved → {synth_path}  shape={X_syn.shape}")


# ── Stage 3: Synth-shadow models ──────────────────────────────────────────────

def ensure_synth_shadows(backend, split_nos: list, device: str, force: bool = False):
    """Train synth-shadow models on cached synthetic datasets."""
    for k in split_nos:
        path = backend.synth_shadow_path(k)
        if not force and os.path.exists(path):
            continue
        _hdr(f"Synth-shadow — {backend.name.upper()} / {backend.dataset_name} / split {k}")

        synth_path = backend.synthetic_data_path(k)
        if not os.path.exists(synth_path):
            raise FileNotFoundError(
                f"Synthetic data missing for split {k}: {synth_path}\n"
                "Run ensure_synthetic first."
            )
        d     = np.load(synth_path)
        X_syn = d["X"].astype(np.float32)
        y_syn = d["y"].astype(np.int64)

        backend.train_shadow(X_syn, y_syn, path, k, device)


# ── Stage 4: Feature extraction ───────────────────────────────────────────────

def ensure_features(backend, split_nos: list, source: str,
                    X_real_np: np.ndarray, y_int_all: np.ndarray,
                    sample_ids: list, device: str, force: bool = False) -> bool:
    """Extract and cache MIA features for all real samples via each shadow.

    source: 'real' → uses real shadows; 'synth' → uses synth shadows.
    The scaler is re-fitted on the shadow's training data to match the scale
    the model was trained on.

    Returns True if any feature files were (re-)generated, so callers can
    force MLP re-training when cached features become stale.
    """
    any_changed = False
    for k in split_nos:
        feat_path = backend.features_path(k, source)
        if not force and os.path.exists(feat_path):
            if backend.features_valid(feat_path):
                continue
            print(f"  [features] Stale features for split {k} "
                  f"(profile mismatch) — re-extracting.")
        _hdr(f"Feature extraction ({source}) — "
             f"{backend.name.upper()} / {backend.dataset_name} / split {k}")

        shadow_path = (backend.real_shadow_path(k) if source == "real"
                       else backend.synth_shadow_path(k))
        if not os.path.exists(shadow_path):
            raise FileNotFoundError(f"Shadow missing: {shadow_path}")

        model = backend.load_shadow(shadow_path, device)

        # Scaler fitted on whatever data this shadow was trained on
        if source == "real":
            X_train = load_real_split(backend.dataset_name, k)
        else:
            X_train = np.load(backend.synthetic_data_path(k))["X"].astype(np.float32)
        scaler, _ = fit_quantile_scaler(X_train)
        X_scaled  = scaler.transform(X_real_np.astype(np.float64)).astype(np.float32)

        raw_features = backend.extract_features(model, X_scaled, y_int_all, device)
        _, y_member  = get_membership_labels(backend.dataset_name, k)

        backend.save_raw_features(raw_features, feat_path, y_member, sample_ids)
        print(f"  Saved → {feat_path}")
        any_changed = True

    return any_changed


# ── Stage 5: MLP training ─────────────────────────────────────────────────────

def train_mlp(backend, train_splits: list, val_splits: list,
              source: str, device: str, force: bool = False):
    """Train membership MLP on features from train_splits, validate on val_splits.

    Saves best checkpoint to backend.classifier_dir(source).
    Returns (model, history).
    """
    save_dir  = backend.classifier_dir(source)
    ckpt_path = os.path.join(save_dir, "mlp_best.pt")

    if not force and os.path.exists(ckpt_path):
        print(f"  MLP checkpoint exists ({source}): {ckpt_path}  (skipping training)")
        input_dim = backend.prepare_features(
            backend.load_raw_features(backend.features_path(train_splits[0], source))[0]
        ).shape[1]
        clf = load_classifier_from_dir(
            save_dir=save_dir, input_dim=input_dim,
            hidden_dim=backend.mlp_kwargs["hidden_dim"],
            dropout=backend.mlp_kwargs["dropout"],
            device="cpu",
        )
        return clf, None

    def _load(splits):
        Xs, ys = [], []
        for k in splits:
            raw, y_member = backend.load_raw_features(backend.features_path(k, source))
            Xs.append(backend.prepare_features(raw))
            ys.append(y_member)
        return np.concatenate(Xs), np.concatenate(ys)

    X_train, y_train = _load(train_splits)
    X_val,   y_val   = _load(val_splits)

    print(f"\n  MLP ({source}): train={len(X_train)} samples (dim={X_train.shape[1]})  "
          f"members={y_train.sum()}  non-members={len(y_train)-y_train.sum()}")
    print(f"  MLP ({source}): val  ={len(X_val)} samples  "
          f"members={y_val.sum()}  non-members={len(y_val)-y_val.sum()}")

    model, history = train_classifier(
        X_train, y_train, X_val, y_val,
        device=device, save_dir=save_dir,
        **backend.mlp_kwargs,
    )
    return model, history


# ── Stage 6: Evaluation ───────────────────────────────────────────────────────

def evaluate_mlp(clf, backend, eval_splits: list, source: str) -> list:
    """Evaluate clf on features from eval_splits.  Returns list of per-split metric dicts."""
    results = []
    for k in eval_splits:
        raw, y_member = backend.load_raw_features(backend.features_path(k, source))
        features      = backend.prepare_features(raw)
        metrics       = _eval_scores(clf, features, y_member)
        metrics["split"] = k
        results.append(metrics)
    return results


def print_eval_results(results: list, label: str = ""):
    header = f"  {'EVAL':30s}  {'AUC':>6}  {'TPR@10%FPR':>10}  {'ACC':>6}  splits"
    if label:
        print(f"\n  ── {label} ──")
    print(header)
    for r in results:
        mem = int(r.get("members", 0))
        print(f"    split {r['split']:3d}:                    "
              f"AUC={r['auc']:.4f}  TPR@10%FPR={r['tpr10fpr']:.4f}  ACC={r['acc']:.4f}")
    avg_auc = float(np.mean([r["auc"] for r in results]))
    avg_tpr = float(np.mean([r["tpr10fpr"] for r in results]))
    avg_acc = float(np.mean([r["acc"] for r in results]))
    print(f"    {'AVERAGE':37s}  AUC={avg_auc:.4f}  TPR@10%FPR={avg_tpr:.4f}  ACC={avg_acc:.4f}")
    return avg_auc, avg_tpr


# ── Challenge submission ──────────────────────────────────────────────────────

def predict_challenge(backend, X_real_df, X_real_np: np.ndarray,
                      y_int_all: np.ndarray, clf, device: str,
                      force: bool = False):
    """Train a proxy on the challenge synthetic data and predict membership."""
    proxy_path = backend.target_proxy_path()

    if not force and os.path.exists(proxy_path):
        print(f"  Loading existing challenge proxy: {proxy_path}")
        model = backend.load_shadow(proxy_path, device)
        X_syn = load_challenge_synthetic(backend.dataset_name)
    else:
        print("  Training challenge proxy on synthetic data...")
        X_syn   = load_challenge_synthetic(backend.dataset_name)
        y_syn   = backend.get_submission_labels(X_syn)
        model, _ = backend.train_shadow(X_syn, y_syn, proxy_path, split_no=0, device=device)

    scaler, _  = fit_quantile_scaler(X_syn)
    X_scaled   = scaler.transform(X_real_np.astype(np.float64)).astype(np.float32)
    raw        = backend.extract_features(model, X_scaled, y_int_all, device)
    features   = backend.prepare_features(raw)

    clf.eval()
    with torch.no_grad():
        scores = clf(torch.tensor(features, dtype=torch.float32)).squeeze(1).numpy()

    sample_ids = list(X_real_df.index)
    ds         = config.DATASETS[backend.dataset_name]
    out_path   = os.path.join(ds["challenge_dir"],
                               f"predictions_{backend.name}.csv")
    pd.DataFrame({"sample_id": sample_ids, "score": scores}).to_csv(out_path, index=False)
    print(f"  Predictions saved → {out_path}")
    return scores


# ── Top-level pipeline ────────────────────────────────────────────────────────

def run_pipeline(backend, k: int, q: int, real_mode: bool,
                 device: str, no_submission: bool, force_stages: set):
    """Unified MIA pipeline.

    Parameters
    ----------
    backend       : MIABackend instance (NDBackend or CVAEBackend)
    k             : number of shadow models used for MLP training
    q             : number of holdout models used for internal evaluation
    real_mode     : if True, also run real-shadow evaluation + domain gap
    device        : torch device
    no_submission : skip challenge prediction step
    force_stages  : set of stage names to force re-run
    """
    dataset_name = backend.dataset_name
    n_total      = k + q
    all_splits   = list(range(1, n_total + 1))
    train_splits = list(range(1, k + 1))
    eval_splits  = list(range(k + 1, n_total + 1))

    # 70/30 split of the K training models for MLP internal train/val
    n_mlp_train     = max(1, int(k * 0.7))
    mlp_train_splits = train_splits[:n_mlp_train]
    mlp_val_splits   = train_splits[n_mlp_train:]

    force = force_stages  # shorthand

    print(f"\n{'='*70}")
    print(f"MIA PIPELINE  model={backend.name}  dataset={dataset_name}  "
          f"K={k}  Q={q}  mode={'real+synth' if real_mode else 'synth'}")
    print(f"  MLP train on: splits {mlp_train_splits[0]}..{mlp_train_splits[-1]} "
          f"({len(mlp_train_splits)} shadows)")
    print(f"  MLP val on:   splits {mlp_val_splits[0]}..{mlp_val_splits[-1]} "
          f"({len(mlp_val_splits)} shadows)")
    print(f"  Eval (holdout): splits {eval_splits[0]}..{eval_splits[-1]} "
          f"({len(eval_splits)} proxy models)")
    print(f"{'='*70}")

    # ── 0. Splits ─────────────────────────────────────────────────────────────
    _hdr(f"STEP 0 — Ensuring {n_total} custom splits")
    ensure_splits(dataset_name, n_total, force=_force("splits", force))

    X_real_df, _ = load_real_data(dataset_name)
    X_real_np    = X_real_df.values.astype(np.float32)
    sample_ids   = list(X_real_df.index)
    y_int_all    = backend.get_all_labels(X_real_np)

    # ── 1. Real shadows (always — needed for synthetic generation) ────────────
    _hdr(f"STEP 1 — Ensuring {n_total} real shadows")
    ensure_real_shadows(backend, all_splits, device,
                        force=_force("real_shadows", force) or _force("shadows", force))

    # ── 2. Synthetic datasets (always — needed for synth shadows + domain gap) ─
    _hdr(f"STEP 2 — Ensuring {n_total} synthetic datasets")
    ensure_synthetic(backend, all_splits, device, force=_force("synthetic", force))

    clf_synth = clf_real = None  # filled in below

    # ── Synth-shadow branch (default) ─────────────────────────────────────────
    _hdr(f"STEP 3 — Ensuring {n_total} synth-shadow models")
    ensure_synth_shadows(backend, all_splits, device,
                         force=_force("synth_shadows", force) or _force("shadows", force))

    _hdr(f"STEP 4 — Extracting synth-shadow features ({n_total} splits)")
    synth_feats_changed = ensure_features(backend, all_splits, "synth", X_real_np, y_int_all,
                                          sample_ids, device, force=_force("features", force))

    _hdr(f"STEP 5 — Training MLP on synth features  "
         f"(train={len(mlp_train_splits)} val={len(mlp_val_splits)})")
    clf_synth, _ = train_mlp(backend, mlp_train_splits, mlp_val_splits,
                              "synth", device,
                              force=_force("classifier", force) or synth_feats_changed)

    _hdr(f"STEP 6 — Internal evaluation on {q} synth-proxy holdout models")
    synth_eval = evaluate_mlp(clf_synth, backend, eval_splits, "synth")
    print_eval_results(synth_eval, label="SYNTH-PROXY EVAL (main result)")

    # ── Real-shadow branch (--real mode adds optimistic upper bound + gap) ────
    if real_mode:
        _hdr(f"STEP 7 — Extracting real-shadow features ({n_total} splits)")
        real_feats_changed = ensure_features(backend, all_splits, "real", X_real_np, y_int_all,
                                             sample_ids, device, force=_force("features", force))

        _hdr(f"STEP 8 — Training MLP on real features  "
             f"(train={len(mlp_train_splits)} val={len(mlp_val_splits)})")
        clf_real, _ = train_mlp(backend, mlp_train_splits, mlp_val_splits,
                                 "real", device,
                                 force=_force("classifier", force) or real_feats_changed)

        _hdr(f"STEP 9 — Upper-bound eval on {q} real-shadow holdout models")
        real_eval = evaluate_mlp(clf_real, backend, eval_splits, "real")
        avg_real_auc, avg_real_tpr = print_eval_results(
            real_eval, label="REAL-SHADOW EVAL (upper bound)")

        _hdr(f"STEP 10 — Domain-gap eval: same MLP on {q} synth-proxy holdout models")
        synth_gap_eval = evaluate_mlp(clf_real, backend, eval_splits, "synth")
        avg_syn_auc, avg_syn_tpr = print_eval_results(
            synth_gap_eval, label="SYNTH-PROXY EVAL (domain gap)")

        print(f"\n  Domain gap  TPR@10%FPR: "
              f"{avg_real_tpr:.4f} (real) − {avg_syn_tpr:.4f} (synth) "
              f"= {avg_real_tpr - avg_syn_tpr:.4f}")
        print(f"  Domain gap  AUC:        "
              f"{avg_real_auc:.4f} (real) − {avg_syn_auc:.4f} (synth) "
              f"= {avg_real_auc - avg_syn_auc:.4f}")

    # ── Challenge submission ──────────────────────────────────────────────────
    if not no_submission:
        _hdr("STEP FINAL — Challenge predictions")
        # Use synth-trained classifier for submission (matches challenge distribution)
        clf_for_submission = clf_synth if clf_synth is not None else clf_real
        predict_challenge(backend, X_real_df, X_real_np, y_int_all,
                          clf_for_submission, device,
                          force=_force("submission", force))
