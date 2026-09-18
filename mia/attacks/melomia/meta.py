"""Meta-classifier zoo, hyperparameter search and ensembling for MeLoMIA.

The meta-classifier maps a summarised loss vector to a membership probability.
Five model families are trained -- an MLP plus four gradient-boosted / bagged
tree ensembles -- and blended, because which family wins varies by cohort and
generator and picking one on the strength of the same validation split that
selected its hyperparameters would overfit that choice.

Two things make the validation here non-obvious, and getting either wrong
inflates the reported numbers badly:

*Grouping.*  Every real sample contributes one row per shadow model, each with
that shadow's membership label.  A random split would put rows for the same
sample on both sides, letting the classifier recognise the sample instead of
learning what membership looks like.  All splitting is therefore grouped by
sample id -- inside Optuna, for early stopping, and for the reported CV.

*Joint search.*  Optuna tunes the sweep-point subset and the draw budget
alongside each model's own hyperparameters, rather than fixing the feature set
first.  Which timesteps or temperatures carry signal depends on what the model
can exploit, so the two choices are not separable.

The objective is TPR at 10% FPR rather than AUC: a membership attack is
interesting when it identifies *some* members confidently, and AUC rewards
broad separation that need not translate into a usable high-confidence regime.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from torch.utils.data import DataLoader, TensorDataset

from ...metrics import tpr_at_fpr

SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# Grouped splitting helper
# ─────────────────────────────────────────────────────────────────────────────

def group_holdout(X, y, groups, test_size=0.15, seed=SEED):
    """Carve off a subject-disjoint slice for early stopping."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    fit_idx, es_idx = next(gss.split(X, y, groups=groups))
    return X[fit_idx], X[es_idx], y[fit_idx], y[es_idx]


def _pos_weight(y) -> float:
    n_pos = max(int(np.sum(y)), 1)
    return float((len(y) - n_pos) / n_pos)


# ─────────────────────────────────────────────────────────────────────────────
# MLP
# ─────────────────────────────────────────────────────────────────────────────

class MembershipMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 200, dropout: float = 0.0):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _train_mlp(X, y, groups, save_dir, hparams=None, device="cuda"):
    hp = hparams or {}
    epochs = int(hp.get("epochs", 400))
    lr = float(hp.get("lr", 1e-4))
    hidden = int(hp.get("hidden_dim", 128))
    dropout = float(hp.get("dropout", 0.1))
    wd = float(hp.get("weight_decay", 1e-4))
    bs = int(hp.get("batch_size", 1024))

    X_fit, X_es, y_fit, y_es = group_holdout(X, y, groups)
    model = MembershipMLP(X.shape[1], hidden, dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    crit = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([_pos_weight(y_fit)], dtype=torch.float32).to(device)
    )
    loader = DataLoader(
        TensorDataset(torch.tensor(X_fit, dtype=torch.float32),
                      torch.tensor(y_fit, dtype=torch.float32)),
        batch_size=bs, shuffle=True,
    )
    X_es_t = torch.tensor(X_es, dtype=torch.float32, device=device)

    best, best_state = -1.0, None
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            loss = crit(model(xb.to(device)).squeeze(1), yb.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            s = torch.sigmoid(model(X_es_t).squeeze(1)).cpu().numpy()
        metric = tpr_at_fpr(y_es.astype(int), s)
        if metric > best:
            best = metric
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    model.eval().cpu()
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "input_dim": X.shape[1],
                    "hidden_dim": hidden, "dropout": dropout},
                   Path(save_dir) / "mlp.pt")
    return model


def _predict_mlp(model, X):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(
            model(torch.tensor(np.asarray(X), dtype=torch.float32)).squeeze(1)
        ).numpy()


def _load_mlp(save_dir):
    ckpt = torch.load(Path(save_dir) / "mlp.pt", map_location="cpu")
    m = MembershipMLP(ckpt["input_dim"], ckpt["hidden_dim"], ckpt["dropout"])
    m.load_state_dict(ckpt["state_dict"])
    m.eval()
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Tree ensembles
# ─────────────────────────────────────────────────────────────────────────────

def _train_xgb(X, y, groups, save_dir, hparams=None, device="cuda"):
    from xgboost import XGBClassifier
    params = {"max_depth": 4, "learning_rate": 0.02, "subsample": 0.7,
              "colsample_bytree": 0.3, "min_child_weight": 20, "gamma": 1.0,
              "reg_alpha": 1.0, "reg_lambda": 5.0, "n_estimators": 2000,
              "tree_method": "hist", **(hparams or {})}
    n_estimators = params.pop("n_estimators")
    X_fit, X_es, y_fit, y_es = group_holdout(X, y, groups)
    clf = XGBClassifier(n_estimators=n_estimators, scale_pos_weight=_pos_weight(y_fit),
                        eval_metric="logloss", early_stopping_rounds=50,
                        verbosity=0, **params)
    clf.fit(X_fit, y_fit, eval_set=[(X_es, y_es)], verbose=False)
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, Path(save_dir) / "xgb.pkl")
    return clf


def _train_rf(X, y, groups, save_dir, hparams=None, device="cuda"):
    from sklearn.ensemble import RandomForestClassifier
    params = {"n_estimators": 500, "max_depth": 8, "max_features": "sqrt",
              "min_samples_leaf": 20, "max_samples": 0.7, **(hparams or {})}
    clf = RandomForestClassifier(
        **params, class_weight={0: 1.0, 1: _pos_weight(y)}, n_jobs=-1, random_state=SEED
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X, y)
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, Path(save_dir) / "rf.pkl")
    return clf


def _lgbm_cols(n):
    return [f"f{i}" for i in range(n)]


def _train_lgbm(X, y, groups, save_dir, hparams=None, device="cuda"):
    import lightgbm as lgb
    from lightgbm import LGBMClassifier
    params = {"num_leaves": 31, "learning_rate": 0.02, "subsample": 0.7,
              "subsample_freq": 1, "colsample_bytree": 0.3, "min_child_samples": 20,
              "reg_alpha": 1.0, "reg_lambda": 5.0, "n_estimators": 2000,
              "verbose": -1, "n_jobs": -1, "random_state": SEED, **(hparams or {})}
    X_fit, X_es, y_fit, y_es = group_holdout(X, y, groups)
    cols = _lgbm_cols(X.shape[1])
    clf = LGBMClassifier(scale_pos_weight=_pos_weight(y_fit), **params)
    clf.fit(pd.DataFrame(X_fit, columns=cols), y_fit,
            eval_set=[(pd.DataFrame(X_es, columns=cols), y_es)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, Path(save_dir) / "lgbm.pkl")
    return clf


def _predict_lgbm(clf, X):
    X = np.asarray(X)
    return clf.predict_proba(pd.DataFrame(X, columns=_lgbm_cols(X.shape[1])))[:, 1]


def _train_cat(X, y, groups, save_dir, hparams=None, device="cuda"):
    from catboost import CatBoostClassifier
    params = {"iterations": 1000, "depth": 6, "learning_rate": 0.02,
              "l2_leaf_reg": 5.0, "bootstrap_type": "Bernoulli", "subsample": 0.7,
              "early_stopping_rounds": 50, "eval_metric": "Logloss",
              "use_best_model": True, "thread_count": -1, "random_seed": SEED,
              "verbose": 0, **(hparams or {})}
    if params.get("bootstrap_type") == "Bayesian":
        params.pop("subsample", None)
    else:
        params.pop("bagging_temperature", None)
    X_fit, X_es, y_fit, y_es = group_holdout(X, y, groups)
    clf = CatBoostClassifier(class_weights=[1.0, _pos_weight(y_fit)], **params)
    clf.fit(X_fit, y_fit, eval_set=(X_es, y_es), verbose=False)
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        clf.save_model(str(Path(save_dir) / "cat.cbm"))
    return clf


def _load_cat(save_dir):
    from catboost import CatBoostClassifier
    clf = CatBoostClassifier()
    clf.load_model(str(Path(save_dir) / "cat.cbm"))
    return clf


def _proba(clf, X):
    return clf.predict_proba(np.asarray(X))[:, 1]


class _Entry:
    def __init__(self, train, predict, load):
        self.train, self.predict, self.load = train, predict, load


REGISTRY = {
    "mlp": _Entry(_train_mlp, _predict_mlp, _load_mlp),
    "xgb": _Entry(_train_xgb, _proba, lambda d: joblib.load(Path(d) / "xgb.pkl")),
    "rf": _Entry(_train_rf, _proba, lambda d: joblib.load(Path(d) / "rf.pkl")),
    "lgbm": _Entry(_train_lgbm, _predict_lgbm, lambda d: joblib.load(Path(d) / "lgbm.pkl")),
    "cat": _Entry(_train_cat, _proba, _load_cat),
}

ALL_CLASSIFIERS = ("xgb", "rf", "lgbm", "cat", "mlp")


def get(name: str) -> _Entry:
    if name not in REGISTRY:
        raise KeyError(f"Unknown classifier {name!r}. Known: {sorted(REGISTRY)}")
    return REGISTRY[name]


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameter search
# ─────────────────────────────────────────────────────────────────────────────

def sweep_buckets(n_sweep: int, n_buckets: int = 5) -> list:
    """Contiguous buckets of sweep indices for Optuna to switch on and off.

    Neighbouring timesteps (or temperatures) carry near-identical information,
    so searching each index independently wastes trials on distinctions that do
    not exist.  Buckets keep the search space small enough to explore in ~100
    trials while still letting the study discover that, say, only the early part
    of the diffusion trajectory matters.
    """
    n_buckets = max(1, min(n_buckets, n_sweep))
    edges = np.linspace(0, n_sweep, n_buckets + 1).astype(int)
    return [list(range(edges[i], edges[i + 1])) for i in range(n_buckets)
            if edges[i + 1] > edges[i]]


def _suggest(trial, clf_name):
    if clf_name == "xgb":
        return {
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 5e-3, 0.1, log=True),
            "subsample": trial.suggest_float("subsample", 0.4, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 0.5),
            "min_child_weight": trial.suggest_int("min_child_weight", 5, 50),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 10.0),
        }
    if clf_name == "rf":
        return {
            "n_estimators": trial.suggest_categorical("n_estimators", [200, 500, 1000]),
            "max_depth": trial.suggest_int("max_depth", 4, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.2, 0.3]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 50),
            "max_samples": trial.suggest_float("max_samples", 0.4, 0.9),
        }
    if clf_name == "lgbm":
        return {
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "learning_rate": trial.suggest_float("learning_rate", 5e-3, 0.05, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.8),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
        }
    if clf_name == "cat":
        bt = trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli"])
        p = {
            "bootstrap_type": bt,
            "iterations": trial.suggest_int("iterations", 200, 1200),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
        }
        p["bagging_temperature" if bt == "Bayesian" else "subsample"] = (
            trial.suggest_float("bagging_temperature", 0.0, 10.0) if bt == "Bayesian"
            else trial.suggest_float("subsample", 0.5, 1.0)
        )
        return p
    if clf_name == "mlp":
        return {
            "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 200]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
            "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
            "epochs": trial.suggest_categorical("epochs", [200, 400, 800]),
        }
    raise KeyError(clf_name)


def split_trial_params(best_params: dict, n_sweep: int, n_noise: int,
                       n_buckets: int = 5) -> tuple:
    """Unpack an Optuna result into (sweep_indices, noise_budget, clf hparams)."""
    buckets = sweep_buckets(n_sweep, n_buckets)
    chosen = []
    for i, b in enumerate(buckets):
        if best_params.get(f"use_bucket_{i}", False):
            chosen.extend(b)
    if len(chosen) < 2:                       # degenerate pick: fall back to everything
        chosen = list(range(n_sweep))
    noise_budget = int(best_params.get("noise_budget", n_noise))
    hp = {k: v for k, v in best_params.items()
          if not k.startswith("use_bucket_") and k != "noise_budget"}
    return sorted(set(chosen)), noise_budget, hp


def grouped_cv_score(clf_name, X, y, groups, hparams, n_folds=4, device="cuda") -> float:
    """Mean TPR@10%FPR over subject-disjoint folds."""
    entry = get(clf_name)
    n_folds = min(n_folds, max(2, len(np.unique(groups)) // 2))
    gkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    scores = []
    for tr, vl in gkf.split(X, y, groups=groups):
        try:
            clf = entry.train(X[tr], y[tr], groups[tr], None, hparams=hparams, device=device)
            scores.append(tpr_at_fpr(y[vl].astype(int), entry.predict(clf, X[vl])))
        except Exception as exc:  # a bad hyperparameter draw should not kill the study
            print(f"      [cv/{clf_name}] fold failed: {exc}", flush=True)
            scores.append(0.0)
    return float(np.mean(scores))


def evaluate_grouped(clf_name, X, y, groups, hparams, n_folds=4, device="cuda") -> dict:
    """Out-of-fold diagnostics for one configured classifier."""
    entry = get(clf_name)
    n_folds = min(n_folds, max(2, len(np.unique(groups)) // 2))
    gkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y), dtype=float)
    for tr, vl in gkf.split(X, y, groups=groups):
        clf = entry.train(X[tr], y[tr], groups[tr], None, hparams=hparams, device=device)
        oof[vl] = entry.predict(clf, X[vl])
    return {
        "auc": float(roc_auc_score(y, oof)),
        "aupr": float(average_precision_score(y, oof)),
        "tpr_at_fpr_0.1": tpr_at_fpr(y.astype(int), oof, 0.10),
        "tpr_at_fpr_0.01": tpr_at_fpr(y.astype(int), oof, 0.01),
    }


def softmax_weights(scores: dict, temperature: float = 0.05,
                    min_gate: float = 0.55) -> dict:
    """Blend weights from validation AUC, with a floor below which a model is cut.

    The low temperature makes this close to winner-take-all: models within a
    point or two of the best share the weight, clearly worse ones get almost
    none.  The gate removes models that failed outright, so they cannot drag the
    blend toward chance.  If every model is below the gate the blend falls back
    to equal weights -- a uniformly weak ensemble is still more honest than
    silently trusting whichever model got closest.
    """
    names = list(scores)
    vals = np.array([scores[n] for n in names], dtype=float)
    gate = (vals >= min_gate).astype(float)
    if gate.sum() == 0:
        warnings.warn(f"[melomia] all meta-classifiers below AUC gate {min_gate}; "
                      "falling back to equal weights")
        gate = np.ones(len(names))
    scaled = vals / temperature
    e = np.exp(scaled - scaled[gate > 0].max()) * gate
    return {n: float(w) for n, w in zip(names, e / e.sum())}


def save_meta(path: Path, meta: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(meta, indent=2, default=str))


def load_meta(path: Path) -> dict:
    return json.loads(Path(path).read_text())
