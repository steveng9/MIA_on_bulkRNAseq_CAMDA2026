"""Trial: per-record calibration across synth-shadows for MeLoMIA (notes/note_per_record_calibration.md).

Runs entirely from cached features (no model is trained):
    artifacts/attacks/melomia_{nd,cvae}/{dataset}/{stack}/features/shadow_k.npz
    artifacts/attacks/melomia_{nd,cvae}/{dataset}/{stack}/proxy_features/{generator}_split_s.npz

Feature variants, all from the same cached loss grids:
    raw        the current pipeline's features (reference-calibrated where a reference exists)
    model      log losses, each model standardised over the candidate records (control)
    record     log losses, per-record z-score across shadows (leave-one-out for training rows)
    both       model standardisation, then per-record z-score
    both_pool  as `both`, but one pooled spread per feature instead of a per-record spread
    draw       model-centred log losses, centred per (record, sweep point, frozen draw)
               across shadows, then summarised, then model-standardised

Two fixed classifiers (no Optuna), so the comparison isolates the features.  "held-out" is
the AUC on whole shadow models left out of training and out of the calibration reference;
the target AUCs use all K shadows.  Black box throughout: only candidates, synth-shadows
and the proxy are used, never the target's membership labels.

    python scripts/trial_per_record_calibration.py BRCA nd
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mia import datasets as D  # noqa: E402
from mia.attacks.melomia import features as F  # noqa: E402

NEW = "pgm@binning=dp_quantile,edge_estimator=threshold,n_bins=16"
GENS = {"MVN": "mvn", "CVAE": "cvae", "ND": "nd", "DP-PGM CAMDA-26": "pgg", "DP-PGM new": NEW}
STACK = {"nd": "n600", "cvae": "n50"}
K = {"BRCA": 30, "COMBINED": 20}
if os.environ.get("TRIAL_K"):                       # shadow-count check
    K = {k: int(os.environ["TRIAL_K"]) for k in K}
HOLD = min(5, max(1, K[sys.argv[1]] // 3))
EPS = 1e-6

ds, backend = sys.argv[1], sys.argv[2]
cache = ROOT / "artifacts" / "attacks" / f"melomia_{backend}" / ds / STACK[backend]


def load(path):
    d = np.load(path, allow_pickle=True)
    return (d["losses"], d["extra"] if "extra" in d else None,
            d["ref_losses"] if "ref_losses" in d else None)


def summ(losses, extra):
    return F.prepare(losses, extra, range(losses.shape[1]), losses.shape[2]).astype(np.float64)


def model_std(S):
    """Standardise each feature over the candidate records of one model."""
    return (S - S.mean(0)) / (S.std(0) + EPS)


def tpr_at(y, s, fpr=0.01):
    f, t, _ = roc_curve(y, s)
    return float(np.interp(fpr, f, t))


def classifiers():
    return {
        "lgbm": lambda: LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31,
                                       subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                       n_jobs=int(os.environ.get("MIA_N_JOBS", 24)), verbose=-1),
        "logreg": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)),
    }


# ── load everything once ────────────────────────────────────────────────────
Ks = list(range(1, K[ds] + 1))
raw, logS, logL, Y = [], [], [], []
for k in Ks:
    d = np.load(cache / "features" / f"shadow_{k}.npz", allow_pickle=True)
    L, E, R = load(cache / "features" / f"shadow_{k}.npz")
    Y.append(d["y_member"].astype(int))
    raw.append(summ(F.calibrate_against_reference(L, R) if R is not None else L, E))
    assert L.min() > 0, "log needs positive losses"
    lg = np.log(L)
    logS.append(summ(lg, E))
    logL.append((lg - lg.mean(axis=(0, 2), keepdims=True)).astype(np.float32))   # model-centred grid
    NX = 0 if E is None else E.shape[1]
raw, logS, Y = np.stack(raw), np.stack(logS), np.stack(Y)
logL = np.stack(logL)                                   # (K, n, sweep, draws)
print(f"{ds} {backend}: K={len(Ks)}, records={raw.shape[1]}, features={raw.shape[2]}, "
      f"member rate {Y.mean():.2f}", flush=True)

proxies = {}
for lab, g in GENS.items():
    for s in range(1, 6):
        p = cache / "proxy_features" / f"{g}_split_{s}.npz"
        if not p.exists():
            continue
        L, E, R = load(p)
        lg = np.log(L)
        proxies[(lab, s)] = dict(
            raw=summ(F.calibrate_against_reference(L, R) if R is not None else L, E),
            logS=summ(lg, E), E=E,
            logL=(lg - lg.mean(axis=(0, 2), keepdims=True)).astype(np.float32),
            y=D.membership_labels(ds, s).astype(int))


def z(x, ref, pooled=False):
    """x: (n, d); ref: (m, n, d) the same records under other models."""
    sd = ref.std(0)
    if pooled:
        sd = np.sqrt((sd ** 2).mean(0, keepdims=True))
    return (x - ref.mean(0)) / (sd + EPS)


def build(variant, idx, attack):
    """Training matrix for shadows `idx` (leave-one-out), and a function for attack rows."""
    idx = list(idx)
    if variant == "raw":
        return np.concatenate([raw[i] for i in idx]), lambda a: a["raw"]
    if variant == "model":
        return np.concatenate([model_std(logS[i]) for i in idx]), lambda a: model_std(a["logS"])
    if variant in ("record", "both", "both_pool"):
        pre = model_std if variant != "record" else (lambda s: s)
        P = np.stack([pre(logS[i]) for i in idx])
        tr = [z(P[j], np.delete(P, j, 0), variant == "both_pool") for j in range(len(idx))]
        return np.concatenate(tr), lambda a: z(pre(a["logS"]), P, variant == "both_pool")
    if variant == "draw":
        G = logL[idx]                                    # (m, n, sweep, draws)
        tot = G.sum(0)
        m = len(idx)
        Ex = np.stack([model_std(logS[i][:, -NX:]) for i in idx]) if NX else None   # extra block

        def ext(x, ref):
            return [z(x, ref)] if NX else []
        tr = [np.concatenate([model_std(summ(G[j] - (tot - G[j]) / (m - 1), None))]
                             + ext(Ex[j], np.delete(Ex, j, 0)) if NX else
                             [model_std(summ(G[j] - (tot - G[j]) / (m - 1), None))], axis=1)
              for j in range(m)]
        mean = tot / m
        return np.concatenate(tr), lambda a: np.concatenate(
            [model_std(summ(a["logL"] - mean, None))]
            + (ext(model_std(a["logS"][:, -NX:]), Ex) if NX else []), axis=1)
    raise KeyError(variant)


VARIANTS = (os.environ.get("TRIAL_VARIANTS", "raw,model,record,both,both_pool,draw")).split(",")
out = {}
for variant in VARIANTS:
    # held-out shadows: train on the first K-HOLD, score the last HOLD
    tr_idx, ho_idx = range(len(Ks) - HOLD), range(len(Ks) - HOLD, len(Ks))
    Xtr, fa = build(variant, tr_idx, None)
    ytr = np.concatenate([Y[i] for i in tr_idx])
    ho = [dict(raw=raw[i], logS=logS[i], logL=logL[i]) for i in ho_idx]
    Xfull, fa_full = build(variant, range(len(Ks)), None)
    yfull = Y.reshape(-1)
    for cname, make in classifiers().items():
        clf = make().fit(Xtr, ytr)
        ho_auc = float(np.mean([roc_auc_score(Y[i], clf.predict_proba(fa(h))[:, 1])
                                for i, h in zip(ho_idx, ho)]))
        clf = make().fit(Xfull, yfull)
        res = {"held_out": ho_auc}
        for lab in GENS:
            aucs, tprs = [], []
            for s in range(1, 6):
                if (lab, s) not in proxies:
                    continue
                a = proxies[(lab, s)]
                sc = clf.predict_proba(fa_full(a))[:, 1]
                aucs.append(roc_auc_score(a["y"], sc))
                tprs.append(tpr_at(a["y"], sc))
            if aucs:
                res[lab] = (float(np.mean(aucs)), float(np.mean(tprs)), len(aucs))
        out[f"{variant}/{cname}"] = res
        print(f"  {variant:9s} {cname:6s} held-out {ho_auc:.3f} | " +
              "  ".join(f"{lab} {res[lab][0]:.3f} (T@1% {res[lab][1]:.3f})" for lab in GENS if lab in res),
              flush=True)

dst = ROOT / "results" / "per_record_calibration"
dst.mkdir(parents=True, exist_ok=True)
json.dump(out, open(dst / f"trial_{ds}_{backend}{'_k' + os.environ['TRIAL_K'] if os.environ.get('TRIAL_K') else ''}.json", "w"), indent=1)
