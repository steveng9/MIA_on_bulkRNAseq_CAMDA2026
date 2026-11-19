"""Calibrate ICL-MIA with shadow releases (from shadows.py) and score the target.

Every release -- each shadow and the target -- is turned into the same
per-candidate statistic on the first G genes (`attack.scores`: zsum, typ),
using the shared pool term.  Then, per candidate x, shadows where x was in
context ("in") and out of it ("out") give reference distributions:

  raw        the uncalibrated statistic (baseline)
  lira       log N(s; mu_in(x), sd_in) - log N(s; mu_out(x), sd_out), variances
             pooled over candidates (Carlini et al. 2022, few-shadow regime)
  out_z      (s - mu_out(x)) / sd_out            (offline LiRA)
  weights    logistic regression on the per-gene terms of all shadow
             (candidate, shadow) pairs, membership known; applied to the target
  weights+lira  the learned score fed through lira; weights cross-fitted by
             shadow round so no shadow is scored by weights it trained

    python scripts/tabpfn_mia/calibrate.py [--genes 200] [--stat typ]
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attack import scores  # noqa: E402


def tpr_at(y, s, f):
    a, b, _ = roc_curve(y, s)
    return float(np.interp(f, a, b))


def features(l_rel, l_ref):
    """Per-gene terms as `scores` sees them: z-scored per gene, typicality removed."""
    D = l_rel - l_ref
    Z = (D - D.mean(0)) / (D.std(0) + 1e-12)
    T = np.column_stack([np.ones(len(Z)), l_ref.sum(1)])
    return Z - T @ np.linalg.lstsq(T, Z, rcond=None)[0]


def lira(s_target, S, M):
    """S: (n_shadows, n_cand) shadow statistics, M: same-shape in/out mask."""
    mu_in = np.nanmean(np.where(M, S, np.nan), 0)
    mu_out = np.nanmean(np.where(~M, S, np.nan), 0)
    sd_in = np.nanstd(np.where(M, S - mu_in, np.nan))
    sd_out = np.nanstd(np.where(~M, S - mu_out, np.nan))
    return (norm.logpdf(s_target, mu_in, sd_in) - norm.logpdf(s_target, mu_out, sd_out),
            (s_target - mu_out) / sd_out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="BRCA")
    ap.add_argument("--genes", type=int, default=200)
    ap.add_argument("--stat", default="typ", choices=["typ", "zsum"])
    ap.add_argument("--full", default="results/tabpfn_mia/full")
    ap.add_argument("--shadows", default="results/tabpfn_mia/shadows")
    args = ap.parse_args()
    G = args.genes

    ref = np.load(f"{args.full}/ref_{args.dataset}_n2_c1-978.npz")
    l_ref = ref["l_ref"][:, :G]
    files = sorted(glob.glob(f"{args.shadows}/shadow_{args.dataset}_g{G}_k*.npz"))
    sh = [np.load(f) for f in files]
    assert all((z["ids"] == ref["ids"]).all() for z in sh)
    rounds = np.array([int(f[-7:-4]) // 5 for f in files])
    M = np.stack([z["member"].astype(bool) for z in sh])
    S = np.stack([scores(z["l_rel"], l_ref)[args.stat] for z in sh])
    F_sh = [features(z["l_rel"], l_ref) for z in sh]
    print(f"{len(sh)} shadows ({len(set(rounds))} rounds); per candidate in/out "
          f"{M.sum(0).mean():.1f}/{(~M).sum(0).mean():.1f}; shadow-level {args.stat} AUC "
          f"{np.mean([roc_auc_score(m, s) for m, s in zip(M, S)]):.3f}")

    # Learned gene weights, cross-fitted by round for the shadows themselves.
    def fit_w(idx):
        Xf = np.concatenate([F_sh[i] for i in idx]); yf = np.concatenate([M[i] for i in idx])
        return LogisticRegression(C=0.1, max_iter=2000).fit(Xf, yf)
    W_all = fit_w(range(len(sh)))
    S_w = np.zeros_like(S)
    for r in set(rounds):
        clf = fit_w(np.flatnonzero(rounds != r))
        for i in np.flatnonzero(rounds == r):
            S_w[i] = clf.decision_function(F_sh[i])

    res = {}
    for split in range(1, 6):
        rel = np.load(f"{args.full}/rel_{args.dataset}_s{split}_n2_c1-978.npz")
        assert (rel["ids"] == ref["ids"]).all()
        m, lr = rel["member"], rel["l_rel"][:, :G]
        s_t = scores(lr, l_ref)[args.stat]
        lr_s, oz = lira(s_t, S, M)
        w_t = W_all.decision_function(features(lr, l_ref))
        lw, _ = lira(w_t, S_w, M)
        for name, v in [("raw", s_t), ("lira", lr_s), ("out_z", oz), ("weights", w_t),
                        ("weights+lira", lw)]:
            res.setdefault(name, []).append((roc_auc_score(m, v), tpr_at(m, v, 0.001),
                                             tpr_at(m, v, 0.01), tpr_at(m, v, 0.1)))
    print(f"target, first {G} genes, stat {args.stat}, mean over 5 splits")
    print(f"{'':14s} AUC    TPR@0.1%  TPR@1%  TPR@10%   AUC per split")
    for name, v in res.items():
        a = np.array(v)
        print(f"{name:14s} {a[:,0].mean():.3f}  {a[:,1].mean():.3f}     {a[:,2].mean():.3f}   "
              f"{a[:,3].mean():.3f}     {np.round(a[:,0], 3)}")


if __name__ == "__main__":
    main()
