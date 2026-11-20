"""Scalable joint attack on PE-Tabular (sparse design, conjugate gradients, closed-form prior).

Same model as joint_attack.py -- per class, released noised counts y_tj = sum_i A_tj,i s_i
+ bg_tj + N(0, sigma^2), A the nearest-cell assignment of the attacker's candidates --
for pools of tens of thousands of candidates:
  * A is sparse; the posterior mean solves (A'A/sigma^2 + I/tau^2) d = A'r/sigma^2 by CG.
  * The pool's member rate s0 (the empirical-Bayes grid of joint_attack.py does not scale)
    is chosen by 2-fold cross-validation over rounds: for each s0 on a grid, fit on the odd
    rounds and score the squared error of the predicted released counts of the even rounds,
    and vice versa.  Label-free.  (--prior moment: a closed form, E[y] = s0 a + (n - s0 N) q;
    biased when Ref occupancy misdescribes the out-of-pool records.)
Inputs as before: released noised counts, sigma, candidates, Ref.  World layout as the
testbed: <world>/{members,nonmembers,ref}.csv and <world>/run (a PE run with a histogram log).

    python scripts/pe_lira/joint_sparse.py --worlds '~/privacy/nfree_worlds/breast-cancer_eps10_w*'
"""
import argparse, glob, os, sys

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, cg
from sklearn.metrics import roc_auc_score, roc_curve

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "artifacts", "evo_mia_head"))


def tpr(y, s, f):
    a, b, _ = roc_curve(y, s)
    return float(np.interp(f, a, b))


def load_world(wdir):
    from attacks.common.scorers import Context
    from attacks.common.transcript import make_view
    from attacks.tabular.transcript_tabular import (info_from_candidates, load_transcript, make_embedder,
                                                    rows_from_csv, run_metadata)
    run = os.path.join(wdir, "run")
    md = run_metadata(run)
    embed = make_embedder(info_from_candidates(run, md), md)
    tr = load_transcript(run, embed)
    mem, mc = rows_from_csv(os.path.join(wdir, "members.csv"), md)
    non, nc = rows_from_csv(os.path.join(wdir, "nonmembers.csv"), md)
    ref, rc = rows_from_csv(os.path.join(wdir, "ref.csv"), md)
    X, cls = embed(mem + non), np.concatenate([mc, nc])
    lab = np.r_[np.ones(len(mem)), np.zeros(len(non))].astype(int)
    R = embed(ref)
    ctx = Context(tr, X, cls, {int(c): R[rc == c] for c in np.unique(rc) if c >= 0})
    view = make_view(tr, "counts")
    assert all(rd.get("clean") is None for rs in view["rounds"].values() for rd in rs)
    return tr, view, ctx, cls, lab


def _posterior(A, y, a, q, n_hat, N, s0, sigma):
    bg = max(n_hat - s0 * N, 0.0) * q
    tau2 = s0 * (1 - s0)
    r = y - s0 * a - bg
    op = LinearOperator((N, N), matvec=lambda v: A.T @ (A @ v) / sigma**2 + v / tau2, dtype=float)
    delta, info = cg(op, A.T @ r / sigma**2, rtol=1e-6, maxiter=2000)
    return s0 + delta, info


def joint_scores(view, ctx, cls, sigma, prior="cv"):
    s_out = np.zeros(len(cls))
    fits = []
    for c, rounds in view["rounds"].items():
        idx = np.flatnonzero(cls == c)
        if idx.size == 0:
            continue
        N = idx.size
        blocks, ys, qs, totals, rid = [], [], [], [], []
        for i, rd in enumerate(rounds):
            y = rd.get("y")
            if y is None or np.isnan(y).all():
                continue
            ok = ~np.isnan(y)
            if ok.all():
                totals.append(float(y.sum()))
            nn = ctx.nn[c][i][:, 0]
            P = len(y)
            Ai = sp.csr_matrix((np.ones(N), (nn, np.arange(N))), shape=(P, N))[np.flatnonzero(ok)]
            q = np.asarray(ctx.q[c][i])
            blocks.append(Ai); ys.append(y[ok]); qs.append((q / max(q.sum(), 1e-12))[ok])
            rid.append(np.full(int(ok.sum()), len(blocks) - 1))
        if not blocks:
            continue
        A, y, q = sp.vstack(blocks).tocsr(), np.concatenate(ys), np.concatenate(qs)
        rid = np.concatenate(rid)
        n_hat = float(np.mean(totals)) if totals else float(y.sum() / len(blocks))
        a = np.asarray(A.sum(1)).ravel()
        if prior == "moment":
            d, r0 = a - N * q, y - n_hat * q
            s0 = float(np.clip(d @ r0 / max(d @ d, 1e-12), 0.02, 0.98))
        elif prior == "cv":
            grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
            fold = rid % 2
            err = []
            for g in grid:
                e = 0.0
                for f in (0, 1):
                    tr_, te_ = fold != f, fold == f
                    sf, _ = _posterior(A[tr_], y[tr_], a[tr_], q[tr_], n_hat, N, g, sigma)
                    pred = A[te_] @ sf + max(n_hat - g * N, 0.0) * q[te_]
                    e += float(np.sum((y[te_] - pred) ** 2))
                err.append(e)
            s0 = float(grid[int(np.argmin(err))])
        else:
            s0 = float(np.clip(n_hat / N, 0.02, 0.98))
        s_fit, info = _posterior(A, y, a, q, n_hat, N, s0, sigma)
        s_out[idx] = s_fit
        fits.append((str(c), N, n_hat, s0, info))
    return s_out, fits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", required=True)
    ap.add_argument("--a4", action="store_true", help="also score evo-mia's A4 on the same targets")
    ap.add_argument("--prior", default="cv", choices=["cv", "moment", "ratio"])
    a = ap.parse_args()
    W = sorted(glob.glob(os.path.expanduser(a.worlds)))
    Y, J, B = [], [], []
    for w in W:
        tr, view, ctx, cls, lab = load_world(w)
        s, fits = joint_scores(view, ctx, cls, float(tr["meta"]["sigma"]), prior=a.prior)
        Y.append(lab); J.append(s)
        line = f"{os.path.basename(w)}: {len(lab)} targets ({lab.sum()} members); joint AUC {roc_auc_score(lab, s):.3f}"
        if a.a4:
            from attacks.common.scorers import score_variant
            b = score_variant(ctx, "A4")["score"]
            B.append(b); line += f"   A4 AUC {roc_auc_score(lab, b):.3f}"
        rate = np.mean([f[3] for f in fits])
        print(line + f"   fitted pool member rate {rate:.2f} (true {lab.mean():.2f}, evaluation only)", flush=True)
    y = np.concatenate(Y)
    z = lambda L: np.concatenate([(s - s.mean()) / s.std() for s in L])
    for name, L in (("joint", J), ("A4", B)):
        if L:
            s = z(L)
            print(f"pooled {name}: AUC {roc_auc_score(y, s):.3f}  TPR@1% {tpr(y, s, 0.01):.3f}  TPR@0.1% {tpr(y, s, 0.001):.3f}")


if __name__ == "__main__":
    main()
