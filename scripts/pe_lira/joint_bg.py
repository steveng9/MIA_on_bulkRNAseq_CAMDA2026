"""Joint attack with a background learned from the release (partial pool coverage).

joint_sparse.py spreads the votes of private records outside the target pool over cells in
proportion to Ref occupancy.  With a small Ref (a few records per cell) that estimate is very
noisy, and it is what limits the attack when the pool covers only part of D.  Here the
background is learned from the release itself, alternating

  1. s  <- posterior mean of the memberships given the current background (as joint_sparse)
  2. bg <- per round, each cell's leftover count (y - A s), clipped at 0, averaged over the
           k nearest OTHER cells of that round in the candidate embedding (leave-cell-out, so
           a target's own vote never enters its own background), then rescaled so the round's
           background totals n_hat - (expected pool votes)

starting from the Ref-occupancy background.  The pool's member rate s0 is the
cross-validated value from joint_sparse.  Inputs: released noised counts, candidates, sigma,
Ref -- as before.

    python scripts/pe_lira/joint_bg.py --worlds '~/privacy/nfree_worlds/artificial-characters_eps10_w*' [--hide 0.3]
"""
import argparse, glob, os, sys

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, cg
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from joint_sparse import joint_scores, load_world  # noqa: E402


def tpr(y, s, f):
    a, b, _ = roc_curve(y, s)
    return float(np.interp(f, a, b))


def solve(A, y, bg, s0, sigma):
    N = A.shape[1]
    tau2 = s0 * (1 - s0)
    r = y - s0 * np.asarray(A.sum(1)).ravel() - bg
    op = LinearOperator((N, N), matvec=lambda v: A.T @ (A @ v) / sigma**2 + v / tau2, dtype=float)
    d, _ = cg(op, A.T @ r / sigma**2, rtol=1e-6, maxiter=2000)
    return s0 + d


def learned_bg_scores(view, ctx, cls, sigma, s0_by, k=8, iters=4):
    out = np.zeros(len(cls))
    for c, rounds in view["rounds"].items():
        idx = np.flatnonzero(cls == c)
        if idx.size == 0 or str(c) not in s0_by:
            continue
        N, s0 = idx.size, s0_by[str(c)]
        blocks, ys, qs, nbrs, totals = [], [], [], [], []
        for i, rd in enumerate(rounds):
            y = rd.get("y")
            if y is None or np.isnan(y).all():
                continue
            ok = np.flatnonzero(~np.isnan(y))
            if len(ok) == len(y):
                totals.append(float(y.sum()))
            nn = ctx.nn[c][i][:, 0]
            Ai = sp.csr_matrix((np.ones(N), (nn, np.arange(N))), shape=(len(y), N))[ok]
            q = np.asarray(ctx.q[c][i])[ok]
            Xc = np.asarray(rd["X"])[ok]
            kk = min(k + 1, len(ok))
            nb = NearestNeighbors(n_neighbors=kk).fit(Xc).kneighbors(Xc, return_distance=False)[:, 1:]  # leave cell out
            blocks.append(Ai); ys.append(y[ok]); qs.append(q / max(q.sum(), 1e-12)); nbrs.append(nb)
        if not blocks:
            continue
        n_hat = float(np.mean(totals)) if totals else float(np.mean([v.sum() for v in ys]))
        A = sp.vstack(blocks).tocsr(); y = np.concatenate(ys)
        sizes = [len(v) for v in ys]; offs = np.cumsum([0] + sizes)
        bg = np.concatenate([max(n_hat - s0 * N, 0.0) * q for q in qs])           # Ref-occupancy start
        for _ in range(iters):
            s = solve(A, y, bg, s0, sigma)
            left = np.clip(y - A @ s, 0, None)
            new = []
            for t in range(len(ys)):
                lt = left[offs[t]:offs[t + 1]]
                sm = lt[nbrs[t]].mean(1) if nbrs[t].shape[1] else lt
                target = max(n_hat - float(np.clip(s, 0, 1).sum()), 0.0)
                new.append(sm * (target / max(sm.sum(), 1e-12)))
            bg = np.concatenate(new)
        out[idx] = solve(A, y, bg, s0, sigma)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", required=True)
    ap.add_argument("--hide", type=float, default=0.0, help="drop this share of member targets from the pool")
    ap.add_argument("--k", type=int, default=8)
    a = ap.parse_args()
    W = sorted(glob.glob(os.path.expanduser(a.worlds)))
    Y, G, L = [], [], []
    for n, w in enumerate(W):
        tr, view, ctx, cls, lab = load_world(w)
        if a.hide > 0:
            keep = np.ones(len(lab), bool)
            m = np.flatnonzero(lab == 1)
            keep[np.random.default_rng(n).choice(m, int(round(a.hide * len(m))), replace=False)] = False
            from attacks.common.scorers import Context
            ctx = Context(tr, ctx.X[keep], cls[keep], ctx.ref)
            cls, lab = cls[keep], lab[keep]
        sigma = float(tr["meta"]["sigma"])
        g, fits = joint_scores(view, ctx, cls, sigma)
        l = learned_bg_scores(view, ctx, cls, sigma, {f[0]: f[3] for f in fits}, k=a.k)
        Y.append(lab); G.append(g); L.append(l)
        print(f"{os.path.basename(w)}: Ref-background {roc_auc_score(lab, g):.3f}   learned background {roc_auc_score(lab, l):.3f}", flush=True)
    y = np.concatenate(Y)
    for name, S in (("Ref background (joint_sparse)", G), ("learned background", L)):
        s = np.concatenate([(v - v.mean()) / (v.std() + 1e-12) for v in S])
        print(f"pooled {name:30s} AUC {roc_auc_score(y, s):.3f}  TPR@1% {tpr(y, s, 0.01):.3f}  TPR@0.1% {tpr(y, s, 0.001):.3f}")


if __name__ == "__main__":
    main()
