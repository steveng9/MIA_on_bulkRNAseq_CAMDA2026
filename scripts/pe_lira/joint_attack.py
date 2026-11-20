"""Joint membership attack on PE-Tabular: explain every released count with the candidates.

The per-record attacks (evo-mia A*/B*/C*) score each target alone and treat the other
votes in its cell as anonymous noise.  But each released count is a sum of votes of
records the attacker may hold as candidates.  For one class, over all rounds t and
cells j with a released (noised) count:

    y_tj = sum_i A_tj,i s_i + bg_tj + xi_tj,   xi ~ N(0, sigma^2)

A_tj,i = 1 if candidate i's nearest cell in round t is j (computed from the public
candidates and embedding, as the per-record attacks do), s_i in {0,1} is membership,
bg_tj the votes of private records outside the candidate pool.  Score = posterior mean of
s under a Gaussian prior s_i ~ N(s0, s0 (1 - s0)):

    s_hat = s0 + (A'A / sigma^2 + I / tau^2)^-1 A' (y - A s0 - bg) / sigma^2

Inputs: released noised counts (the 'counts' view), sigma, the candidates, Ref.  The class
size is estimated from the released totals (mean over rounds of sum_j y_tj); s0 =
min(n_hat / N, 1) and bg = max(n_hat - s0 N, 0) * Ref occupancy.  No clean counts, no true n.

--hide f removes a random fraction f of the member targets from the attacker's pool (they
still voted, so their votes are unexplained background), to test the "pool covers D"
assumption; metrics are then on the remaining targets.

    python scripts/pe_lira/joint_attack.py --worlds '~/privacy/nfree_worlds/breast-cancer_eps10_w*'
"""
import argparse, glob, os, sys

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "artifacts", "evo_mia_head"))


def tpr(y, s, f):
    a, b, _ = roc_curve(y, s)
    return float(np.interp(f, a, b))


def world_scores(wdir, hide=0.0, seed=0, placebo=False, prior="eb"):
    from attacks.common.scorers import Context, score_variant
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
    X = embed(mem + non)
    cls = np.concatenate([mc, nc])
    lab = np.r_[np.ones(len(mem)), np.zeros(len(non))].astype(int)
    keep = np.ones(len(lab), bool)
    if hide > 0:
        m = np.flatnonzero(lab == 1)
        keep[np.random.default_rng(seed).choice(m, int(round(hide * len(m))), replace=False)] = False
    R = embed(ref)
    ref_by_class = {int(c): R[rc == c] for c in np.unique(rc) if c >= 0}
    # The attacker only knows the kept candidates.
    ctx = Context(tr, X[keep], cls[keep], ref_by_class)
    view = make_view(tr, "counts")
    assert all(rd.get("clean") is None for rs in view["rounds"].values() for rd in rs)
    sigma = float(tr["meta"]["sigma"])
    s_out = np.zeros(keep.sum())
    fitted = []
    clsk = cls[keep]
    for c, rounds in view["rounds"].items():
        idx = np.flatnonzero(clsk == c)
        if idx.size == 0:
            continue
        rows_A, ys, bgw = [], [], []
        totals = []
        for i, rd in enumerate(rounds):
            y = rd.get("y")
            if y is None:
                continue
            ok = ~np.isnan(y)
            if not ok.any():
                continue
            if ok.all():
                totals.append(float(np.sum(y)))
            nn = ctx.nn[c][i][:, 0]                         # nearest cell of each kept target
            P = len(y)
            Ai = np.zeros((P, idx.size))
            Ai[nn, np.arange(idx.size)] = 1.0
            yy = y[ok]
            if placebo:                                   # control: counts shuffled across cells
                yy = np.random.default_rng(1000 * seed + i).permutation(yy)
            rows_A.append(Ai[ok]); ys.append(yy); bgw.append(np.asarray(ctx.q[c][i])[ok])
        if not rows_A:
            continue
        A, y, q = np.vstack(rows_A), np.concatenate(ys), np.concatenate(bgw)
        n_hat = float(np.mean(totals)) if totals else float(y.sum() / len(rows_A))
        N = idx.size
        # votes of private records outside the pool, spread over each round's cells by Ref occupancy
        qn = np.concatenate([qq / max(qq.sum(), 1e-12) for qq in bgw])
        G = A.T @ A
        lam = np.clip(np.linalg.eigvalsh(G), 0, None)
        a1 = A.sum(1)

        def setup(s0):
            bg = max(n_hat - s0 * N, 0.0) * qn
            return bg, s0 * (1 - s0), y - s0 * a1 - bg

        def marginal_ll(s0):
            """log N(y; A s0 + bg, sigma^2 I + tau^2 A A') via the N x N system (Woodbury)."""
            _, tau2, r = setup(s0)
            Ar = A.T @ r
            quad = (r @ r - tau2 * Ar @ np.linalg.solve(sigma**2 * np.eye(N) + tau2 * G, Ar)) / sigma**2
            logdet = (len(y) - N) * np.log(sigma**2) + np.sum(np.log(sigma**2 + tau2 * lam))
            return -0.5 * (quad + logdet)

        if prior == "eb":
            # empirical Bayes: the member rate of the pool that best explains the release
            grid = np.linspace(0.02, 0.98, 49)
            s0 = float(grid[np.argmax([marginal_ll(g) for g in grid])])
        else:
            s0 = min(max(n_hat / N, 1e-3), 1 - 1e-3)
        bg, tau2, r = setup(s0)
        M = G / sigma**2 + np.eye(N) / tau2
        s_out[idx] = s0 + np.linalg.solve(M, A.T @ r / sigma**2)
        fitted.append((int(c), N, n_hat, s0))
    a4 = score_variant(ctx, "A4")["score"]
    return lab[keep], s_out, a4, fitted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", required=True)
    ap.add_argument("--hide", type=float, default=0.0)
    ap.add_argument("--placebo", action="store_true", help="shuffle each round's counts across cells")
    ap.add_argument("--prior", default="eb", choices=["eb", "ratio"],
                    help="eb: member rate of the pool by marginal likelihood; ratio: n_hat / pool size")
    a = ap.parse_args()
    W = sorted(glob.glob(os.path.expanduser(a.worlds)), key=lambda p: int(p.rsplit("_w", 1)[1]))
    Y, J, B, per = [], [], [], []
    for k, w in enumerate(W):
        y, s, a4, fit = world_scores(w, a.hide, seed=k, placebo=a.placebo, prior=a.prior)
        Y.append(y); J.append(s); B.append(a4)
        per.append((roc_auc_score(y, s), roc_auc_score(y, a4)))
        s0m = np.mean([f[3] for f in fit]); true_rate = y.mean()
        print(f"{os.path.basename(w)}: joint AUC {per[-1][0]:.3f}   A4 AUC {per[-1][1]:.3f}   "
              f"fitted member rate {s0m:.2f} (true {true_rate:.2f}, evaluation only)", flush=True)
    y, j, b = np.concatenate(Y), np.concatenate(J), np.concatenate(B)
    # per-world standardisation before pooling, as the scales differ between worlds
    jz = np.concatenate([(s - s.mean()) / s.std() for s in J]); bz = np.concatenate([(s - s.mean()) / s.std() for s in B])
    for name, s in (("joint", jz), ("A4", bz)):
        print(f"pooled {name}: AUC {roc_auc_score(y, s):.3f}  TPR@1% {tpr(y, s, 0.01):.3f}  TPR@0.1% {tpr(y, s, 0.001):.3f}")
    p = np.array(per); print(f"mean within-world AUC: joint {p[:,0].mean():.3f} +- {p[:,0].std():.3f}, A4 {p[:,1].mean():.3f}")


if __name__ == "__main__":
    main()
