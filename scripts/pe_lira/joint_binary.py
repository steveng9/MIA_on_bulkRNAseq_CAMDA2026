"""Binary-membership versions of the joint attack, and a known-rest ceiling.

Same model and inputs as joint_sparse.py (per class: released noised counts
y = A s + bg + N(0, sigma^2), A the nearest-cell assignment of the candidates; s0 and bg
from joint_sparse's cross-validated fit).  joint_sparse relaxes s to a Gaussian; here s_i in
{0, 1}.  For target i, with R the residual counts y - A s - bg, the exact conditional is

    logit P(s_i = 1 | rest) = logit s0 + sum_t (2 R_t^(-i) - 1) / (2 sigma^2)

where R_t^(-i) is the residual of i's cell in round t with i's own vote removed.

  gauss      joint_sparse.py (Gaussian relaxation, CG)                      [baseline]
  meanfield  damped mean-field: p_i <- sigmoid(... with E[s] = p ...)      [attack]
  gibbs      Gibbs sampling, posterior marginals P(s_i = 1)                 [attack]
  known_rest the conditional above with every OTHER target's true membership
             plugged in: a diagnostic ceiling for joint inference (uses other
             targets' labels, never the clean counts) -- not an attack.

    python scripts/pe_lira/joint_binary.py --worlds '~/privacy/nfree_worlds/breast-cancer_eps10_w*'
"""
import argparse, glob, os, sys

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from joint_sparse import joint_scores, load_world  # noqa: E402


def tpr(y, s, f):
    a, b, _ = roc_curve(y, s)
    return float(np.interp(f, a, b))


def class_system(view, ctx, idx, c):
    """A (rows: released (round, cell); cols: targets idx), y, q, n_hat, and each target's row per round."""
    N = idx.size
    blocks, ys, qs, totals = [], [], [], []
    for i, rd in enumerate(view["rounds"][c]):
        y = rd.get("y")
        if y is None or np.isnan(y).all():
            continue
        ok = ~np.isnan(y)
        if ok.all():
            totals.append(float(y.sum()))
        nn = ctx.nn[c][i][:, 0]
        Ai = sp.csr_matrix((np.ones(N), (nn, np.arange(N))), shape=(len(y), N))[np.flatnonzero(ok)]
        q = np.asarray(ctx.q[c][i])
        blocks.append(Ai); ys.append(y[ok]); qs.append((q / max(q.sum(), 1e-12))[ok])
    A, y, q = sp.vstack(blocks).tocsr(), np.concatenate(ys), np.concatenate(qs)
    n_hat = float(np.mean(totals)) if totals else float(y.sum() / len(blocks))
    return A, y, q, n_hat


def conditional_logodds(A, R_plus_own, s0, sigma):
    """sum over each target's cells of (2 R^(-i) - 1) / (2 sigma^2), vectorised.
    R_plus_own: residual with the target's own (expected) vote added back, per row and target
    -> A.T @ R gives sum of residuals; the own-vote term is added by the caller."""
    return np.log(s0 / (1 - s0)) + R_plus_own / (2 * sigma**2)


def meanfield(A, y, q, n_hat, s0, sigma, iters=200, damp=0.3):
    N = A.shape[1]
    bg = max(n_hat - s0 * N, 0.0) * q
    T = np.asarray(A.sum(0)).ravel()                  # rounds each target appears in
    p = np.full(N, s0)
    for _ in range(iters):
        R = y - A @ p - bg                            # residual with expected memberships
        # residual excluding i's own expected vote, summed over i's cells: A'R + T p
        L = np.log(s0 / (1 - s0)) + (2 * (A.T @ R + T * p) - T) / (2 * sigma**2)
        p = (1 - damp) * p + damp / (1 + np.exp(-np.clip(L, -50, 50)))
    return p, L


def gibbs(A, y, q, n_hat, s0, sigma, sweeps=300, burn=100, seed=0):
    rng = np.random.default_rng(seed)
    A = A.tocsc()
    N = A.shape[1]
    bg = max(n_hat - s0 * N, 0.0) * q
    rows = [A.indices[A.indptr[i]:A.indptr[i + 1]] for i in range(N)]
    s = (rng.random(N) < s0).astype(float)
    R = y - A @ s - bg
    acc = np.zeros(N)
    lp = np.log(s0 / (1 - s0))
    for sw in range(sweeps):
        for i in rng.permutation(N):
            r = rows[i]
            Rm = R[r] + s[i]                          # residual with i's vote removed
            L = lp + np.sum(2 * Rm - 1) / (2 * sigma**2)
            new = float(rng.random() < 1 / (1 + np.exp(-L)))
            if new != s[i]:
                R[r] -= new - s[i]
                s[i] = new
        if sw >= burn:
            acc += s
    return acc / (sweeps - burn)


def known_rest(A, y, q, n_hat, s0, sigma, truth):
    N = A.shape[1]
    bg = max(n_hat - s0 * N, 0.0) * q
    T = np.asarray(A.sum(0)).ravel()
    R = y - A @ truth - bg
    return (2 * (A.T @ R + T * truth) - T) / (2 * sigma**2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", required=True)
    ap.add_argument("--no_gibbs", action="store_true")
    a = ap.parse_args()
    W = sorted(glob.glob(os.path.expanduser(a.worlds)))
    out = {k: [] for k in ("gauss", "meanfield", "gibbs", "known_rest")}
    Y = []
    for w in W:
        tr, view, ctx, cls, lab = load_world(w)
        sigma = float(tr["meta"]["sigma"])
        g, fits = joint_scores(view, ctx, cls, sigma)            # also gives the CV-fitted s0 per class
        s0_by = {f[0]: f[3] for f in fits}
        sc = {k: np.zeros(len(lab)) for k in out}
        sc["gauss"] = g
        for c in view["rounds"]:
            idx = np.flatnonzero(cls == c)
            if idx.size == 0 or str(c) not in s0_by:
                continue
            A, y, q, n_hat = class_system(view, ctx, idx, c)
            s0 = s0_by[str(c)]
            sc["meanfield"][idx], _ = meanfield(A, y, q, n_hat, s0, sigma)
            if not a.no_gibbs:
                sc["gibbs"][idx] = gibbs(A, y, q, n_hat, s0, sigma)
            sc["known_rest"][idx] = known_rest(A, y, q, n_hat, s0, sigma, lab[idx].astype(float))
        Y.append(lab)
        for k in out:
            out[k].append(sc[k])
        print(f"{os.path.basename(w)}: " + "  ".join(f"{k} {roc_auc_score(lab, sc[k]):.3f}" for k in out
                                                    if not (k == "gibbs" and a.no_gibbs)), flush=True)
    y = np.concatenate(Y)
    for k, L in out.items():
        if k == "gibbs" and a.no_gibbs:
            continue
        s = np.concatenate([(v - v.mean()) / (v.std() + 1e-12) for v in L])
        print(f"pooled {k:10s} AUC {roc_auc_score(y, s):.3f}  TPR@1% {tpr(y, s, 0.01):.3f}  TPR@0.1% {tpr(y, s, 0.001):.3f}"
              + ("   [diagnostic: uses other targets' labels]" if k == "known_rest" else ""))


if __name__ == "__main__":
    main()
