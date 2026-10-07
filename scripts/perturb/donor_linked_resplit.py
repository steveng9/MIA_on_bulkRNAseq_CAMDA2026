"""Donor-linked membership inference on many redrawn splits (cheap generators).

The five canonical splits make each donor a non-member once.  Here the target
is retrained on R fresh splits in which the donors that have a second tumour
sample are placed half in, half out, and every other donor is a member with
probability 0.8.  That measures each second sample many times in both states,
which buys two things:

  * a linked AUC per repetition on balanced groups (about 35 donors a side),
    instead of one pooled number;
  * a per-record calibrated attack.  A second sample's score is z-scored
    against the same sample's scores in the other repetitions where its donor
    was OUT.  Those repetitions play the part of shadow releases that are known
    not to contain the donor (offline LiRA / difficulty calibration with ideal
    shadows), so this is what an adversary with good shadow modelling could
    reach, not what MahalaMIA does out of the box.

Attack: MahalaMIA with the ridge, d_aux / (d_syn + d_aux), uncalibrated scores.
Writes results/perturb/donor_linked_resplit_<dataset>_<generator>.csv (one row
per sample group) and an npz of all scores under artifacts/perturb/.

    python scripts/perturb/donor_linked_resplit.py --dataset COMBINED --generator mvn --reps 60
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mia import datasets as D  # noqa: E402
from mia import generators as G  # noqa: E402
from mia import paths  # noqa: E402
from mia import targets as T  # noqa: E402
from mia.attacks.mahalamia import mahalanobis, precision  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from donor_linked import extras  # noqa: E402

RIDGE = 1e-4


def auc(y, s):
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    return (stats.rankdata(s)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def calibrate(S, Y):
    """z-score each column's entries against that column's OTHER out-of-training reps."""
    out_ = np.where(Y == 0, S, np.nan)
    n = (~np.isnan(out_)).sum(0)
    tot, tot2 = np.nansum(out_, 0), np.nansum(out_ ** 2, 0)
    own = np.nan_to_num(out_)                        # leave the row's own rep out if it is "out"
    k = n - (Y == 0)
    mu = (tot - own) / np.maximum(k, 1)
    var = np.maximum((tot2 - own ** 2) / np.maximum(k, 1) - mu ** 2, 0)
    Z = (S - mu) / (np.sqrt(var) + 1e-12)
    Z[:, n < 3] = np.nan
    return Z


def summarise(name, Y, S, donors, rng, n_boot=500):
    """Mean over reps of the AUC, raw and per-record calibrated, with donor-bootstrap intervals."""
    Z = calibrate(S, Y)

    def mean_auc(cols):
        raw = np.nanmean([auc(Y[r, cols], S[r, cols]) for r in range(len(Y))])
        ok = ~np.isnan(Z[0, cols])
        cal = np.nanmean([auc(Y[r, cols][ok], Z[r, cols][ok]) for r in range(len(Y))])
        return raw, cal

    cols = np.arange(Y.shape[1])
    raw, cal = mean_auc(cols)
    uniq, inv = np.unique(donors, return_inverse=True)
    by = [np.where(inv == i)[0] for i in range(len(uniq))]
    bs = np.array([mean_auc(np.concatenate([by[i] for i in rng.integers(0, len(uniq), len(uniq))]))
                   for _ in range(n_boot)])
    lo, hi = np.nanpercentile(bs, [2.5, 97.5], axis=0)
    # null for the calibrated attack: each sample keeps its scores, its in/out
    # labels are shuffled across repetitions
    null = []
    for _ in range(5):
        Yp = np.stack([rng.permutation(Y[:, j]) for j in range(Y.shape[1])], 1)
        Zp = calibrate(S, Yp)
        null.append(np.nanmean([auc(Yp[r], Zp[r]) for r in range(len(Y))]))
    return dict(calibrated_null=float(np.mean(null)), group=name, n=Y.shape[1], n_donors=len(uniq), auc=raw, auc_lo=lo[0], auc_hi=hi[0],
                calibrated_auc=cal, calibrated_lo=lo[1], calibrated_hi=hi[1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="COMBINED")
    ap.add_argument("--generator", default="mvn")
    ap.add_argument("--reps", type=int, default=60)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--reuse", action="store_true", help="re-analyse the saved scores")
    args = ap.parse_args()
    ds = args.dataset

    expr = D.load_expression(ds)
    X = expr.values.astype(np.float32)
    y = D.encode_subtypes(ds, D.load_subtypes(ds).values)
    donors = np.array([a[:12] for a in expr.index])
    XB, mB = extras(ds)
    XBv = XB.values.astype(np.float64)
    ref = D.load_reference(ds)
    X_aux = None if ref is None else ref.values.astype(np.float64)
    tumour = (mB.tier != "matched normal").values
    half = np.unique(mB.patient[tumour])                 # donors placed half in / half out
    a_idx = np.array([expr.index.get_loc(a) for a in mB.a_sample])
    params = T.default_params(args.generator, ds)

    cache = paths.ARTIFACTS / "perturb" / f"donor_linked_resplit_{ds}_{args.generator}.npz"
    cache.parent.mkdir(parents=True, exist_ok=True)
    R = args.reps
    M = np.zeros((R, len(X)), dtype=np.int8)
    SC, SB = np.zeros((R, len(X))), np.zeros((R, len(XB)))
    t0 = time.time()
    if args.reuse:
        z = np.load(cache, allow_pickle=True)
        M, SC, SB = z["member"], z["scores_candidates"], z["scores_B"]
        R = len(M)
    for r in range(0 if args.reuse else R):
        rng = np.random.default_rng(10_000 + r)
        member = rng.random(len(X)) < 0.8
        chosen = rng.permutation(half)[: len(half) // 2]
        is_half = np.isin(donors, half)
        member[is_half] = np.isin(donors[is_half], chosen)
        for attempt in range(5):
            # LAPACK's SVD occasionally fails on a small class's covariance;
            # another generator seed (same split) gets past it.
            try:
                gen = G.build(args.generator, seed=20_000 + r + 1000 * attempt,
                              device=args.device, **params)
                gen.fit(X[member], y[member], D.n_classes(ds))
                Xs, _ = gen.sample(int(member.sum()))
                break
            except np.linalg.LinAlgError:
                print(f"rep {r}: SVD did not converge, reseeding", flush=True)
        Xs = np.asarray(Xs, dtype=np.float64)

        def score(Q):
            d_syn = mahalanobis(Q, Xs.mean(0), precision(Xs, "ridge", RIDGE))
            if X_aux is None:
                return 1.0 / (d_syn + 1e-10)
            d_aux = mahalanobis(Q, X_aux.mean(0), P_aux)
            return d_aux / (d_syn + d_aux + 1e-10)

        if r == 0 and X_aux is not None:
            P_aux = precision(X_aux, "ridge", RIDGE)
        M[r], SC[r], SB[r] = member, score(X.astype(np.float64)), score(XBv)
        if r % 10 == 0:
            print(f"rep {r}: cohort AUC {auc(M[r], SC[r]):.3f}  ({time.time() - t0:.0f}s)", flush=True)
    if not args.reuse:
        np.savez_compressed(cache, member=M, scores_candidates=SC, scores_B=SB,
                            b_ids=np.array(mB.index), candidates=np.array(expr.index))

    rng = np.random.default_rng(0)
    YB = M[:, a_idx]
    bd = mB.patient.values
    rows = [summarise("all candidates", M, SC, donors, rng, n_boot=50)]
    hsel = np.isin(donors, half)
    rows.append(summarise("A samples of the donors with a second tumour sample (overlap control)",
                          M[:, hsel], SC[:, hsel], donors[hsel], rng))
    rows.append(summarise("any second tumour sample", YB[:, tumour], SB[:, tumour], bd[tumour], rng))
    for tier in sorted(mB.tier.unique()):
        sel = (mB.tier == tier).values
        rows.append(summarise(tier, YB[:, sel], SB[:, sel], bd[sel], rng))
    df = pd.DataFrame(rows)
    df.insert(0, "generator", args.generator)
    df.insert(0, "dataset", ds)
    df["reps"] = R
    out = paths.RESULTS / "perturb" / f"donor_linked_resplit_{ds}_{args.generator}.csv"
    df.to_csv(out, index=False)
    pd.set_option("display.width", 220)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
