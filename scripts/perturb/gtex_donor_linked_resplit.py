"""GTEx donor-linked attack on many redrawn splits, with per-record calibration.

The GTEx counterpart of donor_linked_resplit.py.  MVN is retrained on R fresh
donor-level splits of cohort GTEX (each candidate donor a member with
probability 0.8, all of its samples together).  Every other-tissue sample of a
candidate donor (scripts/external/gtex_prepare.py) is scored in every
repetition, so each is seen with its donor in and out; the calibrated attack
z-scores a sample against its own scores in the repetitions where its donor
was out (ideal shadows: an upper bound on what shadow modelling could reach).

Attack: MahalaMIA with the ridge and the reference set, d_aux / (d_syn + d_aux).
Writes results/perturb/gtex_donor_linked_resplit_<generator>.csv.

    python scripts/perturb/gtex_donor_linked_resplit.py --reps 100
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mia import datasets as D  # noqa: E402
from mia import generators as G  # noqa: E402
from mia import paths  # noqa: E402
from mia import targets as T  # noqa: E402
from mia.attacks.mahalamia import mahalanobis, precision  # noqa: E402
from donor_linked_resplit import RIDGE, auc, summarise  # noqa: E402
from gtex_donor_linked import GTEX, ORGAN, OTHER  # noqa: E402

DATASET = "GTEX"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator", default="mvn")
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--reuse", action="store_true", help="re-analyse the saved scores")
    args = ap.parse_args()

    expr = D.load_expression(DATASET)
    X = expr.values.astype(np.float32)
    y = D.encode_subtypes(DATASET, D.load_subtypes(DATASET).values)
    meta = pd.read_csv(GTEX / "GTEX_meta.csv", index_col=0)
    donors = meta.loc[expr.index, "donor"].values
    mB = pd.read_csv(GTEX / "GTEX_extras_meta.csv", index_col=0)
    mB = mB[mB.donor_role == "candidate"]
    XB = pd.read_csv(GTEX / "GTEX_extras_vst.tsv", sep="\t", index_col=0).loc[mB.index]
    XBv = XB.values.astype(np.float64)
    X_aux = D.load_reference(DATASET).values.astype(np.float64)
    P_aux = precision(X_aux, "ridge", RIDGE)
    params = T.default_params(args.generator, DATASET)
    udon = np.unique(donors)

    cache = paths.ARTIFACTS / "perturb" / f"gtex_donor_linked_resplit_{args.generator}.npz"
    cache.parent.mkdir(parents=True, exist_ok=True)
    R = args.reps
    MD = np.zeros((R, len(udon)), dtype=np.int8)            # donor membership per rep
    SC, SB = np.zeros((R, len(X))), np.zeros((R, len(XB)))
    t0 = time.time()
    if args.reuse:
        z = np.load(cache, allow_pickle=True)
        MD, SC, SB = z["member_donor"], z["scores_candidates"], z["scores_B"]
        R = len(MD)
    for r in range(0 if args.reuse else R):
        rng = np.random.default_rng(30_000 + r)
        MD[r] = rng.random(len(udon)) < 0.8
        member = np.isin(donors, udon[MD[r] == 1])
        for attempt in range(5):
            try:
                gen = G.build(args.generator, seed=40_000 + r + 1000 * attempt,
                              device=args.device, **params)
                gen.fit(X[member], y[member], D.n_classes(DATASET))
                Xs, _ = gen.sample(int(member.sum()))
                break
            except np.linalg.LinAlgError:
                print(f"rep {r}: SVD did not converge, reseeding", flush=True)
        Xs = np.asarray(Xs, dtype=np.float64)
        P_syn, mu_syn = precision(Xs, "ridge", RIDGE), Xs.mean(0)

        def score(Q):
            d_syn = mahalanobis(Q, mu_syn, P_syn)
            d_aux = mahalanobis(Q, X_aux.mean(0), P_aux)
            return d_aux / (d_syn + d_aux + 1e-10)

        SC[r], SB[r] = score(X.astype(np.float64)), score(XBv)
        if r % 10 == 0:
            print(f"rep {r}: cohort AUC {auc(member.astype(int), SC[r]):.3f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
    if not args.reuse:
        np.savez_compressed(cache, member_donor=MD, donors=udon, scores_candidates=SC,
                            scores_B=SB, b_ids=np.array(mB.index), candidates=np.array(expr.index))

    pos = {d: i for i, d in enumerate(udon)}
    YC = MD[:, [pos[d] for d in donors]]
    YB = MD[:, [pos[d] for d in mB.donor]]
    bd = mB.donor.values
    rng = np.random.default_rng(0)
    rows = [summarise("cohort samples (trained on)", YC, SC, donors, rng, n_boot=50)]
    for tier in ("same organ, other site", "other organ", "cell line"):
        sel = (mB.tier == tier).values
        rows.append(summarise(tier, YB[:, sel], SB[:, sel], bd[sel], rng, n_boot=200))
    for tissue in list(ORGAN) + OTHER:
        sel = (mB.SMTSD == tissue).values
        if sel.sum() >= 20:
            rows.append(summarise(tissue, YB[:, sel], SB[:, sel], bd[sel], rng, n_boot=200))
    df = pd.DataFrame(rows)
    df.insert(0, "generator", args.generator)
    df["reps"] = R
    out = paths.RESULTS / "perturb" / f"gtex_donor_linked_resplit_{args.generator}.csv"
    df.to_csv(out, index=False)
    pd.set_option("display.width", 220)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
