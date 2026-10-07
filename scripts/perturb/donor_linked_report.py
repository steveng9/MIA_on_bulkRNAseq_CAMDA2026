"""Tables for the donor-linked experiment (see donor_linked.py).

For every (dataset, generator, attack, tier):

  linked AUC      B samples of member donors against B samples of non-member
                  donors -- membership of the DONOR, inferred from a sample the
                  generator never saw.  The five splits are pooled after
                  replacing each score by its percentile among that split's
                  non-member candidates: the splits are a 5-fold partition, so
                  every B sample counts once as a non-member's and four times
                  as a member's.  The mean of per-split AUCs is kept too; it is
                  unstable when a tier has one or two non-members per split.
  overlap AUC     the same donors' A samples (the ones that were or were not
                  trained on), scored without any B sample in the pool.  The
                  control: same donors, same count, standard membership.
  cohort AUC      all candidates, for reference.
  95% interval    donors resampled with replacement (a donor's rows in all five
                  splits move together), 1000 draws.
  within donor    every donor is a non-member in exactly one split.  For one B
                  sample: the share of its four member splits in which it
                  scores (as a percentile) above its non-member split,
                  averaged over donors.  0.5 = membership of the donor does not
                  move the second sample's score.  Every donor is its own
                  control, so differences between donors cancel.

Also a linkability table, independent of any generator: how often a B sample's
most-correlated candidate is its own donor's A sample.

Writes results/perturb/donor_linked.csv and donor_linked_linkability.csv.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mia import datasets as D  # noqa: E402
from mia import paths  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from donor_linked import extras  # noqa: E402

SUFFIX = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--suffix=")), "")
ADAPTED = SUFFIX == "_adapted"
SRC = paths.RESULTS / "perturb" / f"donor_linked_scores{SUFFIX}"
N_BOOT = 1000


def auc(y, s):
    ok = ~np.isnan(s)
    y, s = y[ok], s[ok]
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    return (stats.rankdata(s)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


POOLED = "any second tumour sample (three tiers pooled)"


def zscore(s, reference):
    """Percentile of each score among `reference` (mid-rank for ties).

    Puts the five splits on one scale without assuming anything about the
    score's shape: some attacks emit saturated or heavily skewed scores, for
    which a mean/sd standardisation is not comparable across splits.
    """
    ref = np.sort(reference[~np.isnan(reference)])
    return (np.searchsorted(ref, s, "left") + np.searchsorted(ref, s, "right")) / (2 * len(ref))


def within_donor(Y, Z, donors, rng):
    """P(score when the donor is a member > score when it is not), per donor, averaged.

    Each member split is compared with the donor's one non-member split
    separately (ties count half), so the expectation is exactly 0.5 when
    membership does not matter, whatever the score distribution.
    """
    zn = np.nanmean(np.where(Y == 0, Z, np.nan), 0)
    win = np.where(Y == 1, (Z > zn) + 0.5 * (Z == zn), np.nan)
    per = pd.Series(np.nanmean(win, 0)).groupby(donors).mean().dropna().values
    if not len(per):
        return np.nan, np.nan, np.nan
    bs = [per[rng.integers(0, len(per), len(per))].mean() for _ in range(N_BOOT)]
    return float(per.mean()), *np.percentile(bs, [2.5, 97.5])


def mean_auc(Y, S, w=None):
    """Mean over splits of AUC; Y, S are splits x samples; w = bootstrap multiplicities."""
    out = []
    for y, s in zip(Y, S):
        if w is not None:
            y, s = np.repeat(y, w), np.repeat(s, w)
        out.append(auc(y, s))
    return float(np.nanmean(out))


def boot(Y, S, donors, rng):
    uniq, inv = np.unique(donors, return_inverse=True)
    vals = []
    for _ in range(N_BOOT):
        w = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))[inv]
        vals.append(mean_auc(Y, S, w))
    return np.nanpercentile(vals, [2.5, 97.5])


def linkability(dataset: str) -> pd.DataFrame:
    X, meta = extras(dataset)
    expr = D.load_expression(dataset)
    mu = expr.values.mean(0)
    A = expr.values - mu
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    B = X.values - mu
    B /= np.linalg.norm(B, axis=1, keepdims=True)
    C = B @ A.T
    own = np.array([expr.index.get_loc(a) for a in meta.a_sample])
    c_own = C[np.arange(len(C)), own]
    rank = (C > c_own[:, None]).sum(1) + 1
    Cx = C.copy()
    Cx[np.arange(len(C)), own] = -np.inf
    df = pd.DataFrame({"tier": meta.tier.values, "rank": rank, "corr_own": c_own,
                       "corr_best_other": Cx.max(1)})
    g = df.groupby("tier")
    return pd.DataFrame({
        "dataset": dataset, "n": g.size(),
        "own_donor_is_nearest": g["rank"].apply(lambda r: (r == 1).mean()),
        "own_donor_in_top10": g["rank"].apply(lambda r: (r <= 10).mean()),
        "median_rank": g["rank"].median(),
        "corr_to_own_A": g.corr_own.median(),
        "corr_to_best_other": g.corr_best_other.median()}).reset_index()


def main() -> None:
    rng = np.random.default_rng(0)
    rows, link = [], []
    files = sorted(SRC.glob("*.npz"))
    keys = sorted({tuple(f.stem.split("__")[:2]) for f in files})
    for dataset in sorted({k[0] for k in keys}):
        link.append(linkability(dataset))
    for dataset, gen in keys:
        runs = {int(re.search(r"__s(\d+)", f.stem).group(1)): np.load(f, allow_pickle=True)
                for f in files if f.stem.startswith(f"{dataset}__{gen}__s")}
        if len(runs) < 5:
            print(f"skip {dataset}/{gen}: {len(runs)} splits")
            continue
        splits = sorted(runs)
        r0 = runs[splits[0]]
        cand = list(r0["candidates"])
        pos = {a: i for i, a in enumerate(cand)}
        Ym = np.stack([runs[s]["y_member"] for s in splits])            # splits x candidates
        attacks = list(r0["attacks"])
        _, meta = extras(dataset)
        for ai, attack in enumerate(attacks):
            S_alone = np.stack([runs[s]["scores_alone"][ai] for s in splits])
            cohort = mean_auc(Ym, S_alone)
            ZA_all = np.stack([zscore(S_alone[k], S_alone[k][Ym[k] == 0])
                               for k in range(len(splits))])
            tiers, t = {}, 0
            while f"tier{t}_name" in r0:
                ids = list(r0[f"tier{t}_ids"])
                a_idx = np.array([pos[meta.loc[b, "a_sample"]] for b in ids])
                S_B = np.stack([runs[s][f"tier{t}_extras"][ai] for s in splits])
                S_C = np.stack([runs[s][f"tier{t}_candidates"][ai] for s in splits])
                # B scores against each split's non-member candidates
                Z = np.stack([zscore(S_B[k], S_C[k][Ym[k] == 0]) for k in range(len(splits))])
                tiers[str(r0[f"tier{t}_name"])] = dict(
                    donors=np.array([b[:12] for b in ids]), Y=Ym[:, a_idx], S_B=S_B, Z=Z,
                    S_A=S_alone[:, a_idx], ZA=ZA_all[:, a_idx],
                    pool_auc=np.array([mean_auc(Ym, S_C)]))
                t += 1
            tum = [v for k, v in tiers.items() if k != "matched normal"]
            if tum and not ADAPTED:
                tiers[POOLED] = {k: np.concatenate([v[k] for v in tum], axis=-1) for k in tum[0]}
            for tier, v in tiers.items():
                Y, Z, ZA, donors = v["Y"], v["Z"], v["ZA"], v["donors"]
                d5 = np.tile(donors, len(splits))
                flat = lambda A: A.reshape(1, -1)                       # noqa: E731
                lo, hi = boot(flat(Y), flat(Z), d5, rng)
                olo, ohi = boot(flat(Y), flat(ZA), d5, rng)
                win, wlo, whi = within_donor(Y, Z, donors, rng)
                owin, owlo, owhi = within_donor(Y, ZA, donors, rng)
                rows.append(dict(
                    dataset=dataset, generator=gen, attack=attack, tier=tier,
                    n_B=Y.shape[1], n_donors=len(set(donors)),
                    linked_auc=mean_auc(flat(Y), flat(Z)), linked_lo=lo, linked_hi=hi,
                    overlap_auc=mean_auc(flat(Y), flat(ZA)), overlap_lo=olo, overlap_hi=ohi,
                    linked_auc_mean_of_splits=mean_auc(Y, v["S_B"]),
                    overlap_auc_mean_of_splits=mean_auc(Y, v["S_A"]),
                    cohort_auc=cohort, cohort_auc_with_tier_in_pool=float(v["pool_auc"].mean()),
                    within_donor=win, within_lo=wlo, within_hi=whi,
                    overlap_within_donor=owin, overlap_within_lo=owlo, overlap_within_hi=owhi))
        print(f"{dataset}/{gen} done", flush=True)
    out = paths.RESULTS / "perturb"
    pd.DataFrame(rows).to_csv(out / f"donor_linked{SUFFIX}.csv", index=False)
    if not SUFFIX:
        pd.concat(link).to_csv(out / "donor_linked_linkability.csv", index=False)
        print(pd.concat(link).round(3).to_string())


if __name__ == "__main__":
    main()
