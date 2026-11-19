"""Compare the oracle ceilings with ICL-MIA on the same candidates and columns.

  exact    : oracle under the generator's TabPFN seed (needs D_-x and the seed)
  smoothed : oracle averaged over other seeds (x's influence on the release)
  ICL-MIA  : the passive attack, restricted to the same candidates and columns
Also: the oracle restricted to the k release rows nearest each candidate, to
see where in the release the information sits.
"""

import glob
import sys

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, __file__.rsplit("/", 1)[0])
out = sys.argv[1] if len(sys.argv) > 1 else "results/tabpfn_mia"
ds, split = "BRCA", 1


def tpr(m, s, f=0.05):
    a, b, _ = roc_curve(m, s)
    return float(np.interp(f, a, b))


runs = {int(np.load(f)["rs"]): np.load(f) for f in glob.glob(f"{out}/oracle_{ds}_s{split}_rs*.npz")}
gen_rs = 42 + split
z0 = next(iter(runs.values()))
pick, m, cols = z0["pick"], z0["member"], list(z0["cols"])

icl = np.load(f"{out}/{ds}_s{split}.npz")
jcols = [list(icl["cols"]).index(c) for c in cols]
d = icl["l_rel"][pick][:, jcols] - icl["l_ref"][pick][:, jcols]


def row(name, s):
    print(f"{name:34s} AUC {roc_auc_score(m, s):.3f}   TPR@5%FPR {tpr(m, s):.3f}")


print(f"{len(pick)} candidates ({m.sum()} members), genes {cols[0]}..{cols[-1]}, split {split}")
print("(TPR at 5% FPR: with 100 non-members, 1% FPR is a single record)")
row("ICL-MIA (passive, same columns)", d.sum(1))
if gen_rs in runs:
    row(f"oracle, exact (seed {gen_rs})", runs[gen_rs]["diff"].sum((1, 2)))
others = [r for r in runs if r != gen_rs]
if others:
    sm = np.mean([runs[r]["diff"] for r in others], axis=0)
    row(f"oracle, smoothed (seeds {others})", sm.sum((1, 2)))
    for r in others:
        row(f"  single seed {r}", runs[r]["diff"].sum((1, 2)))
    # Where does the information sit?  Only the release rows nearest each candidate.
    from data import load
    t = load(ds, split)
    mu, sd = t.X.mean(0), t.X.std(0) + 1e-6
    A, B = (t.X[pick] - mu) / sd, (t.S - mu) / sd
    dist = ((A[:, None, :] - B[None]) ** 2).sum(-1)
    order = np.argsort(dist, axis=1)
    for k in (10, 50, 200):
        idx = order[:, :k]
        s = np.take_along_axis(sm.sum(1), idx, axis=1).sum(1)
        row(f"  smoothed, {k} nearest release rows", s)
