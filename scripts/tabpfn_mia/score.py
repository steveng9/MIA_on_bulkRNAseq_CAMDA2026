"""Score ICL-MIA runs from `attack.py` (two-stage layout in results/tabpfn_mia/full).

Merges column ranges, aligns the split's release term with the shared pool
term by sample id, and reports AUC / TPR at 1% FPR for `zsum` and `typ`, with
the AUC at growing column counts (every-k-th gene subsets).

    python scripts/tabpfn_mia/score.py [dir] [dataset] [n_estimators]
"""

import glob
import re
import sys

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from attack import scores  # noqa: E402


def tpr_at(y, s, fpr=0.01):
    f, t, _ = roc_curve(y, s)
    return float(np.interp(fpr, f, t))


def merged(pattern, key):
    files = sorted(glob.glob(pattern), key=lambda f: int(re.search(r"_c(\d+)-", f).group(1)))
    if not files:
        return None
    zs = [np.load(f) for f in files]
    cols = np.concatenate([z["cols"] for z in zs])
    assert len(set(cols)) == len(cols), f"overlapping column ranges in {pattern}"
    out = {k: zs[0][k] for k in zs[0].files if k not in (key, "cols")}
    out[key] = np.concatenate([z[key] for z in zs], axis=1)
    out["cols"] = cols
    return out


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "results/tabpfn_mia/full"
    ds = sys.argv[2] if len(sys.argv) > 2 else "BRCA"
    n = sys.argv[3] if len(sys.argv) > 3 else "2"
    ref = merged(f"{out}/ref_{ds}_n{n}_c*.npz", "l_ref")
    rows = []
    for split in range(1, 6):
        rel = merged(f"{out}/rel_{ds}_s{split}_n{n}_c*.npz", "l_rel")
        if rel is None:
            continue
        assert (rel["ids"] == ref["ids"]).all()
        common = np.intersect1d(rel["cols"], ref["cols"])
        jr = np.searchsorted(rel["cols"], common)
        jf = np.searchsorted(ref["cols"], common)
        lr, lf, m = rel["l_rel"][:, jr], ref["l_ref"][:, jf], rel["member"]
        s = scores(lr, lf)
        curve = {k: roc_auc_score(m, scores(lr[:, ::k], lf[:, ::k])["typ"]) for k in (10, 4, 2, 1)}
        rows.append([roc_auc_score(m, s["zsum"]), tpr_at(m, s["zsum"]),
                     roc_auc_score(m, s["typ"]), tpr_at(m, s["typ"])])
        print(f"s{split} ({len(common)} genes): zsum AUC {rows[-1][0]:.3f} TPR@1% {rows[-1][1]:.3f} | "
              f"typ AUC {rows[-1][2]:.3f} TPR@1% {rows[-1][3]:.3f} | typ AUC using every "
              + ", ".join(f"{k}th: {v:.3f}" for k, v in curve.items()))
    if rows:
        r = np.array(rows)
        print(f"mean over {len(r)} splits: zsum AUC {r[:,0].mean():.3f} +- {r[:,0].std():.3f}, "
              f"TPR@1% {r[:,1].mean():.3f}; typ AUC {r[:,2].mean():.3f} +- {r[:,2].std():.3f}, "
              f"TPR@1% {r[:,3].mean():.3f}")


if __name__ == "__main__":
    main()
