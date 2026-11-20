"""Target pools of increasing coverage of one PE run's private set (one run, many pools).

Private set D = the run's training CSV.  The test CSV is split once (seeded) into Ref and
non-member targets; member targets are m random records of D.  Each pool is a world folder
(members.csv, nonmembers.csv, ref.csv, train.csv, run -> the shared PE run) readable by the
attack scripts.  Coverage = m / |D|.

    python scripts/pe_lira/make_coverage_worlds.py --run artifacts/pe_worlds/adult_eps10_full \
        --train ~/privacy/nfree_data/adult_train.csv --test ~/privacy/nfree_data/adult_test.csv \
        --out artifacts/pe_worlds/adult_cov --sizes 500:3,2000:3,8000:2,all:1
"""
import argparse, os
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True); ap.add_argument("--train", required=True); ap.add_argument("--test", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--sizes", default="500:3,2000:3,8000:2,all:1")
a = ap.parse_args()
D = pd.read_csv(os.path.expanduser(a.train)); T = pd.read_csv(os.path.expanduser(a.test))
perm = np.random.default_rng(12345).permutation(len(T))
ref, non = T.iloc[perm[: len(T) // 2]], T.iloc[perm[len(T) // 2:]]
for spec in a.sizes.split(","):
    m, reps = spec.split(":")
    for r in range(int(reps)):
        mm = len(D) if m == "all" else int(m)
        w = f"{a.out}_m{m}_r{r}"
        os.makedirs(w, exist_ok=True)
        D.iloc[np.sort(np.random.default_rng(1000 * r + mm).choice(len(D), mm, replace=False))].to_csv(f"{w}/members.csv", index=False)
        non.to_csv(f"{w}/nonmembers.csv", index=False); ref.to_csv(f"{w}/ref.csv", index=False)
        for name, src in (("train.csv", os.path.expanduser(a.train)), ("run", os.path.abspath(a.run))):
            if not os.path.lexists(f"{w}/{name}"):
                os.symlink(src, f"{w}/{name}")
        print(w, mm, "members", len(non), "non-members", len(ref), "ref")
