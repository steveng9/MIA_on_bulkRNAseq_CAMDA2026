"""Can a second sample be linked to its donor's training sample at all?

Generator-free upper-level question behind the donor-linked attack: for a B
sample, where does its own donor's A sample rank among all candidates by
correlation?  Asked in the raw space and after the adversary removes structure
that is not donor identity:

  centre     subtract the class mean, separately for A samples (tumours of that
             class) and for the B tier (e.g. normals of that organ), so the
             tumour-versus-normal shift is gone
  drop k     additionally project out the k leading principal components of
             the class-centred candidate pool (shared programmes: purity,
             proliferation, stroma) and compare what is left

Writes results/perturb/donor_linkability.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mia import datasets as D  # noqa: E402
from mia import paths  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from donor_linked import extras  # noqa: E402


def unit(M):
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)


def main() -> None:
    rows = []
    for dataset in ("BRCA", "COMBINED"):
        X, meta = extras(dataset)
        expr = D.load_expression(dataset)
        lab = D.load_subtypes(dataset).values
        own = np.array([expr.index.get_loc(a) for a in meta.a_sample])
        A0 = expr.values.astype(np.float64)
        Ac = A0.copy()
        for c in np.unique(lab):
            Ac[lab == c] -= A0[lab == c].mean(0)
        _, _, Vt = np.linalg.svd(Ac - Ac.mean(0), full_matrices=False)
        for tier, m in meta.groupby("tier"):
            rowsel = meta.index.get_indexer(m.index)
            B0 = X.values[rowsel].astype(np.float64)
            Bc = B0.copy()
            for c in np.unique(m.label):
                sel = (m.label == c).values
                Bc[sel] -= B0[sel].mean(0)
            variants = {"raw (gene-centred)": (A0 - A0.mean(0), B0 - A0.mean(0))}
            for k in (0, 5, 10, 20, 50, 100, 200, 400):
                P = Vt[:k]
                variants[f"class-centred, drop {k} PCs"] = (Ac - Ac @ P.T @ P, Bc - Bc @ P.T @ P)
            for name, (A, B) in variants.items():
                C = unit(B) @ unit(A).T
                c_own = C[np.arange(len(C)), own[rowsel]]
                rank = (C > c_own[:, None]).sum(1) + 1
                rows.append(dict(dataset=dataset, tier=tier, n=len(m), space=name,
                                 own_is_nearest=(rank == 1).mean(), own_in_top10=(rank <= 10).mean(),
                                 median_rank=float(np.median(rank)), n_candidates=len(A),
                                 median_corr_own=float(np.median(c_own))))
    df = pd.DataFrame(rows)
    df.to_csv(paths.RESULTS / "perturb" / "donor_linkability.csv", index=False)
    pd.set_option("display.width", 220)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
