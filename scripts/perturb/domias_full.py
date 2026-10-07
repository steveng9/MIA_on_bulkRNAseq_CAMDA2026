"""DOMIAS on all 978 genes, without PCA (TCGA-COMBINED).

The corrected DOMIAS of gene_subsets.py (one PCA fitted on the reference set,
log density ratio) uses scipy's Gaussian KDE, whose kernel is the data
covariance.  That kernel cannot be fitted on all 978 genes: the reference set
has 824 samples, so its covariance is singular.  Here the three sets are
standardised as in the baseline and the densities use an isotropic Gaussian
kernel with Scott's bandwidth, which is defined for any dimension.

    python scripts/perturb/domias_full.py          -> results/perturb/domias_full.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mia import datasets as D  # noqa: E402
from mia import paths  # noqa: E402
from mia import targets as T  # noqa: E402
from mia.attacks.generic import _sq_dists  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from donor_linked import GENERATORS  # noqa: E402
from gene_subsets import std  # noqa: E402


def log_kde(t, x):
    """log density at `t` of an isotropic Gaussian KDE on `x` (Scott's bandwidth)."""
    n, d = x.shape
    h2 = n ** (-2.0 / (d + 4))
    return logsumexp(-_sq_dists(t, x) / (2 * h2), axis=1) - np.log(n) - 0.5 * d * np.log(2 * np.pi * h2)


def main() -> None:
    dataset, rows = "COMBINED", []
    Xt, Xr = std(D.load_expression(dataset).values), std(D.load_reference(dataset).values)
    log_r = log_kde(Xt, Xr)
    for gen in GENERATORS:
        for split in range(1, 6):
            if not T.exists(dataset, gen, split):
                continue
            y = D.membership_labels(dataset, split)
            s = log_kde(Xt, std(T.load_target(dataset, gen, split)["X"])) - log_r
            fpr, tpr, _ = roc_curve(y, s)
            rows.append(dict(dataset=dataset, generator=gen, split=split, attack="DOMIAS-KDE",
                             rule="none (978 genes, isotropic kernel)", k=Xt.shape[1],
                             auc=roc_auc_score(y, s), tpr_at_fpr_0_01=np.interp(0.01, fpr, tpr),
                             tpr_at_fpr_0_1=np.interp(0.1, fpr, tpr)))
            print(gen[:20], split, round(rows[-1]["auc"], 3), flush=True)
    pd.DataFrame(rows).to_csv(paths.RESULTS / "perturb" / "domias_full.csv", index=False)


if __name__ == "__main__":
    main()
