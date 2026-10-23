"""Corrected DOMIAS-KDE on the GTEx targets (one PCA fitted on the reference set,
log density ratio; aux_mismatch.domias_baseline).  The grid's `generic_domias_kde`
is the uncorrected benchmark version and scores 0.50 (tracker item 0e).

    python scripts/perturb/gtex_domias.py mvn cvae nd
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mia import datasets as D  # noqa: E402
from aux_mismatch import domias_baseline  # noqa: E402


def main() -> None:
    rows = []
    for g in sys.argv[1:]:
        for s in range(1, 6):
            y = D.membership_labels("GTEX", s)
            sc = domias_baseline("GTEX", g, s)
            fpr, tpr, _ = roc_curve(y, sc)
            rows.append(dict(generator=g, split=s, auc=roc_auc_score(y, sc),
                             tpr_at_fpr_0_01=np.interp(0.01, fpr, tpr)))
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    print(df.groupby("generator")[["auc", "tpr_at_fpr_0_01"]].agg(["mean", "std"]).round(3))


if __name__ == "__main__":
    main()
