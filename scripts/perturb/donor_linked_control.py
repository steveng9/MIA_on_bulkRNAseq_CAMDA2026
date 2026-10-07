"""Negative control for the donor-linked experiment.

Each second sample keeps its scores but is given ANOTHER donor's membership
pattern across the five splits (a random derangement within the tier).  If the
linked AUC came from anything other than the donor link -- pooling across
splits, sample type, the percentile calibration -- it would survive this.
Averaged over 200 derangements; writes results/perturb/donor_linked_control.csv.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mia import paths  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from donor_linked import extras  # noqa: E402
from donor_linked_report import auc, zscore  # noqa: E402

SRC = paths.ARTIFACTS / "perturb" / "donor_linked_scores"


def main() -> None:
    rng = np.random.default_rng(1)
    rows = []
    files = sorted(SRC.glob("*.npz"))
    for dataset, gen in sorted({tuple(f.stem.split("__")[:2]) for f in files}):
        runs = {int(re.search(r"__s(\d+)", f.stem).group(1)): np.load(f, allow_pickle=True)
                for f in files if f.stem.startswith(f"{dataset}__{gen}__s")}
        if len(runs) < 5:
            continue
        splits = sorted(runs)
        r0 = runs[splits[0]]
        pos = {a: i for i, a in enumerate(r0["candidates"])}
        Ym = np.stack([runs[s]["y_member"] for s in splits])
        _, meta = extras(dataset)
        for ai, attack in enumerate(r0["attacks"]):
            Ys, Zs = [], []
            t = 0
            while f"tier{t}_name" in r0:
                if str(r0[f"tier{t}_name"]) != "matched normal":
                    a_idx = np.array([pos[meta.loc[b, "a_sample"]] for b in r0[f"tier{t}_ids"]])
                    Ys.append(Ym[:, a_idx])
                    Zs.append(np.stack([zscore(runs[s][f"tier{t}_extras"][ai],
                                               runs[s][f"tier{t}_candidates"][ai][Ym[k] == 0])
                                        for k, s in enumerate(splits)]))
                t += 1
            Y, Z = np.concatenate(Ys, 1), np.concatenate(Zs, 1)
            true = auc(Y.ravel(), Z.ravel())
            null = []
            for _ in range(200):
                perm = rng.permutation(Y.shape[1])
                null.append(auc(Y[:, perm].ravel(), Z.ravel()))
            rows.append(dict(dataset=dataset, generator=gen, attack=str(attack), linked_auc=true,
                             control_mean=float(np.mean(null)),
                             control_p95=float(np.percentile(null, 95)),
                             control_max=float(np.max(null))))
    df = pd.DataFrame(rows)
    df.to_csv(paths.RESULTS / "perturb" / "donor_linked_control.csv", index=False)
    df["generator"] = df.generator.str.split("@").str[0]
    print(df[df.linked_auc > df.control_max].round(3).to_string(index=False))
    print("cells:", len(df), " control mean range:", df.control_mean.min().round(3),
          df.control_mean.max().round(3))


if __name__ == "__main__":
    main()
