"""Donor-linked membership inference on GTEx: another tissue of a training donor.

TCGA's second samples are tumour or adjacent normal tissue of the same organ.
GTEx takes about twenty tissues from every donor, so it tests the case TCGA
cannot: the adversary holds a sample of *another tissue* of the donor (or the
same organ at another site) and asks whether the donor's samples trained the
generator.  Cohort GTEX (configs/datasets/GTEX.yaml) is ten tissues; every
other GTEx sample of its candidate donors was placed in the release's space
with the cohort's frozen VST (scripts/external/gtex_prepare.py) and was never
seen by a generator.

Groups, appended to the candidate pool one at a time:
  same organ, other site   Colon - Sigmoid, Esophagus - Muscularis, Esophagus -
                           Gastroesophageal Junction, Skin - Not Sun Exposed,
                           Kidney - Medulla (labelled as their organ's class)
  other organ              the largest other tissues (labelled with the nearest
                           class of the release, since the attacks need one)

A B sample is a member when its donor is a member of the split; AUC is over B
samples per split, averaged over the five splits.  `trained_auc` is the same
attack on the cohort's own candidates.  Nothing is retrained.

    python scripts/perturb/gtex_donor_linked.py --jobs 6
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mia import datasets as D  # noqa: E402
from mia import metrics as M  # noqa: E402
from mia import paths  # noqa: E402
from mia import targets as T  # noqa: E402
from mia import views as V  # noqa: E402

GTEX = Path("~/data/GTEx/processed").expanduser()
DATASET = "GTEX"
GENERATORS = ["mvn", "cvae", "nd", "pgg", "pgm@binning=dp_quantile,edge_estimator=threshold,n_bins=16"]
ATTACKS = ["MahalaMIA (as submitted)", "MahalaMIA (ridge 1e-4)", "MahalaMIA (no aux)",
           "MahalaMIA (no aux, ridge 1e-4)", "MAMA-MIA v1", "MAMA-MIA v2", "RedSigma",
           "GAN-leaks", "MC", "GAN-leaks cal."]
ORGAN = {"Colon - Sigmoid": "Colon", "Esophagus - Muscularis": "Esophagus",
         "Esophagus - Gastroesophageal Junction": "Esophagus",
         "Skin - Not Sun Exposed (Suprapubic)": "Skin", "Kidney - Medulla": "Kidney"}
OTHER = ["Muscle - Skeletal", "Whole Blood", "Adipose - Subcutaneous", "Artery - Tibial",
         "Thyroid", "Nerve - Tibial", "Testis", "Cells - Cultured fibroblasts"]


def extras() -> tuple:
    meta = pd.read_csv(GTEX / "GTEX_extras_meta.csv", index_col=0)
    meta = meta[(meta.donor_role == "candidate") & meta.SMTSD.isin(list(ORGAN) + OTHER)]
    X = pd.read_csv(GTEX / "GTEX_extras_vst.tsv", sep="\t", index_col=0).loc[meta.index]
    return X, meta


def nearest_class(B: np.ndarray, tg: dict) -> np.ndarray:
    ys = np.asarray(tg["y_str"]).astype(str)
    classes = np.unique(ys)
    mu = np.stack([tg["X"][ys == c].mean(0) for c in classes])
    d = ((B[:, None, :] - mu[None]) ** 2).sum(-1)
    return classes[d.argmin(1)]


def one_target(args):
    generator, split, out_dir = args
    out = Path(out_dir) / f"{generator[:60]}__s{split}.csv"
    if out.exists():
        return f"{out.name} cached"
    if not T.exists(DATASET, generator, split):
        return f"{generator}/s{split} not built"
    t0 = time.time()
    X, meta = extras()
    tg = T.load_target(DATASET, generator, split)
    cand = D.load_expression(DATASET).index
    y_c = D.membership_labels(DATASET, split)
    member_donor = set(pd.read_csv(GTEX / "GTEX_meta.csv", index_col=0).loc[cand[y_c == 1], "donor"])
    rows = []

    def run(**kw):
        res = {}
        with V.view(DATASET, **kw):
            for label in ATTACKS:
                try:
                    res[label] = np.asarray(V.build(label).score(DATASET, generator, split))
                except Exception:
                    print(f"  FAILED {generator}/s{split} {label}\n"
                          + traceback.format_exc(limit=2), flush=True)
        return res

    alone = run()
    trained = {a: M.evaluate(y_c, s)["auc"] for a, s in alone.items()}
    for tissue in list(ORGAN) + OTHER:
        m = meta[meta.SMTSD == tissue]
        if len(m) < 20:
            continue
        B = X.loc[m.index]
        lab = (np.full(len(m), ORGAN[tissue]) if tissue in ORGAN
               else nearest_class(B.values.astype(np.float64), tg))
        y_b = m.donor.isin(member_donor).values.astype(int)
        S = run(extra=B, extra_labels=lab)
        for a, s in S.items():
            sb = s[len(cand):]
            met = M.evaluate(y_b, sb)
            rows.append(dict(generator=generator, split=split, tissue=tissue,
                             group="same organ, other site" if tissue in ORGAN else "other organ",
                             attack=a, n_b=len(m), n_member_b=int(y_b.sum()), auc=met["auc"],
                             tpr_at_fpr_0_01=met["tpr_at_fpr_0.01"], trained_auc=trained.get(a)))
    pd.DataFrame(rows).to_csv(out, index=False)
    return f"{out.name}: {len(rows)} rows in {time.time() - t0:.0f}s"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--generators", nargs="+", default=GENERATORS)
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args()
    out_dir = paths.RESULTS / "perturb" / "gtex_donor_linked_parts"
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(g, s, str(out_dir)) for g in args.generators for s in args.splits]
    pool = ProcessPoolExecutor(args.jobs)
    for msg in pool.map(one_target, jobs):
        print(msg, flush=True)
    parts = [pd.read_csv(f) for f in sorted(out_dir.glob("*.csv"))]
    pd.concat(parts).to_csv(paths.RESULTS / "perturb" / "gtex_donor_linked.csv", index=False)
    # as in candidate_mismatch.py: do not wait on the pool's shutdown
    for proc in list(pool._processes.values()):
        proc.terminate()
    os._exit(0)


if __name__ == "__main__":
    main()
