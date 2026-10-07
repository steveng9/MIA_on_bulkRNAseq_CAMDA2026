"""Donor-linked membership inference: attack a sample that was NOT in the
training data but comes from a donor whose other sample was.

Bulk analogue of the single-cell "disjoint cells" experiment.  TCGA holds a
second RNA-seq aliquot for several hundred donors of the challenge cohorts.
The challenge distributed one primary-tumour aliquot per donor (sample A); the
generators trained on the A samples of the member donors.  Here the adversary
holds a *different* sample B of a donor and asks whether the donor was in the
training data.  B samples come in four tiers of biological distance from A:

  same tumour sample, other aliquot     the same piece of tissue, re-extracted / re-sequenced
  same tumour, other vial               another piece of the same primary tumour
  other lesion                          metastasis, recurrence or a new primary
  matched normal                        adjacent non-tumour tissue of the same organ

B samples are placed in the cohort's VST space with the cohort's own frozen
transform (scripts/external/tcga_vst.py), appended to the candidate pool one
tier at a time, and scored against the existing targets.  Nothing is retrained:
no B sample was ever seen by a generator.

Per target this writes `artifacts/perturb/donor_linked_scores/<...>.npz` with the
score of every candidate and every B sample under every attack;
`donor_linked_report.py` turns those into the tables.

    python scripts/perturb/donor_linked.py --dataset COMBINED --jobs 8
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mia import datasets as D  # noqa: E402
from mia import paths  # noqa: E402
from mia import targets as T  # noqa: E402
from mia import views as V  # noqa: E402

TCGA = Path("~/data/TCGA_GDC/processed").expanduser()
GENERATORS = ["mvn", "cvae", "nd", "pgg",
              "pgm@binning=dp_quantile,edge_estimator=threshold,n_bins=16",
              "tabsyn@class_freq=train,diff_schedule=steps,latent_scale=std,"
              "preprocess=clip:0.001:0.999+standard", "tabpfn"]
ATTACKS = ["MahalaMIA (as submitted)", "MahalaMIA (ridge 1e-4)", "MahalaMIA (no aux)",
           "MahalaMIA (no aux, ridge 1e-4)", "MAMA-MIA v1", "MAMA-MIA v2", "RedSigma",
           "GAN-leaks", "MC", "GAN-leaks cal."]


def extras(dataset: str) -> tuple:
    """(B-sample expression in the cohort's space, their metadata) for donors in the cohort."""
    meta = pd.read_csv(TCGA / "extras_meta.csv").set_index("aliquot")
    meta = meta[meta[f"donor_in_{dataset}"]]
    X = pd.read_csv(TCGA / f"{dataset}_extras_vst.tsv", sep="\t", index_col=0).loc[meta.index]
    lab = D.load_subtypes(dataset)
    by_donor = pd.Series(lab.values, index=[a[:12] for a in lab.index])
    a_of = pd.Series(lab.index, index=[a[:12] for a in lab.index])
    meta["label"] = by_donor.loc[meta.patient].values
    meta["a_sample"] = a_of.loc[meta.patient].values
    return X, meta


def align_to_release(B: pd.DataFrame, labels, target: dict, classes, scale: bool) -> pd.DataFrame:
    """Move a tier onto the release, class by class: subtract the tier's class
    mean and add the release's (and match the spread when `scale`).  All the
    adversary needs is a handful of samples of the same kind (e.g. normals of
    the organ) and the labelled release.  Classes with fewer than five tier
    samples use the tier-wide shift instead."""
    out = B.values.astype(np.float64).copy()
    Xs, ys = target["X"].astype(np.float64), np.asarray(target["y_str"]).astype(str)
    labels = np.asarray(labels).astype(str)
    for c in np.unique(labels):
        rows = labels == c
        src = out[rows] if rows.sum() >= 5 else out
        syn = Xs[ys == c] if (ys == c).sum() >= 5 else Xs
        z = out[rows] - src.mean(0)
        if scale:
            z = z / (src.std(0) + 1e-9) * syn.std(0)
        out[rows] = z + syn.mean(0)
    return pd.DataFrame(out, index=B.index, columns=B.columns)


def one_target(args):
    dataset, generator, split, out_dir, adapt = args
    out = Path(out_dir) / f"{dataset}__{generator[:60]}__s{split}.npz"
    if out.exists():
        return f"{out.name} cached"
    if not T.exists(dataset, generator, split):
        return f"{dataset}/{generator}/s{split} not built"
    t0 = time.time()
    X, meta = extras(dataset)
    cand = list(D.load_expression(dataset).index)
    has_ref = D.load_reference(dataset) is not None
    attacks = [a for a in ATTACKS if has_ref or a != "GAN-leaks cal."]
    payload = {"candidates": np.array(cand), "y_member": D.membership_labels(dataset, split),
               "attacks": np.array(attacks)}

    def run(key, **kw):
        rows = []
        with V.view(dataset, **kw) as expr:
            for label in attacks:
                try:
                    rows.append(V.build(label).score(dataset, generator, split))
                except Exception:
                    print(f"  FAILED {out.name} {label} {key}\n" + traceback.format_exc(limit=2),
                          flush=True)
                    rows.append(np.full(len(expr), np.nan))
        return np.stack(rows)

    payload["scores_alone"] = run("alone")                     # attacks x candidates
    groups = [(tier, m, X.loc[m.index]) for tier, m in meta.groupby("tier")]
    if adapt:
        tg = T.load_target(dataset, generator, split)
        groups = [(f"{tier}, {how}", m, align_to_release(B, m.label.values, tg, None, scale))
                  for tier, m, B in groups
                  for how, scale in (("mean-aligned to the release", False),
                                     ("mean- and scale-aligned to the release", True))]
    for i, (tier, m, B) in enumerate(groups):
        S = run(tier, extra=B, extra_labels=m.label.values)
        payload[f"tier{i}_name"] = tier
        payload[f"tier{i}_ids"] = np.array(m.index)
        payload[f"tier{i}_candidates"] = S[:, :len(cand)]      # with the tier appended
        payload[f"tier{i}_extras"] = S[:, len(cand):]
    np.savez_compressed(out, **payload)
    return f"{out.name} in {time.time() - t0:.0f}s"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--splits", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--generators", nargs="+", default=GENERATORS)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--adapt", action="store_true",
                    help="score each tier after aligning it to the release (adaptive adversary)")
    args = ap.parse_args()
    out_dir = paths.ARTIFACTS / "perturb" / ("donor_linked_scores" + ("_adapted" if args.adapt else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(args.dataset, g, s, str(out_dir), args.adapt)
            for g in args.generators for s in args.splits]
    with ProcessPoolExecutor(args.jobs) as pool:
        for msg in pool.map(one_target, jobs):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
