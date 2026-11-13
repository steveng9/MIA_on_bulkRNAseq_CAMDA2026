#!/usr/bin/env python
"""MahalaMIA ablation: what each part of the paper's default buys.

    python scripts/mahalamia_ablation.py --workers 3
    python scripts/mahalamia_ablation.py --latex          # appendix table from the saved CSV

The paper's MahalaMIA measures a candidate's distance to its own class in the
release, with class covariance (1 - g) S_c + g S (g = 0.5), a ridge of 1e-4
times the mean gene variance, and the ratio to the reference set where one
exists (configs/experiments/paper_grid_*.yaml).  This script scores the same
targets with one setting changed at a time:

  covariance model   pooled mean and covariance (the challenge attack, with the
                     pseudo-inverse and with the ridge), pure per-class (g = 0),
                     and g from 0.1 to 1;
  ridge              1e-8 to 1e-2 at g = 0.5;
  reference set      the default without the reference set.

It calls the attack directly and writes results/mahalamia_ablation.csv, one row
per (dataset, generator, split, variant), so nothing here enters results/index.csv.
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mia import datasets as D  # noqa: E402
from mia import metrics as M  # noqa: E402
from mia import targets as T  # noqa: E402
from mia.attacks.mahalamia import MahalaMIA  # noqa: E402

OUT = ROOT / "results" / "mahalamia_ablation.csv"
TABSYN = ("tabsyn@class_freq=train,diff_schedule=steps,latent_scale=std,"
          "preprocess=clip:0.001:0.999+standard")
GENERATORS = ["mvn", "cvae", "nd", TABSYN, "tabpfn"]
NAMES = {"mvn": "MVN", "cvae": "CVAE", "nd": "NoisyDiffusion", TABSYN: "TabSyn", "tabpfn": "TabPFN"}
DEFAULT = dict(use_reference=True, covariance="ridge", ridge_alpha=1e-4, class_shrink=0.5)

# (group, label, parameters).  The default appears once, under "default".
VARIANTS = [
    ("default", "MahalaMIA (default)", DEFAULT),
    ("covariance", "pooled, pseudo-inverse (challenge)", dict(use_reference=True, covariance="pinv")),
    ("covariance", "pooled, ridge", dict(use_reference=True, covariance="ridge", ridge_alpha=1e-4)),
    ("covariance", "own class only ($\\gamma=0$)", {**DEFAULT, "class_shrink": 0.0}),
    ("covariance", "$\\gamma=0.1$", {**DEFAULT, "class_shrink": 0.1}),
    ("covariance", "$\\gamma=0.25$", {**DEFAULT, "class_shrink": 0.25}),
    ("covariance", "$\\gamma=0.75$", {**DEFAULT, "class_shrink": 0.75}),
    ("covariance", "$\\gamma=0.9$", {**DEFAULT, "class_shrink": 0.9}),
    ("covariance", "own class mean, pooled covariance ($\\gamma=1$)", {**DEFAULT, "class_shrink": 1.0}),
    ("ridge", "ridge $10^{-8}$", {**DEFAULT, "ridge_alpha": 1e-8}),
    ("ridge", "ridge $10^{-6}$", {**DEFAULT, "ridge_alpha": 1e-6}),
    ("ridge", "ridge $10^{-2}$", {**DEFAULT, "ridge_alpha": 1e-2}),
    ("reference", "no reference set", {**DEFAULT, "use_reference": False}),
]


def one(job) -> list:
    ds, gen, split = job
    if not T.exists(ds, gen, split):
        return []
    y = D.membership_labels(ds, split)
    has_ref = D.load_reference(ds) is not None
    rows = []
    for group, label, params in VARIANTS:
        if group == "reference" and not has_ref:
            continue
        s = MahalaMIA(calibrate=False, **params).score(ds, gen, split)
        m = M.evaluate(y, s)
        rows.append(dict(dataset=ds, generator=gen, split=split, group=group, variant=label,
                         auc=m["auc"], tpr_at_fpr_1=m.get("tpr_at_fpr_0.01", np.nan),
                         tpr_at_fpr_10=m.get("tpr_at_fpr_0.1", np.nan)))
    print(f"  {ds} {NAMES[gen]} split {split}", flush=True)
    return rows


def latex() -> None:
    """Rows of the appendix table: mean AUC over splits, variants by cohort and generator."""
    df = pd.read_csv(OUT)
    cols = [(ds, g) for ds in ("BRCA", "COMBINED", "GTEX") for g in GENERATORS
            if ((df.dataset == ds) & (df.generator == g)).any()]
    mean = df.groupby(["dataset", "generator", "variant"]).auc.mean()
    last = None
    for group, label, _ in VARIANTS:
        if group != last and last is not None:
            print("\\hline")
        last = group
        cells = []
        for ds, g in cols:
            v = mean.get((ds, g, label), np.nan)
            n = df[(df.dataset == ds) & (df.generator == g) & (df.variant == label)].split.nunique()
            cells.append("--" if np.isnan(v) else f"{v:.2f}" + ("$^{*}$" if n < 5 else ""))
        print(f"{label} & " + " & ".join(cells) + " \\\\")
    print("% columns: " + ", ".join(f"{ds}/{NAMES[g]}" for ds, g in cols))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", default=["BRCA", "COMBINED", "GTEX"])
    p.add_argument("--generators", nargs="+", default=GENERATORS)
    p.add_argument("--splits", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--latex", action="store_true")
    args = p.parse_args()
    if args.latex:
        return latex()
    jobs = [(ds, g, s) for ds in args.datasets for g in args.generators for s in args.splits]
    rows = []
    with ProcessPoolExecutor(args.workers) as pool:
        for r in pool.map(one, jobs):
            rows += r
    new = pd.DataFrame(rows)
    key = ["dataset", "generator", "split", "variant"]
    if OUT.exists():
        new = pd.concat([pd.read_csv(OUT), new]).drop_duplicates(key, keep="last")
    new.sort_values(key).to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(new)} rows)")
    print(new.groupby(["dataset", "generator", "variant"]).auc.mean().unstack("variant").round(3)
          .rename(index=NAMES, level=1).T.to_string())


if __name__ == "__main__":
    main()
