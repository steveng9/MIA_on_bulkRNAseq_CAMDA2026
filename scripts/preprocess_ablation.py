#!/usr/bin/env python
"""How much does model-level preprocessing move a generator?

    python scripts/preprocess_ablation.py --dataset BRCA --generator cvae \\
        --specs standard quantile minmax clip:0.001:0.999+standard \\
                classcenter+standard standard+pca:64 --splits 1

For each spec this builds the target `<generator>@preprocess=<spec>` (a normal
target variant: cached, attackable, listed by `scripts/zoo.py`) and scores it
with `mia.fidelity`.  Rows go to results/preprocess_ablation.csv, de-duplicated
on (dataset, target, split); `--table` prints the per-spec means as Markdown.

Extra generator parameters apply to every arm: `--set model_version=v2`.
Only fidelity is scored here.  Attack success on the same targets comes from
the ordinary experiment runner -- name the targets in a config's `generators:`.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mia import datasets as D  # noqa: E402
from mia import fidelity as F  # noqa: E402
from mia import preprocessing as pp  # noqa: E402
from mia import targets as T  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "results" / "preprocess_ablation.csv"
SHOW = ["utility_ratio", "tstr_macro_f1", "wasserstein_mean", "corr_mae",
        "discriminator_auc"]


def score(dataset, target, split, seed=0, max_genes=300):
    X_all = D.load_expression(dataset).values.astype(np.float64)
    y_all = D.encode_subtypes(dataset, D.load_subtypes(dataset).values)
    member = D.membership_labels(dataset, split).astype(bool)
    tgt = T.load_target(dataset, target, split)
    return F.evaluate(tgt["X"], tgt["y_int"], X_all[~member], y_all[~member],
                      X_all[member], y_all[member], seed=seed, max_genes=max_genes)


def table(df):
    g = (df.groupby(["dataset", "generator", "preprocess"], sort=False)[SHOW + ["build_seconds"]]
           .mean().round(3).reset_index())
    g.insert(3, "splits", df.groupby(["dataset", "generator", "preprocess"], sort=False)
             .size().values)
    return g.to_markdown(index=False)


def save(row):
    df = pd.DataFrame([row])
    if OUT.exists():
        df = pd.concat([pd.read_csv(OUT), df], ignore_index=True)
    df.drop_duplicates(subset=["dataset", "target", "split"], keep="last").to_csv(OUT, index=False)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="BRCA")
    p.add_argument("--generator", default="cvae")
    p.add_argument("--specs", nargs="+", default=["standard", "quantile"])
    p.add_argument("--splits", nargs="+", type=int, default=[1])
    p.add_argument("--set", nargs="*", default=[], metavar="K=V",
                   help="generator parameters shared by every arm")
    p.add_argument("--device", default="cuda")
    p.add_argument("--table", action="store_true", help="print the table and exit")
    args = p.parse_args()

    if args.table:
        print(table(pd.read_csv(OUT)))
        return

    shared = {k: yaml.safe_load(v) for k, v in (kv.split("=", 1) for kv in args.set)}
    rows = []
    for spec in args.specs:
        pp.describe(spec)                      # fail on a typo before training
        target = T.variant_name(args.generator, args.dataset,
                                {**shared, "preprocess": spec})
        for split in args.splits:
            t = time.time()
            built = not T.exists(args.dataset, target, split)
            T.build_target(args.dataset, target, split, device=args.device)
            secs = time.time() - t if built else float("nan")
            row = score(args.dataset, target, split)
            row.update(dataset=args.dataset, generator=args.generator, preprocess=spec,
                       target=target, split=split, build_seconds=secs,
                       **{f"pp_{k}": v for k, v in pp.describe(spec).items() if k != "spec"})
            rows.append(row)
            save(row)                          # one row at a time: a late arm failing loses nothing
            print(f"  {target} s{split}: " + "  ".join(f"{k}={row[k]:.3f}" for k in SHOW),
                  flush=True)

    df = pd.read_csv(OUT)
    print(f"\n{len(rows)} rows -> {OUT}\n")
    print(table(df[(df.dataset == args.dataset) & (df.generator == args.generator)]))


if __name__ == "__main__":
    main()
