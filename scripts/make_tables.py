#!/usr/bin/env python
"""Aggregate results/index.csv into the attack x generator grid.

    python scripts/make_tables.py --config configs/experiments/grid_brca.yaml
    python scripts/make_tables.py --dataset BRCA --format latex
    python scripts/make_tables.py --dataset BRCA --out results/tables/

Each cell is one attack scored against one generator, averaged over splits, in
the abstract's four metrics: AUC, AUPR, TPR at 1% FPR, TPR at 10% FPR.

Rows are keyed by (attack, tag), so hyperparameter variants of the same attack
stay separate instead of being silently averaged together.  Passing --config
narrows the table to exactly the variants that experiment defines, which is what
you want for the headline grid; without it every recorded variant is shown.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mia import runs as R  # noqa: E402

METRICS = ["auc", "aupr", "tpr_at_fpr_0.01", "tpr_at_fpr_0.1"]
METRIC_LABELS = {"auc": "AUC", "aupr": "AUPR",
                 "tpr_at_fpr_0.01": "T@1", "tpr_at_fpr_0.1": "T@10"}
ATTACK_ORDER = ["mahalamia", "melomia_cvae", "melomia_nd", "mamamia"]
GENERATOR_ORDER = ["mvn", "cvae", "nd", "pgm"]
GENERATOR_LABELS = {"mvn": "MVN (noise=0.7)", "cvae": "CVAE",
                    "nd": "NoisyDiffusion", "pgm": "DP-PGM (eps=10)"}


def load(dataset=None, tag=None) -> pd.DataFrame:
    df = R.load_index()
    if df.empty:
        raise SystemExit("No runs recorded yet. Run scripts/run_experiment.py first.")
    if dataset:
        df = df[df.dataset == dataset]
    if tag:
        df = df[df.tag == tag]
    for c in METRICS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["tag"] = df["tag"].fillna("")
    return df


def summarise(df: pd.DataFrame, labels=None) -> pd.DataFrame:
    """Mean, std and split count per (attack variant, generator) cell."""
    g = df.groupby(["attack", "tag", "generator"])
    out = g[METRICS].agg(["mean", "std", "count"])
    out.columns = [f"{m}_{s}" for m, s in out.columns]
    out = out.reset_index()
    labels = labels or {}
    out["row"] = [labels.get((a, t), a if not t or t in ("default", "aux") else f"{a} [{t}]")
                  for a, t in zip(out.attack, out.tag)]
    return out

def selection_from_config(config_path) -> tuple:
    """Resolve an experiment YAML into the (attack, tag) pairs it produces.

    Run identity is a hash of the attack's parameters, so a run belongs to
    whichever configurations share those parameters -- the experiment that
    happened to trigger it is not a reliable filter.  The attack tag is, since
    it encodes exactly the hyperparameters that distinguish one variant of an
    attack from another.
    """
    from mia.experiment import Experiment
    exp = Experiment.load(config_path)
    pairs, labels = [], {}
    for label in exp.attacks:
        attack = exp.build_attack(label)
        key = (attack.params().get("attack", attack.name), attack.tag())
        pairs.append(key)
        labels[key] = label
    return exp.dataset, pairs, labels


def apply_selection(df, pairs):
    keys = set(pairs)
    return df[[(a, t) in keys for a, t in zip(df.attack, df.tag)]]



def _order(values, preferred):
    known = [v for v in preferred if v in values]
    return known + sorted(v for v in values if v not in preferred)


def grid_text(summary: pd.DataFrame, dataset: str) -> str:
    attacks = _order(summary.row.unique(), ATTACK_ORDER)
    gens = _order(summary.generator.unique(), GENERATOR_ORDER)
    lines = [f"\n{'=' * 100}",
             f"  ATTACK x GENERATOR GRID -- {dataset}",
             f"  cells are mean over splits; +- is the standard deviation across splits",
             f"{'=' * 100}"]

    header = f"{'attack':<22}" + "".join(f"{GENERATOR_LABELS.get(g, g):>21}" for g in gens)
    for metric in METRICS:
        lines.append(f"\n-- {METRIC_LABELS[metric]} " + "-" * 60)
        lines.append(header)
        for a in attacks:
            row = f"{a:<22}"
            for g in gens:
                cell = summary[(summary.row == a) & (summary.generator == g)]
                if cell.empty or np.isnan(cell.iloc[0][f"{metric}_mean"]):
                    row += f"{'--':>21}"
                else:
                    m = cell.iloc[0][f"{metric}_mean"]
                    s = cell.iloc[0][f"{metric}_std"]
                    s = 0.0 if np.isnan(s) else s
                    row += f"{m:>14.3f}+-{s:<5.3f}"
            lines.append(row)
    return "\n".join(lines)


def grid_latex(summary: pd.DataFrame, dataset: str) -> str:
    attacks = _order(summary.row.unique(), ATTACK_ORDER)
    gens = _order(summary.generator.unique(), GENERATOR_ORDER)
    out = [r"% " + f"attack x generator grid, {dataset}",
           r"\begin{tabular}{l" + "cccc" * len(gens) + "}", r"\toprule"]
    out.append(" & " + " & ".join(
        rf"\multicolumn{{4}}{{c}}{{\textbf{{{GENERATOR_LABELS.get(g, g)}}}}}" for g in gens
    ) + r" \\")
    out.append("".join(rf"\cmidrule(lr){{{2 + 4 * i}-{5 + 4 * i}}}" for i in range(len(gens))))
    out.append(r"\textbf{Attack} & " +
               " & ".join(" & ".join(METRIC_LABELS[m] for m in METRICS) for _ in gens) + r" \\")
    out.append(r"\midrule")
    for a in attacks:
        cells = []
        for g in gens:
            row = summary[(summary.row == a) & (summary.generator == g)]
            for m in METRICS:
                v = row.iloc[0][f"{m}_mean"] if not row.empty else np.nan
                cells.append("--" if np.isnan(v) else f"{v:.3f}")
        out.append(rf"\textsc{{{a.replace('_', '-')}}} & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None,
                   help="experiment YAML whose attack variants define the rows")
    p.add_argument("--dataset", default=None)
    p.add_argument("--tag", default=None, help="restrict to one attack-config tag")
    p.add_argument("--format", default="text", choices=["text", "latex", "csv"])
    p.add_argument("--out", default=None, help="directory to write the table into")
    args = p.parse_args()

    pairs = labels = None
    if args.config:
        cfg_dataset, pairs, labels = selection_from_config(args.config)
        args.dataset = args.dataset or cfg_dataset

    datasets = [args.dataset] if args.dataset else sorted(load().dataset.unique())
    for ds in datasets:
        df = load(ds, args.tag)
        if pairs:
            df = apply_selection(df, pairs)
            if df.empty:
                print(f"No runs yet for the variants in {args.config} on {ds}.")
                continue
        summary = summarise(df, labels)
        if args.format == "csv":
            rendered = summary.to_csv(index=False)
        elif args.format == "latex":
            rendered = grid_latex(summary, ds)
        else:
            rendered = grid_text(summary, ds)
        print(rendered)

        if args.out:
            d = Path(args.out)
            d.mkdir(parents=True, exist_ok=True)
            ext = {"text": "txt", "latex": "tex", "csv": "csv"}[args.format]
            (d / f"grid_{ds}.{ext}").write_text(rendered)
            summary.to_csv(d / f"grid_{ds}_summary.csv", index=False)
            print(f"\n  wrote {d / f'grid_{ds}.{ext}'}")


if __name__ == "__main__":
    main()
