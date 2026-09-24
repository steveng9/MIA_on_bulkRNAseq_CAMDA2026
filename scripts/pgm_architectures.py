#!/usr/bin/env python
"""Every DP-PGM table-selection method we have run, in one table.

    python scripts/pgm_architectures.py

For Steven (2026-09-24): one place to compare the architectures apples to
apples.  Reads every DP-PGM sweep CSV, names each row's architecture in plain
words, joins MAMA-MIA v2 on the same targets, and writes

    results/pgm_architectures.csv   one row per target (dataset, eps, split, config)
    results/PGM_ARCHITECTURES.md    the catalogue, what has been run under which
                                    conditions, and the head-to-head tables

Head-to-head tables use only the conditions every architecture shares
(dp_quantile 16 bins, split 1, eps 10 and 1000); the coverage table shows where
that is not yet true.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

SOURCES = ["pgm_structure_sweep", "pgm_forest_sweep", "pgm_pqrs_retry",
           "pgm_pqrs_maxdeg", "pgm_pqrs_maxdeg16", "pgm_hairy_star", "pgm_hairy_pairs", "pgm_baselines"]

# (key, plain name, tables measured, how chosen, DP end to end)
CATALOGUE = [
    ("one_way", "1-way only",
     "a 1-way table per gene + the label marginal (genes independent, no label link)",
     "fixed", "yes"),
    ("star", "Star (last year's winner)",
     "978 gene 1-way + 978 (gene, label)", "fixed by config", "yes"),
    ("label_star", "Label-only star",
     "978 (gene, label), no 1-way", "fixed", "yes"),
    ("tree", "Star + DP gene tree",
     "star + 977 (gene, gene) spanning tree", "MST-style exponential mechanism", "yes"),
    ("tree_label", "Star + DP gene-gene-label tree",
     "star + 977 (gene, gene, label) triangles", "MST-style exponential mechanism", "yes"),
    ("forest", "k/l forest",
     "k (gene, label) + l (gene, gene) + 1-way for uncovered genes",
     "DP top-k + DP truncated Kruskal", "yes"),
    ("hairy", "Star with hairs",
     "l (gene, gene) + one (gene, label) per component [+ every 1-way]: a spanning tree over genes + label",
     "DP truncated Kruskal + DP hub per component", "yes"),
    ("pqrs", "PQRS (Steven's k-way)",
     "star + n2 (gene, gene) + n3 3-way + n4 4-way", "Spearman ranking, NOT DP", "no (selection)"),
    ("mst", "Classic MST (label as a free node)",
     "spanning tree over genes + label, label edges compete with gene edges", "MST", "yes"),
]


def architecture(r) -> tuple[str, str]:
    """(catalogue key, knobs) for one sweep row."""
    s = r.get("structure", "hierarchical")
    if s == "forest":
        k, l = int(r["k_label"]), int(r["l_pairs"])
        if k >= 978 and l == 0:
            return "label_star", "forest k978 l0"
        if k == 0 and l == 0:
            return "one_way", ""
        return "forest", f"k={k} l={l}"
    if s == "hairy_star":
        l = int(r["l_pairs"])
        one = str(r.get("with_1way")) in ("True", "true", "1", "1.0")
        if l == 0 and not one:
            return "label_star", "hairy l0"
        mc = r.get("max_component")
        pairs = "" if mc is None or pd.isna(mc) else f" comp<={int(mc)}"
        return "hairy", f"l={l}" + pairs + (" +1way" if one else "")
    if s in ("tree", "tree_label"):
        return s, ""
    cfg = str(r["config"])
    if "pqrs" in cfg:
        md = r.get("max_degree")
        md = "" if pd.isna(md) else f" max_degree={int(md)}"
        return "pqrs", cfg.split("/")[-1].replace("pqrs_", "(").replace("_md", ")").split(")")[0].replace("_", ",") + ")" + md
    return "star", ""


def load() -> pd.DataFrame:
    frames = []
    for name in SOURCES:
        p = RES / f"{name}.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p)
        d["source"] = name
        frames.append(d)
    d = pd.concat(frames, ignore_index=True, sort=False)
    d = d[d.binning.isin(["dp_quantile", "dp_uniform", "uniform", "quantile"])]
    d = d[~((d.binning.str.startswith("dp_")) & (d.edge_estimator != "threshold"))]
    for c in ("structure", "k_label", "l_pairs", "with_1way", "max_degree", "max_component"):
        if c not in d:
            d[c] = np.nan
    d["structure"] = d.structure.fillna("hierarchical")
    arch = d.apply(architecture, axis=1)
    d["arch"] = [a for a, _ in arch]
    d["knobs"] = [k for _, k in arch]
    mcols = [c for c in d.columns if c.startswith("auc_mahalamia") and not c.startswith("auc_t1")]
    d["mahalamia_best"] = d[mcols].max(axis=1)
    return d


def v2_columns(d: pd.DataFrame) -> pd.DataFrame:
    """Fixed MAMA-MIA v2 paths (class-centred, all tables), no max over paths.

    black box: the star's tables are public, so `public`; a data-dependent
    selection is guessed by held-out shadows, `shadow`.  Edges `recovered`
    (release quantiles) everywhere, and `grid` where it has been run.
    white box: the true tables on the true edges.
    """
    p = RES / "mamamia_v2.csv"
    if not p.exists():
        return d
    v = pd.read_csv(p)
    v = v[v.arm == "all_cc"]
    key = ["target", "split", "fingerprint"]
    v = v.drop_duplicates(key + ["cliques", "edges"], keep="last")
    piv = v.pivot_table(index=key, columns=["cliques", "edges"], values="auc", aggfunc="first")
    piv.columns = [f"{a}/{b}" for a, b in piv.columns]
    piv = piv.reset_index()
    d = d.merge(piv, on=key, how="left")
    fixed = d.structure.isin(["hierarchical"])

    def pick(fixed_col, dep_col):
        a = d.get(fixed_col, pd.Series(np.nan, index=d.index))
        b = d.get(dep_col, pd.Series(np.nan, index=d.index))
        return np.where(fixed, a, b)

    d["v2_blackbox"] = pick("public/recovered", "shadow/recovered")
    d["v2_blackbox_grid"] = pick("public/grid", "shadow/grid")
    d["v2_whitebox"] = pick("public/known", "true/known")
    return d


COLS = ["utility_ratio", "corr_mae", "wasserstein_mean", "discriminator_auc",
        "mahalamia_best", "v2_blackbox", "v2_blackbox_grid", "v2_whitebox"]
HEAD = {"utility_ratio": "utility", "corr_mae": "corr MAE", "wasserstein_mean": "W1",
        "discriminator_auc": "discrim.", "mahalamia_best": "MahalaMIA",
        "v2_blackbox": "v2 black", "v2_blackbox_grid": "v2 black+grid",
        "v2_whitebox": "v2 white"}


FOREST_SHOWN = {f"k={k} l={l}" for k in (100, 500) for l in (0, 100, 400)}
ORDER = {n: i for i, (_, n, *_r) in enumerate(CATALOGUE)}


def knob_key(k: str) -> tuple:
    """Sort knobs numerically: 'k=100 l=20' before 'k=100 l=100'."""
    import re
    return tuple(float(x) for x in re.findall(r"=(\d+)", str(k))) + (str(k),)


def md_table(t: pd.DataFrame) -> str:
    cols = list(t.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in t.iterrows():
        out.append("| " + " | ".join("" if (isinstance(x, float) and np.isnan(x))
                                     else (f"{x:.3f}" if isinstance(x, float) else str(x))
                                     for x in r) + " |")
    return "\n".join(out)


def main():
    d = v2_columns(load())
    name = dict((k, n) for k, n, *_ in CATALOGUE)
    d["architecture"] = d.arch.map(name)
    keep = ["architecture", "knobs", "dataset", "epsilon", "split", "binning", "n_bins",
            "config", "source", "target", "fingerprint", *COLS]
    d = d[[c for c in keep if c in d.columns]].sort_values(
        ["architecture", "dataset", "epsilon", "binning", "n_bins", "knobs"])
    d.to_csv(RES / "pgm_architectures.csv", index=False)

    L = ["# DP-PGM architectures: everything in one place",
         "",
         "Generated by `scripts/pgm_architectures.py` from every DP-PGM sweep CSV; "
         "one row per target in `results/pgm_architectures.csv`.  Rerun it after any "
         "sweep.  Only DP-edge targets built with the fixed edge estimator are "
         "included (see `results/BROKEN.md`).",
         "",
         "## The architectures",
         "",
         "| architecture | tables measured | how the tables are chosen | DP end to end | targets |",
         "|---|---|---|---|---|"]
    n_t = d.groupby("architecture").size()
    for k, n, tabs, how, dp in CATALOGUE:
        L.append(f"| {n} | {tabs} | {how} | {dp} | {int(n_t.get(n, 0)) or '**not run**'} |")
    L += ["",
          "All run in joint mode (one model, label as a node) under rho-zCDP, "
          "add/remove neighbours, 0.1 of rho on DP bin edges and (where tables are "
          "chosen from data) 0.3 on selection.",
          "",
          "## What has been run where (number of configs)",
          ""]
    cov = d.assign(cond=lambda x: x.binning + x.n_bins.astype(int).astype(str) + " eps"
                   + x.epsilon.astype(float).map("{:g}".format) + " s" + x.split.astype(str))
    cov = cov[cov.split == 1]
    c = cov.pivot_table(index="architecture", columns="cond", values="target",
                        aggfunc=lambda s: s.nunique() // 1).fillna(0).astype(int)
    c = c[[x for x in c.columns if x.startswith("dp_quantile")]]
    L.append(md_table(c.reset_index()))
    L += ["", "Counts are configurations, each built for both cohorts; split 1 only.  "
          "Head-to-head tables show the k/l forest at k in {100, 500} x l in {0, 100, 400}; "
          "all 35 forest configurations are in the CSV.", ""]

    for eps in (10.0, 1000.0):
        for nb in (16, 32):
            s = d[(d.binning == "dp_quantile") & (d.n_bins == nb) & (d.split == 1)
                  & (d.epsilon.astype(float) == eps)]
            if s.empty:
                continue
            L += [f"## Head to head: dp_quantile {nb} bins, eps={eps:g}, split 1", ""]
            for ds in ("BRCA", "COMBINED"):
                t = s[s.dataset == ds]
                if t.empty:
                    continue
                # The full forest grid is in the CSV; here a representative slice.
                t = t[(t.architecture != name["forest"]) | t.knobs.isin(FOREST_SHOWN)]
                t = t.assign(_o=t.architecture.map(ORDER), _k=t.knobs.map(knob_key))
                t = t.sort_values(["_o", "_k"])
                t = t[["architecture", "knobs", *COLS]].rename(columns=HEAD)
                L += [f"**{ds}**", "", md_table(t), ""]
    L += ["## Reading the columns", "",
          "- **utility**: TSTR macro-F1 over train-on-real (1 = as good as real).",
          "- **corr MAE**, **W1**: gene-gene correlation error and mean per-gene "
          "Wasserstein distance (lower is better).",
          "- **discrim.**: AUC of a real-vs-synthetic classifier (0.5 = indistinguishable).",
          "- **MahalaMIA**: best of its covariance variants (an upper envelope).",
          "- **v2 black / v2 black+grid / v2 white**: MAMA-MIA v2, class-centred, one "
          "fixed path each (no maximum over paths).  Black box: the star's tables "
          "are public; data-dependent tables are guessed by held-out shadows; edges "
          "are release quantiles (black) or the grid estimator (black+grid).  White "
          "box: the true tables on the true edges.  DP bound at eps=10: 0.89.",
          ""]
    (RES / "PGM_ARCHITECTURES.md").write_text("\n".join(L))
    print(f"wrote {RES / 'pgm_architectures.csv'} ({len(d)} rows) and {RES / 'PGM_ARCHITECTURES.md'}")


if __name__ == "__main__":
    sys.exit(main())
