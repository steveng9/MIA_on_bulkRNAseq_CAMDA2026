#!/usr/bin/env python
"""Draft figures for the paper, from the recorded runs.

    python scripts/paper_figures.py              # every figure
    python scripts/paper_figures.py --cells      # print the numbers behind them

Writes paper/Fig/<name>.pdf (for LaTeX) and .png (for review), and the numbers
each figure shows to paper/Fig/data/<name>.csv.  Figure 1 (the overview
diagram) is TikZ: paper/Fig/overview.tex.

  mechanism_heatmap  AUC of every attack against every generator, one cohort.
                     The pattern to see is the block diagonal: each attack is
                     strong against the family it was designed for.
  exposure           per generator and cohort, the strongest of our attacks
                     against the strongest generic attack and RedSigma, as AUC
                     and as TPR at 1% FPR.

Cells are fixed attack settings, not the best of every variant ever run, and
each cell is the mean over the splits recorded for it (the latest run of each
split, so superseded runs drop out).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mia import runs as R  # noqa: E402

OUT = ROOT / "paper" / "Fig"

COHORTS = {"BRCA": "TCGA-BRCA", "COMBINED": "TCGA-COMBINED", "GTEX": "GTEx"}

TABSYN = ("tabsyn@class_freq=train,diff_schedule=steps,latent_scale=std,"
          "preprocess=clip:0.001:0.999+standard")
PGM = "pgm@binning=dp_quantile,edge_estimator=threshold,n_bins=16"
GENERATORS = {  # target name -> label, in the order of the figures
    "mvn": "MVN",
    "cvae": "CVAE",
    "nd": "NoisyDiffusion",
    TABSYN: "TabSyn",
    "tabpfn": "TabPFN",
    "pgg": "DP-PGM (CAMDA 2026)",
    PGM: "DP-PGM (corrected)",
}
DP = {"pgg", PGM}

# Our attacks, as (attack, tag) per cohort.  MeLoMIA uses K=30 shadows on BRCA
# and K=20 elsewhere, all with per-record calibration and black box.  MahalaMIA
# is the ridge (1e-4) with the reference set; it was only run with the
# pseudo-inverse on TabSyn and TabPFN, where it is never the strongest.
OURS = {
    "MahalaMIA": [("mahalamia", "aux_ridge0.0001"), ("mahalamia", "aux")],
    "MeLoMIA-CVAE": [("melomia_cvae", "k30_n50_prc"), ("melomia_cvae", "k20_n50_prc")],
    "MeLoMIA-ND": [("melomia_nd", "k30_n600_prc"), ("melomia_nd", "k20_n600_prc")],
    "MeLoMIA-TabSyn": [("melomia_tabsyn", "k30_n200_prc"), ("melomia_tabsyn", "k20_n200_prc")],
    "MAMA-MIA": [("mamamia_v2", "cl-public_ed-auto_cc")],
}
FAMILY = {"MahalaMIA": "MahalaMIA", "MeLoMIA-CVAE": "MeLoMIA", "MeLoMIA-ND": "MeLoMIA",
          "MeLoMIA-TabSyn": "MeLoMIA", "MAMA-MIA": "MAMA-MIA"}
REDSIGMA = [("redsigma", "auto")]  # rule chosen by generator, as submitted
GENERIC = [("generic", t) for t in
           ("mc", "gan_leaks", "conf_lr", "conf_rf", "logan_d1", "gan_leaks_cal", "domias_kde")]
GENERIC_LABELS = {"mc": "MC", "gan_leaks": "GAN-leaks", "conf_lr": "Conf-LR",
                  "conf_rf": "Conf-RF", "logan_d1": "LOGAN", "gan_leaks_cal": "GAN-leaks cal.",
                  "domias_kde": "DOMIAS"}
METRICS = ["auc", "tpr_at_fpr_0.01"]


def load_runs() -> pd.DataFrame:
    df = R.load_index()
    df = df[(df.status == "ok") & df.generator.isin(GENERATORS) & df.dataset.isin(COHORTS)]
    for m in METRICS:
        df[m] = pd.to_numeric(df[m], errors="coerce")
    df["tag"] = df["tag"].fillna("")
    return (df.sort_values("timestamp")
              .drop_duplicates(["dataset", "attack", "tag", "generator", "split"], keep="last"))


def cell(df, dataset, generator, choices):
    """Mean over splits of the first (attack, tag) in `choices` that was run."""
    for attack, tag in choices:
        d = df[(df.dataset == dataset) & (df.generator == generator)
               & (df.attack == attack) & (df.tag == tag)]
        if len(d):
            out = {m: d[m].mean() for m in METRICS}
            out.update(n_splits=d.split.nunique(), attack=attack, tag=tag,
                       run_ids=";".join(d.sort_values("split").run_id))
            return out
    return None


def cells(df) -> pd.DataFrame:
    rows = []
    for ds in COHORTS:
        for g in GENERATORS:
            for name, choices in OURS.items():
                c = cell(df, ds, g, choices)
                if c:
                    rows.append(dict(dataset=ds, generator=g, row=name, group="ours", **c))
            c = cell(df, ds, g, REDSIGMA)
            if c:
                rows.append(dict(dataset=ds, generator=g, row="RedSigma", group="redsigma", **c))
            for choice in GENERIC:
                c = cell(df, ds, g, [choice])
                if c:
                    rows.append(dict(dataset=ds, generator=g, row=GENERIC_LABELS[choice[1]],
                                     group="generic", **c))
    return pd.DataFrame(rows)


def best(c: pd.DataFrame, group: str, metric: str) -> pd.DataFrame:
    """Per cohort and generator, the strongest row of a group under one metric."""
    d = c[c.group == group]
    i = d.groupby(["dataset", "generator"])[metric].idxmax()
    return d.loc[i, ["dataset", "generator", "row", metric, "n_splits"]]


# ---- Style --------------------------------------------------------------------
# One sans face (Arial metrics, as journals ask for; a TrueType face embeds cleanly), 6.5-7.5 pt at final
# size, recessive axes, and colour kept for identity: the three attack families.
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, NA_FILL = "#e1e0d9", "#c3c2b7", "#f0efec"
FAMILY_COLOR = {"MahalaMIA": "#2a78d6", "MeLoMIA": "#eb6834", "MAMA-MIA": "#1baf7a"}
FAMILY_MARKER = {"MahalaMIA": "o", "MeLoMIA": "s", "MAMA-MIA": "^"}
# Sequential blue, chance (AUC 0.5) near white, AUC 1.0 darkest.
SEQ = ["#f7f9fc", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SHORT = {"mvn": "MVN", "cvae": "CVAE", "nd": "Noisy-\nDiffusion", TABSYN: "TabSyn",
         "tabpfn": "TabPFN", "pgg": "DP-PGM\n(CAMDA 2026)", PGM: "DP-PGM\n(corrected)"}


def style():
    import matplotlib as mpl
    mpl.use("Agg")
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 7, "legend.fontsize": 6.5,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK2,
        "xtick.color": INK2, "ytick.color": INK, "xtick.major.width": 0.6,
        "ytick.major.size": 0, "xtick.major.size": 2.5,
        "pdf.fonttype": 42, "savefig.dpi": 300, "figure.dpi": 150,
    })


def save(fig, name: str, data: pd.DataFrame) -> None:
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02)
    data.to_csv(OUT / "data" / f"{name}.csv", index=False)
    print(f"wrote paper/Fig/{name}.pdf, .png, data/{name}.csv")


# ---- Figure: attack x generator heatmap -----------------------------------------
HEAT_ROWS = ["MahalaMIA", "MeLoMIA-CVAE", "MeLoMIA-ND", "MeLoMIA-TabSyn", "MAMA-MIA",
             None, "RedSigma", "Best generic attack"]


def heat_table(c: pd.DataFrame, dataset: str) -> pd.DataFrame:
    d = c[c.dataset == dataset]
    gen = d[d.group == "generic"].groupby("generator")["auc"].max().rename("Best generic attack")
    t = d[d.group != "generic"].pivot_table(index="row", columns="generator", values="auc")
    t = pd.concat([t, gen.to_frame().T])
    return t.reindex(index=[r for r in HEAT_ROWS if r], columns=list(GENERATORS))


def fig_mechanism_heatmap(c: pd.DataFrame, cohorts=("BRCA", "COMBINED"),
                          name="mechanism_heatmap") -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.patches import Rectangle

    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    norm = Normalize(0.5, 1.0, clip=True)
    # rows and columns with no run in any of the cohorts shown are left out
    tabs = {ds: heat_table(c, ds) for ds in cohorts}
    have = pd.concat(tabs.values()).notna().groupby(level=0).any()
    rows = [r for r in HEAT_ROWS if r is None or have.loc[r].any()]
    gens = [g for g in GENERATORS if have[g].any()]
    split_at = sum(g not in DP for g in gens)
    fig, axes = plt.subplots(len(cohorts), 1,
                             figsize=(7.1 * (len(gens) + 2) / 9, 0.2 * len(rows) * len(cohorts) + 0.35),
                             gridspec_kw=dict(hspace=0.32))
    axes = np.atleast_1d(axes)
    out = []
    for ax, ds in zip(axes, cohorts):
        t = tabs[ds]
        best_ours = t.loc[[r for r in OURS if r in rows]].fillna(0).idxmax()
        for i, r in enumerate(rows):
            if r is None:
                continue
            for j, g in enumerate(gens):
                v = t.loc[r, g]
                y = len(rows) - 1 - i
                if np.isnan(v):
                    ax.add_patch(Rectangle((j, y), 1, 1, facecolor="white", edgecolor=GRID,
                                           hatch="//////", linewidth=0))
                    ax.text(j + 0.5, y + 0.5, "not run", ha="center", va="center",
                            fontsize=5.5, color=MUTED)
                    continue
                rgba = cmap(norm(v))
                ax.add_patch(Rectangle((j, y), 1, 1, facecolor=rgba, edgecolor="white",
                                       linewidth=1.2))
                lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                bold = r == best_ours[g]
                ax.text(j + 0.5, y + 0.5, f"{v:.2f}", ha="center", va="center",
                        fontsize=6.8, color="white" if lum < 0.5 else INK,
                        fontweight="bold" if bold else "normal")
                out.append(dict(dataset=ds, attack=r, generator=GENERATORS[g], auc=round(v, 4)))
        ax.set_xlim(0, len(gens))
        ax.set_ylim(0, len(rows))
        ax.set_yticks([len(rows) - 1 - i + 0.5 for i, r in enumerate(rows) if r])
        ax.set_yticklabels([r for r in rows if r])
        ax.set_xticks(np.arange(len(gens)) + 0.5)
        ax.set_xticklabels([SHORT[g] for g in gens], fontsize=6.8, color=INK)
        ax.xaxis.tick_top()
        ax.tick_params(axis="x", length=0, pad=2)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(COHORTS[ds], loc="left", fontweight="bold", color=INK, pad=20 if ax is axes[0] else 4)
        if ax is not axes[0]:
            ax.set_xticklabels([])
        # DP columns separated from the rest
        ax.axvline(split_at, color=INK2, lw=0.6, ymin=0, ymax=1)
    cax = fig.add_axes([0.93, 0.25, 0.012, 0.5])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_ticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    cb.set_ticklabels(["0.5\nchance", "0.6", "0.7", "0.8", "0.9", "1.0"])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6, length=2, color=AXIS)
    cb.set_label("AUC", fontsize=6.5, color=INK2, labelpad=2)
    fig.subplots_adjust(left=0.14, right=0.91)
    save(fig, name, pd.DataFrame(out))
    plt.close(fig)


# ---- Figure: exposure of each generator in each cohort --------------------------
def exposure_table(c: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in METRICS:
        for group in ("ours", "generic", "redsigma"):
            b = best(c, group, m).rename(columns={m: "value"})
            rows.append(b.assign(metric=m, group=group))
    t = pd.concat(rows)
    t["family"] = t.row.map(FAMILY)
    return t


def fig_exposure(c: pd.DataFrame, name="exposure") -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    t = exposure_table(c)
    gens = list(GENERATORS)
    ypos = {g: len(gens) - 1 - i + (0.0 if g in DP else 0.5) for i, g in enumerate(gens)}
    cohorts = list(COHORTS)
    fig, axes = plt.subplots(2, len(cohorts), figsize=(7.1, 4.3), sharey=True,
                             gridspec_kw=dict(hspace=0.42, wspace=0.08))
    for r, m in enumerate(METRICS):
        for k, ds in enumerate(cohorts):
            ax = axes[r, k]
            d = t[(t.dataset == ds) & (t.metric == m)]
            log = m != "auc"
            chance = 0.5 if m == "auc" else 0.01
            lo, hi = (0.45, 1.02) if m == "auc" else (0.004, 1.15)
            ax.axvline(chance, color=MUTED, lw=0.8, ls=(0, (2, 2)), zorder=1)
            for g in gens:
                y = ypos[g]
                o = d[(d.generator == g) & (d.group == "ours")]
                if o.empty:
                    ax.text((lo * hi) ** 0.5 if log else (lo + hi) / 2, y, "not run",
                            ha="center", va="center", fontsize=6, color=MUTED)
                    continue
                ov = max(o.value.iloc[0], lo * 1.05 if log else lo)
                gv = d[(d.generator == g) & (d.group == "generic")].value
                rv = d[(d.generator == g) & (d.group == "redsigma")].value
                if len(gv):
                    gv = max(gv.iloc[0], lo * 1.05 if log else lo)
                    ax.plot([gv, ov], [y, y], color=AXIS, lw=2.2, solid_capstyle="round", zorder=2)
                    ax.plot(gv, y, "o", ms=5, mfc="white", mec=MUTED, mew=1.0, zorder=3)
                if len(rv):
                    rv = max(rv.iloc[0], lo * 1.05 if log else lo)
                    ax.plot(rv, y, "D", ms=5.5, mfc="none", mec=INK2, mew=0.9, zorder=4)
                fam = o.family.iloc[0]
                ax.plot(ov, y, FAMILY_MARKER[fam], ms=6, mfc=FAMILY_COLOR[fam], mec="white",
                        mew=0.7, zorder=5)
                n = o.n_splits.iloc[0]
                if n < 5:
                    ax.text(hi, y, f"{n} split{'s' if n > 1 else ''}", ha="right",
                            va="center", fontsize=5.5, color=MUTED)
            ax.set_xlim(lo, hi)
            if log:
                ax.set_xscale("log")
                ax.set_xticks([0.01, 0.03, 0.1, 0.3, 1.0])
                ax.set_xticklabels(["0.01", "0.03", "0.1", "0.3", "1"])
                ax.minorticks_off()
            else:
                ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
                ax.set_xticklabels(["0.5", "0.6", "0.7", "0.8", "0.9", "1.0"])
            ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
            ax.set_axisbelow(True)
            for s in ("top", "right", "left"):
                ax.spines[s].set_visible(False)
            ax.axhline(1.75, color=AXIS, lw=0.5)  # DP generators below
            ax.set_ylim(-0.6, len(gens) + 0.1)
            if r == 0:
                ax.set_title(COHORTS[ds], fontweight="bold", color=INK, loc="left")
            ax.set_xlabel("AUC" if m == "auc" else "TPR at 1% FPR (log scale)")
            if k == 0:
                ax.set_yticks([ypos[g] for g in gens])
                ax.set_yticklabels([GENERATORS[g] for g in gens])
            ax.text(chance, len(gens) + 0.05, "chance", ha="center", va="bottom",
                    fontsize=5.5, color=MUTED)
    ours = [Line2D([], [], ls="", marker=FAMILY_MARKER[f], ms=6, mfc=FAMILY_COLOR[f], mec="white",
                   label=f) for f in FAMILY_COLOR]
    others = [Line2D([], [], ls="", marker="D", ms=5.5, mfc="none", mec=INK2, label="RedSigma"),
              Line2D([], [], ls="", marker="o", ms=5, mfc="white", mec=MUTED,
                     label="strongest generic attack (CAMDA 2025 kit)")]
    kw = dict(frameon=False, handletextpad=0.3, columnspacing=1.2, title_fontsize=6.5,
              alignment="left", loc="lower left")
    leg = fig.legend(handles=ours, ncol=3, title="Strongest of our attacks, by family",
                     bbox_to_anchor=(0.12, 0.965), **kw)
    fig.add_artist(leg)
    fig.legend(handles=others, ncol=2, title="Comparison", bbox_to_anchor=(0.50, 0.965), **kw)
    save(fig, name, t.drop(columns=["family"]).assign(generator=t.generator.map(GENERATORS)))
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cells", action="store_true", help="print the cells and stop")
    args = ap.parse_args()

    c = cells(load_runs())
    if not args.cells:
        style()
        fig_mechanism_heatmap(c)
        fig_mechanism_heatmap(c, cohorts=("GTEX",), name="mechanism_heatmap_gtex")
        fig_exposure(c)
        return
    if args.cells:
        pd.set_option("display.width", 220)
        p = c.assign(g=c.generator.map(GENERATORS)).pivot_table(
            index=["dataset", "group", "row"], columns="g", values="auc", sort=False)
        print(p[[v for v in GENERATORS.values() if v in p.columns]].round(3).to_string())
        n = c[c.n_splits < 5]
        print("\nCells with fewer than five splits:")
        print(n[["dataset", "generator", "row", "n_splits"]].to_string(index=False))
        return


if __name__ == "__main__":
    main()
