#!/usr/bin/env python
"""Draft figures for the paper, from the recorded runs.

    python scripts/paper_figures.py              # every figure
    python scripts/paper_figures.py --cells      # print the numbers behind them

Writes paper/Fig/<name>.pdf (for LaTeX) and .png (for review), and the numbers
each figure shows to paper/Fig/data/<name>.csv.  Figure 1 (the overview
diagram) is TikZ: paper/Fig/overview.tex.

  attack_grid        AUC of our three attacks, RedSigma and the best generic
                     attack against every generator, one panel per cohort.
                     MeLoMIA is one row: the backend built for the generator's
                     family (the diffusion backend where there is none).
  attack_grid_backends
                     the same with MeLoMIA's three backends as separate rows
                     (appendix): how each backend does off its own family.
  exposure           per generator and cohort, the strongest of our attacks
                     against the strongest generic attack and RedSigma, as AUC
                     and as TPR at 1% FPR.
  cohort_size        MahalaMIA against MVN and CVAE on TCGA-COMBINED with the
                     training set subsampled, as AUC and TPR at 1% FPR: the
                     exposure falls once there are more donors than genes.

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
    "pgg": "DP-PGM (challenge)",
    PGM: "DP-PGM (corrected)",
}
DP = {"pgg", PGM}

# Our attacks, as (attack, tag) per cohort.  MeLoMIA uses K=30 shadows on BRCA
# and K=20 elsewhere, all with per-record calibration and black box.  MahalaMIA
# scores a candidate against its own class, with the class covariance averaged
# with the pooled one (ridge 1e-4), and uses the reference set where one exists.
OURS = {
    "MahalaMIA": [("mahalamia", "aux_ridge0.0001_shrink0.5")],
    "MeLoMIA-CVAE": [("melomia_cvae", "k30_n50_prc"), ("melomia_cvae", "k20_n50_prc")],
    "MeLoMIA-ND": [("melomia_nd", "k30_n600_prc"), ("melomia_nd", "k20_n600_prc")],
    "MeLoMIA-TabSyn": [("melomia_tabsyn", "k30_n200_prc"), ("melomia_tabsyn", "k20_n200_prc")],
    "MAMA-MIA": [("mamamia_v2", "cl-public_ed-auto_cc")],
}
FAMILY = {"MahalaMIA": "MahalaMIA", "MeLoMIA": "MeLoMIA", "MeLoMIA-CVAE": "MeLoMIA",
          "MeLoMIA-ND": "MeLoMIA", "MeLoMIA-TabSyn": "MeLoMIA", "MAMA-MIA": "MAMA-MIA"}
# In the main text MeLoMIA is one attack: the backend built for the generator's
# family, and the diffusion backend for generators that have none.
MELOMIA_BACKEND = {"cvae": "MeLoMIA-CVAE", "nd": "MeLoMIA-ND", TABSYN: "MeLoMIA-TabSyn"}
MELOMIA_DEFAULT = "MeLoMIA-ND"
MAIN_OURS = ["MahalaMIA", "MeLoMIA", "MAMA-MIA"]
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
            out.update(auc_sd=d["auc"].std(ddof=1) if len(d) > 1 else np.nan,
                       n_splits=d.split.nunique(), attack=attack, tag=tag,
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
            mine = {r["row"]: r for r in rows if r["dataset"] == ds and r["generator"] == g
                    and r["group"] == "ours"}
            for name in MAIN_OURS:
                src = MELOMIA_BACKEND.get(g, MELOMIA_DEFAULT) if name == "MeLoMIA" else name
                if src in mine:
                    rows.append(dict(mine[src], row=name, group="main", backend=src))
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
         "tabpfn": "TabPFN", "pgg": "DP-PGM\nchallenge", PGM: "DP-PGM\ncorrected"}


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
             None, "RedSigma", "Best generic attack"]          # appendix: one row per backend
MAIN_ROWS = MAIN_OURS + [None, "RedSigma", "Best generic attack"]  # main text


def heat_table(c: pd.DataFrame, dataset: str, rows=None) -> pd.DataFrame:
    rows = rows or HEAT_ROWS
    d = c[c.dataset == dataset]
    gen = d[d.group == "generic"].groupby("generator")["auc"].max().rename("Best generic attack")
    ours = "main" if "MeLoMIA" in rows else "ours"
    t = d[d.group.isin([ours, "redsigma"])].pivot_table(index="row", columns="generator", values="auc")
    t = pd.concat([t, gen.to_frame().T])
    return t.reindex(index=[r for r in rows if r], columns=list(GENERATORS))


def fig_attack_grid(c: pd.DataFrame, cohorts=("BRCA", "COMBINED", "GTEX"), rows=None,
                    name="attack_grid", width=3.5) -> None:
    """One panel per cohort, stacked; a generator that was not built for a cohort
    (TabSyn and TabPFN on GTEx) and a backend that was not run leave an empty cell."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.patches import Rectangle

    rows = rows or MAIN_ROWS
    ours = [r for r in rows if r in FAMILY]
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    norm = Normalize(0.5, 1.0, clip=True)
    tabs = {ds: heat_table(c, ds, rows) for ds in cohorts}
    gens = list(GENERATORS)
    split_at = sum(g not in DP for g in gens)
    small = width < 5
    row_h = 0.185
    GAP = 0.3                                   # height of the blank between our attacks and the rest
    height = sum(GAP if r is None else 1.0 for r in rows)
    ypos, y = {}, height
    for i, r in enumerate(rows):
        y -= GAP if r is None else 1.0
        ypos[i] = y
    fig, axes = plt.subplots(len(cohorts), 1,
                             figsize=(width, row_h * height * len(cohorts) + 0.22 * len(cohorts) + 0.42),
                             gridspec_kw=dict(hspace=0.2))
    axes = np.atleast_1d(axes)
    out = []
    for ax, ds in zip(axes, cohorts):
        t = tabs[ds]
        best_ours = t.loc[ours].fillna(0).idxmax()
        for i, r in enumerate(rows):
            if r is None:
                continue
            for j, g in enumerate(gens):
                v = t.loc[r, g]
                y = ypos[i]
                if np.isnan(v):
                    ax.add_patch(Rectangle((j, y), 1, 1, facecolor=NA_FILL, edgecolor="white",
                                           linewidth=1.2))
                    continue
                rgba = cmap(norm(v))
                ax.add_patch(Rectangle((j, y), 1, 1, facecolor=rgba, edgecolor="white",
                                       linewidth=1.2))
                lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                bold = r == best_ours[g] and v >= 0.55
                ax.text(j + 0.5, y + 0.47, f"{v:.2f}", ha="center", va="center",
                        fontsize=6.3 if small else 6.8, color="white" if lum < 0.5 else INK,
                        fontweight="bold" if bold else "normal")
                out.append(dict(dataset=ds, attack=r, generator=GENERATORS[g], auc=round(v, 4)))
        ax.set_xlim(0, len(gens))
        ax.set_ylim(0, height)
        ax.set_yticks([ypos[i] + 0.5 for i, r in enumerate(rows) if r])
        ax.set_yticklabels([r for r in rows if r], fontsize=6.5 if small else 7)
        ax.set_xticks(np.arange(len(gens)) + 0.5)
        ax.xaxis.tick_top()
        ax.tick_params(axis="x", length=0, pad=1.5)
        ax.tick_params(axis="y", pad=1.5)
        if ax is axes[0]:
            ax.set_xticklabels([SHORT[g] for g in gens], fontsize=5.6 if small else 6.8, color=INK,
                               linespacing=0.95)
        else:
            ax.set_xticklabels([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.text(-0.02, 1.03, COHORTS[ds], transform=ax.transAxes,
                ha="right", va="bottom", fontweight="bold", color=INK, fontsize=7)
        ax.axvline(split_at, color=INK2, lw=0.6, ymin=0, ymax=1)   # DP generators to the right
    cax = fig.add_axes([0.2, 0.045, 0.6, 0.012])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
    cb.set_ticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    cb.set_ticklabels(["0.5 (chance)", "0.6", "0.7", "0.8", "0.9", "1.0"])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=5.8, length=2, color=AXIS, pad=1.5)
    cb.set_label("AUC", fontsize=6.2, color=INK2, labelpad=1)
    fig.subplots_adjust(left=0.2 if small else 0.14, right=0.995, top=0.94, bottom=0.105)
    save(fig, name, pd.DataFrame(out))
    plt.close(fig)


# ---- Figure: exposure of each generator in each cohort --------------------------
def exposure_table(c: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in METRICS:
        for group in ("main", "generic", "redsigma"):
            b = best(c, group, m).rename(columns={m: "value"})
            rows.append(b.assign(metric=m, group="ours" if group == "main" else group))
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
                    ax.text((lo * hi) ** 0.5 if log else (lo + hi) / 2, y, "not built",
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


# ---- Figure: exposure against the number of training donors ---------------------
def cohort_size_table() -> pd.DataFrame:
    """MahalaMIA (the paper's default) on COMBINED with the training set subsampled to n."""
    df = R.load_index()
    df = df[(df.status == "ok") & (df.experiment.astype(str) == "cohort_size")
            & df.tag.astype(str).str.endswith("_shrink0.5")]
    for m in METRICS:
        df[m] = pd.to_numeric(df[m], errors="coerce")
    df = df.assign(n=df.tag.str.extract(r"^n(\d+)_")[0].astype(int))
    t = (df.groupby(["generator", "n"])
           .agg(auc=("auc", "mean"), tpr_at_fpr_0_01=("tpr_at_fpr_0.01", "mean"),
                trials=("split", "size"))
           .reset_index())
    return t.rename(columns={"tpr_at_fpr_0_01": "tpr_at_fpr_0.01"})


def fig_cohort_size(name="cohort_size") -> None:
    import matplotlib.pyplot as plt

    t = cohort_size_table()
    p = 978
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.75), sharex=True,
                             gridspec_kw=dict(wspace=0.38))
    styles = {"mvn": ("MVN", "o", "-"), "cvae": ("CVAE", "s", (0, (3, 1.5)))}
    for ax, m in zip(axes, METRICS):
        ax.axvline(p, color=AXIS, lw=0.8, zorder=1)
        for g, (label, marker, ls) in styles.items():
            d = t[t.generator == g].sort_values("n")
            ax.plot(d.n, d[m], ls=ls, color=FAMILY_COLOR["MahalaMIA"], lw=1.4, zorder=3)
            ax.plot(d.n, d[m], marker, ms=3.6, mfc=FAMILY_COLOR["MahalaMIA"], mec="white",
                    mew=0.5, ls="", zorder=4)
            last = d.iloc[-1]
            ax.text(last.n * 1.07, last[m], label, va="center", ha="left", fontsize=6,
                    color=INK2)
        ax.set_xscale("log")
        ax.set_xticks([500, 1000, 2000])
        ax.set_xticklabels(["500", "1,000", "2,000"])
        ax.minorticks_off()
        ax.set_xlim(430, 6000)
        if m == "auc":
            ax.set_ylim(0.5, 1.01)  # chance at the bottom, as for TPR
            ax.set_yticks([0.5, 0.75, 1.0])
            ax.set_ylabel("AUC")
        else:
            ax.set_ylim(0, 1.03)
            ax.set_yticks([0, 0.5, 1.0])
            ax.set_ylabel("TPR at 1% FPR")
        ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=2.5, width=0.6)
        ax.set_xlabel("training donors (log scale)")
        lo = ax.get_ylim()[0]
        ax.text(p * 0.96, lo + 0.02 * (ax.get_ylim()[1] - lo), "978 genes",
                rotation=90, ha="right", va="bottom", fontsize=5.5, color=MUTED)
    save(fig, name, t)
    plt.close(fig)


# ---- Figure: the corrected DP-PGM across epsilon ---------------------------------
def epsilon_table() -> pd.DataFrame:
    """Per cohort and epsilon: MAMA-MIA (the paper's setting), the DP bound and utility.

    The releases are those of scripts/pgm_eps_sweep.py at 16 bins (five splits);
    MAMA-MIA was run on them by scripts/queue_eps_curve_attack.sh, and epsilon = 10
    is the grid's corrected DP-PGM.
    """
    sweep = pd.read_csv(ROOT / "results" / "pgm_eps_sweep.csv")
    sweep = sweep[(sweep.binning == "dp_quantile") & (sweep.n_bins == 16)]
    fid = (sweep.groupby(["dataset", "epsilon"])
                .agg(utility=("utility_ratio", "mean"), w1=("wasserstein_mean", "mean"),
                     bound=("dp_bound_auc", "mean"), releases=("split", "nunique"))
                .reset_index())
    df = R.load_index()
    df = df[(df.status == "ok") & (df.attack == "mamamia_v2")
            & (df.tag.fillna("") == OURS["MAMA-MIA"][0][1])
            & df.generator.astype(str).str.startswith("pgm@binning=dp_quantile,edge_estimator=threshold,")
            & df.generator.astype(str).str.endswith(",n_bins=16")]
    eps = df.generator.str.extract(r"epsilon=([0-9.]+)")[0].astype(float)
    df = df.assign(epsilon=eps.where(eps.notna(), 10.0) if len(df) else eps)
    df = df[df.generator.str.contains("epsilon=") | (df.generator == PGM)]
    for m in METRICS:
        df[m] = pd.to_numeric(df[m], errors="coerce")
    df = df.sort_values("timestamp").drop_duplicates(["dataset", "generator", "split"], keep="last")
    att = (df.groupby(["dataset", "epsilon"])
             .agg(auc=("auc", "mean"), auc_sd=("auc", "std"),
                  tpr_at_fpr_0_01=("tpr_at_fpr_0.01", "mean"), splits=("split", "nunique"))
             .reset_index())
    return fid.merge(att, on=["dataset", "epsilon"], how="left").sort_values(["dataset", "epsilon"])


def fig_epsilon(name="epsilon_curve") -> None:
    import matplotlib.pyplot as plt

    t = epsilon_table()
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.7), sharey=True,
                             gridspec_kw=dict(wspace=0.10))
    green = FAMILY_COLOR["MAMA-MIA"]
    for ax, ds in zip(axes, ("BRCA", "COMBINED")):
        d = t[t.dataset == ds]
        ax.plot(d.epsilon, d.bound, color=MUTED, lw=0.9, zorder=2)
        ax.axhline(0.5, color=AXIS, lw=0.6, ls=(0, (2, 2)), zorder=1)
        ax.plot(d.epsilon, d.utility, color=INK2, lw=1.0, ls=(0, (3, 1.5)), zorder=3)
        ax.plot(d.epsilon, d.utility, "s", ms=2.8, mfc="white", mec=INK2, mew=0.7, zorder=4)
        ax.plot(d.epsilon, d.auc, color=green, lw=1.4, zorder=5)
        ax.plot(d.epsilon, d.auc, "^", ms=3.8, mfc=green, mec="white", mew=0.5, zorder=6)
        ax.set_xscale("log")
        ax.set_xticks([0.1, 1, 10, 100, 1000])
        ax.set_xticklabels(["0.1", "1", "10", "100", "1,000"])
        ax.minorticks_off()
        ax.set_xlim(0.07, 1500)
        ax.set_ylim(0, 1.02)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_title(COHORTS[ds], pad=3)
        ax.set_xlabel(r"privacy budget $\varepsilon$ (log scale)")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=2.5, width=0.6)
    axes[0].set_ylabel("AUC, or share of real-data utility")
    axes[0].text(0.085, 0.455, "chance", fontsize=5.5, color=MUTED, va="top")
    axes[0].text(5.5, 0.97, "bound on\nany attack", fontsize=5.5, color=MUTED, va="top",
                 ha="right", linespacing=0.95)
    c = t[t.dataset == "COMBINED"].set_index("epsilon")
    axes[1].text(1300, c.auc[1000.0] - 0.045, "MAMA-MIA", fontsize=6, color=green,
                 ha="right", va="top")
    axes[1].text(1300, c.utility[1000.0] - 0.05, "utility", fontsize=6, color=INK2,
                 ha="right", va="top")
    save(fig, name, t)
    plt.close(fig)


# ---- Table: AUC and TPR of every attack, for the appendix ------------------------
def latex_grid(c: pd.DataFrame, cohorts=("BRCA", "COMBINED", "GTEX")) -> str:
    """Body rows of the extended table: 'AUC / TPR at 1% FPR' per cell.

    The generic row is the generic attack with the highest AUC, with its own TPR.
    """
    gens = list(GENERATORS)
    lines = []
    for ds in cohorts:
        d = c[(c.dataset == ds) & (c.group != "main")]
        cols = [g for g in gens if (d.generator == g).any()]
        lines.append(r"\midrule")
        lines.append(r"\multicolumn{%d}{l}{\emph{%s}} \\" % (len(gens) + 1, COHORTS[ds]))
        for row in [r for r in HEAT_ROWS if r]:
            if row == "Best generic attack":
                g = d[d.group == "generic"]
                pick = g.loc[g.groupby("generator")["auc"].idxmax()].set_index("generator")
            else:
                pick = d[d.row == row].set_index("generator")
            if pick.empty:
                continue
            cellstr = []
            for g in gens:
                if g not in pick.index or g not in cols:
                    cellstr.append("")
                    continue
                r = pick.loc[g]
                mark = r"$^{%d}$" % r.n_splits if r.n_splits < 5 else ""
                cellstr.append(f"{r.auc:.2f} / {r['tpr_at_fpr_0.01']:.2f}{mark}")
            name = {"MahalaMIA": r"\mahala{}", "MAMA-MIA": r"\mama{}", "RedSigma": r"\redsigma{}"}.get(
                row, row.replace("MeLoMIA", r"\melo{}"))
            lines.append(name + " & " + " & ".join(cellstr) + r" \\")
        sd = d.auc_sd.max()
        lines.append(f"% largest split-to-split SD of AUC in {ds}: {sd:.3f}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cells", action="store_true", help="print the cells and stop")
    ap.add_argument("--latex", action="store_true",
                    help="print the rows of the appendix's extended table and stop")
    args = ap.parse_args()

    c = cells(load_runs())
    if args.latex:
        print(latex_grid(c))
        return
    if not args.cells:
        style()
        fig_attack_grid(c)
        fig_attack_grid(c, rows=HEAT_ROWS, name="attack_grid_backends", width=5.6)
        fig_exposure(c)
        fig_cohort_size()
        fig_epsilon()
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
