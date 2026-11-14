"""Figure: what the trained generator adds to each attack (appendix of the paper).

One row per (attack, generator) of scripts/whitebox/whitebox_table.py, one
column per TCGA cohort; AUC on top, TPR at 1% FPR (log scale) below.  An open
marker is the black-box attack (the paper grid's cell), the filled marker the
same attack white box, joined by a line; colour and shape are the attack's
family, as in the main-text figures.  For MAMA-MIA the white box is the fitted
model's marginals (the table also has true tables and bins).  Means over five
splits; a cell with fewer is starred.

    python scripts/whitebox/whitebox_figure.py   # paper/Fig/whitebox.{pdf,png}, data/whitebox.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "whitebox"))

import paper_figures as PF  # noqa: E402
from whitebox_table import COHORTS, ROWS  # noqa: E402

WHITE_ARM = {"MAMA-MIA": "model marginals"}   # the arm plotted where a row has several
FAMILY = {"MahalaMIA": "MahalaMIA", "MeLoMIA-CVAE": "MeLoMIA", "MeLoMIA-TabSyn": "MeLoMIA",
          "MAMA-MIA": "MAMA-MIA"}


def data(df) -> pd.DataFrame:
    out = []
    for attack, glabel, gen, bb, arms in ROWS:
        arm, choices = next(((a, c) for a, c in arms if a == WHITE_ARM.get(attack, a)))
        for ds in COHORTS:
            for access, cell in (("black box", PF.cell(df, ds, gen, bb)),
                                 ("white box", PF.cell(df, ds, gen, choices))):
                if cell is None:
                    continue
                for m in PF.METRICS:
                    out.append(dict(attack=attack, generator=glabel, row=f"{attack}, {glabel}",
                                    dataset=ds, access=access, white_box_reads=arm, metric=m,
                                    value=cell[m], n_splits=cell["n_splits"]))
    return pd.DataFrame(out)


def figure(t: pd.DataFrame, name="whitebox") -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    PF.style()
    rows = list(dict.fromkeys(t.row))
    attack_of = dict(zip(t.row, t.attack))
    # a gap between attacks
    ypos, y = {}, 0.0
    for i, r in enumerate(rows):
        if i and attack_of[r] != attack_of[rows[i - 1]]:
            y += 0.45
        ypos[r] = y
        y += 1
    top = y - 1
    ypos = {r: top - v for r, v in ypos.items()}
    fig, axes = plt.subplots(2, len(COHORTS), figsize=(7.1, 0.95 + 0.62 * y), sharey=True,
                             gridspec_kw=dict(hspace=0.40, wspace=0.08))
    for i, m in enumerate(PF.METRICS):
        log = m != "auc"
        chance = 0.5 if m == "auc" else 0.01
        lo, hi = (0.45, 1.02) if m == "auc" else (0.004, 1.15)
        clip = (lambda v: max(v, lo * 1.05)) if log else (lambda v: max(v, lo))
        for k, ds in enumerate(COHORTS):
            ax = axes[i, k]
            ax.axvline(chance, color=PF.MUTED, lw=0.8, ls=(0, (2, 2)), zorder=1)
            for r in rows:
                d = t[(t.row == r) & (t.dataset == ds) & (t.metric == m)].set_index("access")
                if not {"black box", "white box"} <= set(d.index):
                    continue
                fam = FAMILY[attack_of[r]]
                b, w = clip(d.value["black box"]), clip(d.value["white box"])
                yy = ypos[r]
                ax.plot([b, w], [yy, yy], color=PF.FAMILY_COLOR[fam], alpha=0.35, lw=2.4,
                        solid_capstyle="round", zorder=2)
                ax.plot(b, yy, PF.FAMILY_MARKER[fam], ms=5.5, mfc="white",
                        mec=PF.FAMILY_COLOR[fam], mew=1.1, zorder=3)
                ax.plot(w, yy, PF.FAMILY_MARKER[fam], ms=6, mfc=PF.FAMILY_COLOR[fam],
                        mec="white", mew=0.7, zorder=4)
                if d.n_splits.min() < 5:
                    ax.text(hi, yy, f"* {d.n_splits.min()} of 5 splits", ha="right",
                            va="center", fontsize=5.5, color=PF.MUTED)
            ax.set_xlim(lo, hi)
            if log:
                ax.set_xscale("log")
                ax.set_xticks([0.01, 0.03, 0.1, 0.3, 1.0])
                ax.set_xticklabels(["0.01", "0.03", "0.1", "0.3", "1"])
                ax.minorticks_off()
            else:
                ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
                ax.set_xticklabels(["0.5", "0.6", "0.7", "0.8", "0.9", "1.0"])
            ax.grid(axis="x", color=PF.GRID, lw=0.5, zorder=0)
            ax.set_axisbelow(True)
            for s in ("top", "right", "left"):
                ax.spines[s].set_visible(False)
            ax.set_ylim(-0.6, top + 0.9)
            ax.text(chance, top + 0.55, "chance", ha="center", va="bottom", fontsize=5.5,
                    color=PF.MUTED)
            if i == 0:
                ax.set_title(PF.COHORTS[ds], fontweight="bold", color=PF.INK, loc="left")
            ax.set_xlabel("AUC" if m == "auc" else "TPR at 1% FPR (log scale)")
            if k == 0:
                ax.set_yticks([ypos[r] for r in rows])
                ax.set_yticklabels(rows)
    fams = list(dict.fromkeys(FAMILY[a] for a in t.attack))
    handles = [Line2D([], [], ls="", marker=PF.FAMILY_MARKER[f], ms=6, mfc=PF.FAMILY_COLOR[f],
                      mec="white", label=f) for f in fams]
    handles += [Line2D([], [], ls="", marker="o", ms=5.5, mfc="white", mec=PF.INK2, mew=1.1,
                       label="black box"),
                Line2D([], [], ls="", marker="o", ms=6, mfc=PF.INK2, mec="white",
                       label="white box (holds the trained generator)")]
    fig.legend(handles=handles, ncol=len(handles), frameon=False, handletextpad=0.3,
               columnspacing=1.2, loc="lower left", bbox_to_anchor=(0.12, 0.93))
    PF.save(fig, name, t)
    plt.close(fig)


def main():
    figure(data(PF.load_runs()))


if __name__ == "__main__":
    main()
