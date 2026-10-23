#!/usr/bin/env python
"""UMAP / PCA of real training rows vs synthetic data: 2 cohorts x 5 generators, one figure per split.

    python scripts/umap_grid_slides.py                 # UMAP, splits 1-5
    python scripts/umap_grid_slides.py --kind pca --splits 1

For Steven (2026-09-30): how much closer the improved DP-PGM sits to the real
data than the one the CAMDA-26 challenge released.  Rows are BRCA and COMBINED;
columns are MVN, CVAE, ND, DP-PGM CAMDA-26 (`pgg`, the CAMDA-25 winner's code)
and DP-PGM new (the slide table's generators, `slide_table.py`).
Real = the members of the split (the target's training rows), identical across
all columns of a row; synthetic = that generator's release for the split.

Each panel is its own UMAP, fitted jointly on real + synthetic after a
50-component PCA, so good synthetic data mixes with the real points and bad
data forms its own islands.  Points are drawn in a shuffled order so neither
colour systematically covers the other.

`--kind pca` instead fits 2 principal components on the real training rows
only and projects the synthetic data into them.  The real rows are the same
across a row, so the panels share one set of axes and one frame (the
0.5-99.5th percentiles of everything plotted in that row): the panels show
where each generator puts its data in the real data's own frame.

`--set new` (2026-10-23, PCA only) draws the four generators added in
October in the same frame: TabSyn (tuned), TabPFN, DP-CVAE and DPSynth.  Each
row's frame is the one the challenge-generator figure uses (real rows plus the
five generators above), so a panel here can be laid beside that figure.
DPSynth exists only on the 200 most variable genes (BRCA_HVG200), so its panel
has its own PCA, fitted on the same members' 200 genes.  `--set gtex` draws
the five challenge generators on GTEx, one row per split.  Missing targets
leave an empty panel that says so.

Writes results/figures/{umap,pca}_grid_s<split>.{pdf,png},
pca_new_s<split>.{pdf,png} and pca_gtex.{pdf,png}.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from mia import datasets as D  # noqa: E402
from mia import palette as P  # noqa: E402
from mia import paths  # noqa: E402
from mia import targets as T  # noqa: E402

NEW = "pgm@binning=dp_quantile,edge_estimator=threshold,n_bins=16"
COLS = [("mvn", "MVN"), ("cvae", "CVAE"), ("nd", "NoisyDiffusion"),
        ("pgg", "DP-PGM CAMDA-26"), (NEW, "DP-PGM new")]
TABSYN = "tabsyn@class_freq=train,diff_schedule=steps,latent_scale=std,preprocess=clip:0.001:0.999+standard"
NEW_COLS = [(TABSYN, "TabSyn (tuned)"), ("tabpfn", "TabPFN"), ("dpcvae", "DP-CVAE"),
            ("dpsynth@mechanism=star", "DPSynth (star, 200 genes)")]
HVG_ONLY = {"dpsynth@mechanism=star"}
REAL, SYN = "#2166ac", "#e08214"   # blue / orange


def _pca_frame(Zs):
    both = np.vstack(Zs)
    lo, hi = np.percentile(both, 0.5, 0), np.percentile(both, 99.5, 0)
    pad = 0.05 * (hi - lo)
    return lo - pad, hi + pad


def _pca_panel(ax, Zr, Zs, frame, ev, title=None, ylabel=None, note=None):
    """One real-vs-synthetic panel; Zs None draws an empty panel saying `note`."""
    ax.set_xticks([]), ax.set_yticks([])
    ax.grid(False)
    ax.set_xlim(frame[0][0], frame[1][0])
    ax.set_ylim(frame[0][1], frame[1][1])
    ax.set_xlabel(f"PC1 ({ev[0]:.0%})", fontsize=7)
    if title:
        ax.set_title(title, fontsize=10)
    if ylabel:
        ax.set_ylabel(f"{ylabel}\nPC2 ({ev[1]:.0%})", fontsize=10, fontweight="bold")
    if Zs is None:
        ax.text(0.5, 0.5, note or "not built", ha="center", va="center", fontsize=8,
                color=P.TEXT_MUTED, transform=ax.transAxes)
        return
    Z = np.vstack([Zr, Zs])
    col = np.array([REAL] * len(Zr) + [SYN] * len(Zs))
    order = np.random.default_rng(0).permutation(len(Z))
    ax.scatter(Z[order, 0], Z[order, 1], s=3, c=col[order], alpha=0.5,
               linewidths=0, rasterized=True)
    ax.text(0.02, 0.02, f"real {len(Zr)} · synth {len(Zs)}" + (f" · {note}" if note else ""),
            fontsize=6.5, color=P.TEXT_MUTED, transform=ax.transAxes)


def _load(ds, gen, split):
    if not T.exists(ds, gen, split):
        return None
    return T.load_target(ds, gen, split)["X"].astype(np.float64)


def _finish(fig, title, stem, out, top=1.0):
    handles = [plt.Line2D([], [], ls="", marker="o", ms=6, color=REAL,
                          label="real training data (members of this split)"),
               plt.Line2D([], [], ls="", marker="o", ms=6, color=SYN, label="synthetic")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, top))
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}/{stem}.{{pdf,png}}", flush=True)


def pca_new(splits, out):
    """The four October generators, framed like the challenge-generator figure."""
    data = {ds: D.load_expression(ds).values.astype(np.float64)
            for ds in ("BRCA", "COMBINED", "BRCA_HVG200")}
    for split in splits:
        fig, axes = plt.subplots(2, len(NEW_COLS), figsize=(2.75 * len(NEW_COLS), 5.6),
                                 squeeze=False)
        for r, ds in enumerate(("BRCA", "COMBINED")):
            m = D.membership_labels(ds, split).astype(bool)
            Xr = data[ds][m]
            pca = PCA(n_components=2).fit(Xr)
            Zr = pca.transform(Xr)
            old = [_load(ds, g, split) for g, _ in COLS]
            frame = _pca_frame([Zr] + [pca.transform(X) for X in old if X is not None])
            for c, (gen, label) in enumerate(NEW_COLS):
                ylabel = ds if c == 0 else None
                title = label if r == 0 else None
                if gen in HVG_ONLY:
                    hds = f"{ds}_HVG200"
                    Xs = _load(hds, gen, split) if hds in data else None
                    if Xs is None:
                        _pca_panel(axes[r, c], Zr, None, frame, pca.explained_variance_ratio_,
                                   title, ylabel,
                                   "not built for this split" if hds in data and T.exists(hds, gen, 1)
                                   else f"not run on {ds}")
                        continue
                    Xh = data[hds][D.membership_labels(hds, split).astype(bool)]
                    ph = PCA(n_components=2).fit(Xh)
                    Zh, Zs = ph.transform(Xh), ph.transform(Xs)
                    _pca_panel(axes[r, c], Zh, Zs, _pca_frame([Zh, Zs]),
                               ph.explained_variance_ratio_, title, ylabel, "own PCA")
                else:
                    Xs = _load(ds, gen, split)
                    Zs = None if Xs is None else pca.transform(Xs)
                    note = None if Xs is not None else (
                        "not built for this split" if T.exists(ds, gen, 1) else f"not run on {ds}")
                    if Zs is not None:
                        outside = ((Zs < frame[0]) | (Zs > frame[1])).any(1).mean()
                        print(f"  s{split} {ds} {label}: {outside:.1%} of synthetic rows "
                              "outside the frame", flush=True)
                    _pca_panel(axes[r, c], Zr, Zs, frame, pca.explained_variance_ratio_,
                               title, ylabel, note)
        _finish(fig, f"PCA fitted on the real training data, the generators added in October, "
                     f"split {split} (ε = 10 for DP-CVAE and DPSynth)", f"pca_new_s{split}", out)


def pca_gtex(splits, out):
    """The five challenge generators on GTEx, one row per split."""
    X = D.load_expression("GTEX").values.astype(np.float64)
    fig, axes = plt.subplots(len(splits), len(COLS), figsize=(2.75 * len(COLS), 2.8 * len(splits)),
                             squeeze=False)
    for r, split in enumerate(splits):
        Xr = X[D.membership_labels("GTEX", split).astype(bool)]
        pca = PCA(n_components=2).fit(Xr)
        Zr = pca.transform(Xr)
        Zs_all = [None if (Xs := _load("GTEX", g, split)) is None else pca.transform(Xs)
                  for g, _ in COLS]
        frame = _pca_frame([Zr] + [Z for Z in Zs_all if Z is not None])
        for c, (_, label) in enumerate(COLS):
            _pca_panel(axes[r, c], Zr, Zs_all[c], frame, pca.explained_variance_ratio_,
                       label if r == 0 else None, f"split {split}" if c == 0 else None)
    _finish(fig, "GTEx: PCA fitted on the real training data, real vs synthetic "
                 "(ε = 10 for both DP-PGMs)", "pca_gtex", out, top=0.985)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--kind", choices=("umap", "pca"), default="umap")
    ap.add_argument("--set", choices=("challenge", "new", "gtex"), default="challenge",
                    help="new / gtex: PCA only (see above)")
    args = ap.parse_args()
    if args.set != "challenge":
        P.apply_style()
        out = paths.RESULTS / "figures"
        out.mkdir(parents=True, exist_ok=True)
        (pca_new if args.set == "new" else pca_gtex)(args.splits, out)
        return
    if args.kind == "umap":
        import umap

    P.apply_style()
    out = paths.RESULTS / "figures"
    out.mkdir(parents=True, exist_ok=True)
    data = {ds: D.load_expression(ds).values.astype(np.float64) for ds in ("BRCA", "COMBINED")}
    for split in args.splits:
        fig, axes = plt.subplots(2, len(COLS), figsize=(2.75 * len(COLS), 5.6), squeeze=False)
        for r, ds in enumerate(("BRCA", "COMBINED")):
            m = D.membership_labels(ds, split).astype(bool)
            Xr = data[ds][m]
            if args.kind == "pca":
                pca = PCA(n_components=2).fit(Xr)
                ev = pca.explained_variance_ratio_
                Zs_all = [pca.transform(T.load_target(ds, g, split)["X"].astype(np.float64))
                          for g, _ in COLS]
                both = np.vstack([pca.transform(Xr)] + Zs_all)
                lo, hi = np.percentile(both, 0.5, 0), np.percentile(both, 99.5, 0)
                pad = 0.05 * (hi - lo)
            for c, (gen, label) in enumerate(COLS):
                ax = axes[r, c]
                ax.set_xticks([]), ax.set_yticks([])
                ax.grid(False)
                Xs = T.load_target(ds, gen, split)["X"].astype(np.float64)
                if args.kind == "pca":
                    Z = np.vstack([pca.transform(Xr), Zs_all[c]])
                    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
                    ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
                    ax.set_xlabel(f"PC1 ({ev[0]:.0%})", fontsize=7)
                else:
                    Z = PCA(n_components=50, random_state=0).fit_transform(np.vstack([Xr, Xs]))
                    Z = umap.UMAP(random_state=0, n_neighbors=30, min_dist=0.3).fit_transform(Z)
                col = np.array([REAL] * len(Xr) + [SYN] * len(Xs))
                order = np.random.default_rng(0).permutation(len(Z))
                ax.scatter(Z[order, 0], Z[order, 1], s=3, c=col[order], alpha=0.5,
                           linewidths=0, rasterized=True)
                if r == 0:
                    ax.set_title(label, fontsize=10)
                if c == 0:
                    ax.set_ylabel(ds if args.kind == "umap" else f"{ds}\nPC2 ({ev[1]:.0%})",
                                  fontsize=10, fontweight="bold")
                ax.text(0.02, 0.02, f"real {len(Xr)} · synth {len(Xs)}", fontsize=6.5,
                        color=P.TEXT_MUTED, transform=ax.transAxes)
                print(f"  s{split} {ds} {label}: real {len(Xr)}, synthetic {len(Xs)}", flush=True)
        handles = [plt.Line2D([], [], ls="", marker="o", ms=6, color=REAL,
                              label="real training data (members of this split)"),
                   plt.Line2D([], [], ls="", marker="o", ms=6, color=SYN, label="synthetic")]
        fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9, frameon=False,
                   bbox_to_anchor=(0.5, -0.01))
        title = "UMAP" if args.kind == "umap" else "PCA fitted on the real training data"
        fig.suptitle(f"{title}, real vs synthetic, split {split} (ε = 10 for both DP-PGMs)",
                     fontsize=11)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        for ext in ("pdf", "png"):
            fig.savefig(out / f"{args.kind}_grid_s{split}.{ext}", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}/{args.kind}_grid_s{split}.{{pdf,png}}", flush=True)


if __name__ == "__main__":
    main()
