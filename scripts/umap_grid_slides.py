#!/usr/bin/env python
"""UMAP / PCA of real training rows vs synthetic data: 2 cohorts x 4 generators, one figure per split.

    python scripts/umap_grid_slides.py                 # UMAP, splits 1-5
    python scripts/umap_grid_slides.py --kind pca --splits 1

For Steven (2026-09-30): how much closer the improved DP-PGM sits to the real
data than the CAMDA-25 one.  Rows are BRCA and COMBINED; columns are CVAE, ND,
DP-PGM old and DP-PGM new (the slide table's generators, `slide_table.py`).
Real = the members of the split (the target's training rows), identical across
the four columns of a row; synthetic = that generator's release for the split.

Each panel is its own UMAP, fitted jointly on real + synthetic after a
50-component PCA, so good synthetic data mixes with the real points and bad
data forms its own islands.  Points are drawn in a shuffled order so neither
colour systematically covers the other.

`--kind pca` instead fits 2 principal components on the real training rows
only and projects the synthetic data into them.  The real rows are the same
across a row, so the four panels share one set of axes and one frame (the
0.5-99.5th percentiles of everything plotted in that row): the panels show
where each generator puts its data in the real data's own frame.

Writes results/figures/{umap,pca}_grid_s<split>.{pdf,png}.
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

OLD = "pgm@composition=basic,neighboring=legacy_exact_n"
NEW = "pgm@binning=dp_quantile,edge_estimator=threshold,n_bins=16"
COLS = [("cvae", "CVAE"), ("nd", "NoisyDiffusion"),
        (OLD, "DP-PGM old (CAMDA-25)"), (NEW, "DP-PGM new")]
REAL, SYN = "#2166ac", "#e08214"   # blue / orange


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--kind", choices=("umap", "pca"), default="umap")
    args = ap.parse_args()
    if args.kind == "umap":
        import umap

    P.apply_style()
    out = paths.RESULTS / "figures"
    out.mkdir(parents=True, exist_ok=True)
    data = {ds: D.load_expression(ds).values.astype(np.float64) for ds in ("BRCA", "COMBINED")}
    for split in args.splits:
        fig, axes = plt.subplots(2, 4, figsize=(11, 5.6), squeeze=False)
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
