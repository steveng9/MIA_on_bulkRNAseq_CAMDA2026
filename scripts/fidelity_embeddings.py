#!/usr/bin/env python
"""PCA and UMAP views of real vs synthetic data, one panel per generator.

    python scripts/fidelity_embeddings.py                 # split 1, both cohorts
    python scripts/fidelity_embeddings.py --extra "pgm@...forest..."

For choosing the paper's DP-PGM configuration (Steven, 2026-09-24): does the
synthetic cloud sit where the real training data sits, and does it keep the
subtype structure?  Rows are cohorts; columns are CVAE, ND and several DP-PGM
configurations.  Real = the target's own training rows (the members of the
split), in grey; synthetic in the generator's colour.

  pca   axes fitted on the real training rows only, synthetic projected into
        them -- so a panel shows the synthetic data in the real data's frame
  umap  fitted jointly on real + synthetic (after a 50-component PCA), so good
        synthetic data mixes with the real points and bad data forms its own
        islands

Writes results/figures/fidelity_{pca,umap}_s<split>.{pdf,png}.
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

DQ = "binning=dp_quantile,edge_estimator=threshold"
PANELS = [
    ("cvae", "CVAE", P.GENERATOR_COLORS["cvae"]),
    ("nd", "NoisyDiffusion", P.GENERATOR_COLORS["nd"]),
    ("pgm", "DP-PGM, abstract config\n(4 private-quantile bins, ε=10)", P.GENERATOR_COLORS["pgm"]),
    (f"pgm@{DQ},n_bins=16", "DP-PGM star\n(dp_quantile 16, ε=10)", P.GENERATOR_COLORS["pgm"]),
    (f"pgm@{DQ},n_bins=16,structure=tree", "DP-PGM + DP tree\n(dp_quantile 16, ε=10)", P.GENERATOR_COLORS["pgm"]),
    (f"pgm@{DQ},epsilon=1000,n_bins=32", "DP-PGM star\n(dp_quantile 32, ε=1000)", P.GENERATOR_COLORS["pgm"]),
]


def load(ds, gen, split):
    X = D.load_expression(ds).values.astype(np.float64)
    m = D.membership_labels(ds, split).astype(bool)
    Xs = T.load_target(ds, gen, split)["X"].astype(np.float64)
    return X[m], Xs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", type=int, default=1)
    ap.add_argument("--datasets", nargs="+", default=["BRCA", "COMBINED"])
    ap.add_argument("--extra", nargs="*", default=[],
                    help="more target names, as label=name or bare name")
    args = ap.parse_args()
    panels = list(PANELS)
    for e in args.extra:
        lab, _, name = e.partition("=") if "=" in e else (e, "", e)
        panels.append((name or lab, lab, P.CATEGORICAL[4]))

    P.apply_style()
    import umap
    for kind in ("pca", "umap"):
        fig, axes = plt.subplots(len(args.datasets), len(panels),
                                 figsize=(2.3 * len(panels), 2.5 * len(args.datasets)),
                                 squeeze=False)
        for r, ds in enumerate(args.datasets):
            for c, (gen, label, colour) in enumerate(panels):
                ax = axes[r, c]
                ax.set_xticks([]), ax.set_yticks([])
                ax.grid(False)
                if not T.exists(ds, gen, args.split):
                    ax.text(0.5, 0.5, "not built", ha="center", va="center",
                            color=P.TEXT_MUTED, transform=ax.transAxes)
                    continue
                Xr, Xs = load(ds, gen, args.split)
                if kind == "pca":
                    pca = PCA(n_components=2).fit(Xr)
                    Zr, Zs = pca.transform(Xr), pca.transform(Xs)
                    ev = pca.explained_variance_ratio_
                    ax.set_xlabel(f"PC1 ({ev[0]:.0%})", fontsize=7)
                    ax.set_ylabel(f"PC2 ({ev[1]:.0%})", fontsize=7)
                    # frame on the real data so off-support synthetic data shows as clipped
                    lo, hi = Zr.min(0), Zr.max(0)
                    pad = 0.35 * (hi - lo)
                    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
                    ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
                else:
                    Z = PCA(n_components=50, random_state=0).fit_transform(np.vstack([Xr, Xs]))
                    Z = umap.UMAP(random_state=0, n_neighbors=30, min_dist=0.3).fit_transform(Z)
                    Zr, Zs = Z[:len(Xr)], Z[len(Xr):]
                ax.scatter(Zr[:, 0], Zr[:, 1], s=3, c=P.TEXT_MUTED, alpha=0.45,
                           linewidths=0, label="real (training rows)", rasterized=True)
                ax.scatter(Zs[:, 0], Zs[:, 1], s=3, c=colour, alpha=0.45,
                           linewidths=0, label="synthetic", rasterized=True)
                if r == 0:
                    ax.set_title(label, fontsize=8)
                if c == 0:
                    ax.text(-0.18, 0.5, ds, rotation=90, ha="center", va="center",
                            fontsize=9, fontweight="bold", transform=ax.transAxes)
                print(f"  {kind} {ds} {label.splitlines()[0]}", flush=True)
        handles = [plt.Line2D([], [], ls="", marker="o", ms=5, color=P.TEXT_MUTED,
                              label="real (training rows)"),
                   plt.Line2D([], [], ls="", marker="o", ms=5, color=P.GENERATOR_COLORS["pgm"],
                              label="synthetic (colour = generator)")]
        fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8,
                   bbox_to_anchor=(0.5, -0.02))
        fig.suptitle({"pca": "PCA fitted on the real training rows",
                      "umap": "UMAP fitted jointly on real + synthetic"}[kind],
                     fontsize=10, y=1.01)
        fig.tight_layout()
        out = Path(paths.RESULTS) / "figures"
        out.mkdir(parents=True, exist_ok=True)
        for ext in ("pdf", "png"):
            fig.savefig(out / f"fidelity_{kind}_s{args.split}.{ext}")
        plt.close(fig)
        print(f"wrote {out}/fidelity_{kind}_s{args.split}.pdf", flush=True)


if __name__ == "__main__":
    main()
