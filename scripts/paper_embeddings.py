#!/usr/bin/env python
"""Two-dimensional views of the real training data and every release (split 1).

    python scripts/paper_embeddings.py              # compute (cached) and draw
    python scripts/paper_embeddings.py --draw       # redraw from the cache
    python scripts/paper_embeddings.py --json out.json   # export for the comparison page

For every cohort, the real rows are the members of split 1 (what the generators
were trained on) and each release is that split's synthetic dataset, both
subsampled to at most 1,200 rows.  Everything starts from one 50-component PCA
fitted on the real training rows.  Three views are computed:

  pca          the first two components; the release is projected into the real
               data's axes, so distances and spread are comparable across panels
  tsne_joint   t-SNE fitted on the real rows and one release together: a release
               that resembles the real data mixes with it, one that does not
               forms islands of its own.  This is the paper's figure.
  tsne_shared  one t-SNE map fitted on the real rows alone (openTSNE); each
               release is then placed into that map.  Every panel of a cohort
               has the same coordinates, so panels can be compared side by side
               and coloured by class.  A synthetic row lands beside the real
               rows it is nearest to, however far it is from them, so this view
               shows which class a row resembles, not how close it is.

Figures written to paper/Fig/:
  tsne_strip      TCGA-BRCA, one panel per generator (main text)
  tsne_grid       all cohorts, t-SNE fitted jointly (supplement)
  pca_grid        all cohorts, PCA (supplement)
The cache is artifacts/paper/embeddings.npz.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from mia import datasets as D  # noqa: E402
from mia import paths  # noqa: E402
from mia import targets as T  # noqa: E402

CACHE = paths.ARTIFACTS / "paper" / "embeddings.npz"
OUT = ROOT / "paper" / "Fig"
COHORTS = {"BRCA": "TCGA-BRCA", "COMBINED": "TCGA-COMBINED", "GTEX": "GTEx"}
TABSYN = ("tabsyn@class_freq=train,diff_schedule=steps,latent_scale=std,"
          "preprocess=clip:0.001:0.999+standard")
PGM = "pgm@binning=dp_quantile,edge_estimator=threshold,n_bins=16"
GENERATORS = {"mvn": "MVN", "cvae": "CVAE", "nd": "NoisyDiffusion", TABSYN: "TabSyn",
              "tabpfn": "TabPFN", "pgg": "DP-PGM (challenge)", PGM: "DP-PGM (corrected)"}
SPLIT = 1
N_MAX = 1200
REAL, SYN = "#2a78d6", "#eb6834"


def sub(n: int, rng) -> np.ndarray:
    return np.sort(rng.choice(n, min(n, N_MAX), replace=False))


def compute() -> dict:
    from openTSNE import TSNE as OpenTSNE
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    out = {}
    for ds in COHORTS:
        rng = np.random.default_rng(0)
        X = D.load_expression(ds).values.astype(np.float64)
        lab = np.asarray(D.load_subtypes(ds).values).astype(str)
        m = D.membership_labels(ds, SPLIT).astype(bool)
        pca = PCA(n_components=50, random_state=0).fit(X[m])
        keep = sub(int(m.sum()), rng)
        Zr, yr = pca.transform(X[m])[keep], lab[m][keep]
        shared = OpenTSNE(perplexity=30, random_state=0, n_jobs=4).fit(Zr)
        out[f"{ds}|real|pca"] = Zr[:, :2]
        out[f"{ds}|real|tsne_shared"] = np.asarray(shared)
        out[f"{ds}|real|label"] = yr
        out[f"{ds}|variance"] = pca.explained_variance_ratio_[:2]
        for g in GENERATORS:
            if not T.exists(ds, g, SPLIT):
                continue
            t = T.load_target(ds, g, SPLIT)
            keep = sub(len(t["X"]), rng)
            Zs = pca.transform(t["X"].astype(np.float64)[keep])
            joint = TSNE(n_components=2, perplexity=30, init="pca", random_state=0,
                         n_jobs=4).fit_transform(np.vstack([Zr, Zs]))
            out[f"{ds}|{g}|pca"] = Zs[:, :2]
            out[f"{ds}|{g}|tsne_joint_real"] = joint[:len(Zr)]
            out[f"{ds}|{g}|tsne_joint"] = joint[len(Zr):]
            out[f"{ds}|{g}|tsne_shared"] = np.asarray(shared.transform(Zs, perplexity=30))
            out[f"{ds}|{g}|label"] = np.asarray(t["y_str"]).astype(str)[keep]
            print(f"  {ds} {GENERATORS[g]}", flush=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, **out)
    return out


def present(E, ds):
    return [g for g in GENERATORS if f"{ds}|{g}|pca" in E]


def panel(ax, real, syn, title=None):
    ax.scatter(real[:, 0], real[:, 1], s=1.2, c=REAL, alpha=0.45, linewidths=0, rasterized=True)
    ax.scatter(syn[:, 0], syn[:, 1], s=1.2, c=SYN, alpha=0.45, linewidths=0, rasterized=True)
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.4), s.set_color("#b9b8b3")
    if title:
        ax.set_title(title, fontsize=6.5, pad=2.5)


def legend(fig, y):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", ms=3, color=REAL, label="real training data"),
         Line2D([], [], marker="o", ls="", ms=3, color=SYN, label="synthetic release")]
    fig.legend(handles=h, loc="lower center", ncol=2, frameon=False, fontsize=6.5,
               bbox_to_anchor=(0.5, y), handletextpad=0.2, columnspacing=1.5)


def draw(E) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "pdf.fonttype": 42, "savefig.dpi": 300,
                         "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"]})
    short = {**GENERATORS, "nd": "NoisyDiffusion", "pgg": "DP-PGM (challenge)",
             PGM: "DP-PGM (corrected)"}

    gens = present(E, "BRCA")
    fig, axes = plt.subplots(1, len(gens), figsize=(7.0, 1.12))
    for ax, g in zip(axes, gens):
        panel(ax, E[f"BRCA|{g}|tsne_joint_real"], E[f"BRCA|{g}|tsne_joint"], short[g])
    fig.subplots_adjust(left=0.005, right=0.995, top=0.86, bottom=0.13, wspace=0.06)
    legend(fig, -0.035)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"tsne_strip.{ext}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    for view, name in (("tsne_joint", "tsne_grid"), ("pca", "pca_grid")):
        fig, axes = plt.subplots(len(COHORTS), len(GENERATORS), figsize=(7.0, 3.25))
        for i, ds in enumerate(COHORTS):
            for j, g in enumerate(GENERATORS):
                ax = axes[i, j]
                if f"{ds}|{g}|pca" not in E:
                    ax.axis("off")
                    continue
                real = E[f"{ds}|{g}|tsne_joint_real"] if view == "tsne_joint" else E[f"{ds}|real|pca"]
                panel(ax, real, E[f"{ds}|{g}|{view}"], short[g] if i == 0 else None)
                if view == "pca":   # one frame per cohort: the real data's range, padded
                    r = E[f"{ds}|real|pca"]
                    pad = 0.35 * (r.max(0) - r.min(0))
                    ax.set_xlim(r[:, 0].min() - pad[0], r[:, 0].max() + pad[0])
                    ax.set_ylim(r[:, 1].min() - pad[1], r[:, 1].max() + pad[1])
            axes[i, 0].set_ylabel(COHORTS[ds], fontsize=6.5, labelpad=2)
        fig.subplots_adjust(left=0.03, right=0.995, top=0.95, bottom=0.06, wspace=0.06, hspace=0.06)
        legend(fig, -0.005)
        for ext in ("pdf", "png"):
            fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    print("wrote paper/Fig/tsne_strip, tsne_grid, pca_grid (.pdf, .png)")


def export(E, path: str) -> None:
    """Compact JSON for the comparison page: integer coordinates per view, class index per row."""
    doc = {"split": SPLIT, "cohorts": []}
    for ds, title in COHORTS.items():
        classes = sorted(set(E[f"{ds}|real|label"]))
        idx = {c: i for i, c in enumerate(classes)}
        sets = []
        for g, name in [("real", "Real training data")] + [(g, GENERATORS[g]) for g in present(E, ds)]:
            s = {"name": name, "y": [idx.get(c, -1) for c in E[f"{ds}|{g}|label"]]}
            for view in ("pca", "tsne_shared"):
                s[view] = np.round(E[f"{ds}|{g}|{view}"], 2).tolist()
            if g != "real":
                s["tsne_joint"] = np.round(E[f"{ds}|{g}|tsne_joint"], 1).tolist()
                s["tsne_joint_real"] = np.round(E[f"{ds}|{g}|tsne_joint_real"], 1).tolist()
            sets.append(s)
        doc["cohorts"].append({"id": ds, "title": title, "classes": classes, "sets": sets,
                               "variance": np.round(E[f"{ds}|variance"], 3).tolist()})
    Path(path).write_text(json.dumps(doc, separators=(",", ":")))
    print(f"wrote {path} ({Path(path).stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--draw", action="store_true", help="redraw from the cache")
    p.add_argument("--json", help="also export the embeddings as JSON to this path")
    args = p.parse_args()
    E = dict(np.load(CACHE, allow_pickle=False)) if (args.draw and CACHE.exists()) else compute()
    draw(E)
    if args.json:
        export(E, args.json)


if __name__ == "__main__":
    main()
