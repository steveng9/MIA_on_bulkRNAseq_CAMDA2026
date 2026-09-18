#!/usr/bin/env python
"""Paper figures from the recorded runs.

    python scripts/make_figures.py --config configs/experiments/grid_brca.yaml
    python scripts/make_figures.py --dataset BRCA --figures grid roc
    python scripts/make_figures.py --sweep n_shadows --dataset BRCA

Figures written to results/figures/ as PDF (for LaTeX) and PNG (for review).

  grid   attack x generator heatmap.  Diverging around AUC 0.5, because the
         reader's question about any cell is which side of chance it falls on.
         Every cell carries its number, so the colour is a second channel.
  roc    ROC curve per generator for one attack, from the row-level scores.
         Pooled across splits.
  sweep  a metric against one configuration parameter, one line per attack.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from mia import palette as P  # noqa: E402
from mia import paths, runs as R  # noqa: E402

ATTACK_ORDER = ["mahalamia", "melomia_cvae", "melomia_nd", "mamamia"]
GENERATOR_ORDER = ["mvn", "cvae", "nd", "pgm"]
METRIC_LABELS = {"auc": "AUC-ROC", "aupr": "AUPR",
                 "tpr_at_fpr_0.01": "TPR @ 1% FPR",
                 "tpr_at_fpr_0.1": "TPR @ 10% FPR"}


def _out(name: str):
    d = Path(paths.RESULTS) / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d / name


def _save(fig, stem: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(_out(f"{stem}.{ext}"))
    plt.close(fig)
    print(f"  wrote {_out(stem + '.pdf')}")


def _order(values, preferred):
    return [v for v in preferred if v in values] + \
           sorted(v for v in values if v not in preferred)


# ── Grid heatmap ─────────────────────────────────────────────────────────────

def figure_grid(df: pd.DataFrame, dataset: str, metric: str = "auc") -> None:
    cell = df.groupby(["row", "generator"])[metric].mean().unstack()
    attacks = _order(cell.index, ATTACK_ORDER)
    gens = _order(cell.columns, GENERATOR_ORDER)
    cell = cell.reindex(index=attacks, columns=gens)
    values = cell.values.astype(float)

    # Symmetric about chance so the neutral midpoint lands exactly on 0.5.
    baseline = 0.5 if metric == "auc" else float(np.nanmin(values))
    span = max(0.05, float(np.nanmax(np.abs(values - baseline))))

    fig, ax = plt.subplots(figsize=(1.55 * len(gens) + 2.4, 0.95 * len(attacks) + 1.8))
    im = ax.imshow(values, cmap=P.diverging_cmap(), vmin=baseline - span,
                   vmax=baseline + span, aspect="auto")
    ax.grid(False)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            if np.isnan(v):
                ax.text(j, i, "--", ha="center", va="center", color=P.TEXT_MUTED)
                continue
            # Ink stays in text tokens; the cell fill carries magnitude.
            strong = abs(v - baseline) > 0.62 * span
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9,
                    color="#ffffff" if strong else P.TEXT_PRIMARY)

    ax.set_xticks(range(len(gens)),
                  [P.GENERATOR_LABELS.get(g, g) for g in gens])
    ax.set_yticks(range(len(attacks)),
                  [P.ATTACK_LABELS.get(a, a) for a in attacks])
    ax.set_ylim(len(attacks) - 0.5, -0.5)
    ax.set_xlabel("target generator")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    ax.set_title(f"{METRIC_LABELS.get(metric, metric)} — {dataset}",
                 loc="left", pad=12, fontsize=11)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, colors=P.TEXT_SECONDARY)
    if metric == "auc":
        cb.set_label("0.5 = no better than guessing", color=P.TEXT_SECONDARY)
    _save(fig, f"grid_{dataset}_{metric}")


# ── ROC curves ───────────────────────────────────────────────────────────────

def figure_roc(df: pd.DataFrame, dataset: str, attack: str) -> None:
    from sklearn.metrics import roc_curve

    sub = df[df.row == attack]
    if sub.empty:
        print(f"  [skip] no runs for {attack}")
        return
    gens = _order(sub.generator.unique(), GENERATOR_ORDER)

    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.plot([0, 1], [0, 1], color=P.TEXT_MUTED, linewidth=0.9, linestyle=(0, (4, 3)),
            zorder=1)
    ax.text(0.90, 0.855, "chance", color=P.TEXT_MUTED, fontsize=8, rotation=38,
            ha="center")

    curves, placed = [], []
    for gen in gens:
        rows = sub[sub.generator == gen]
        scores, labels = [], []
        for run_id in rows.run_id:
            s = pd.read_csv(paths.RUNS_DIR / run_id / "scores.csv")
            scores.append(s.score.values)
            labels.append(s.y_member.values)
        if not scores:
            continue
        fpr, tpr, _ = roc_curve(np.concatenate(labels), np.concatenate(scores))
        color = P.GENERATOR_COLORS.get(gen, P.CATEGORICAL[0])
        ax.plot(fpr, tpr, color=color, zorder=3,
                label=f"{P.GENERATOR_LABELS.get(gen, gen)} ({rows.auc.mean():.3f})")
        curves.append((gen, fpr, tpr))

    # Direct labels, so identity never rests on the legend's colour swatches
    # alone.  ROC curves crowd together near the top-left, so each label is
    # placed at the x where its curve is furthest from every already-placed
    # label rather than at a fixed fraction along the curve.
    for gen, fpr, tpr in curves:
        best_x, best_y, best_gap = None, None, -1.0
        for frac in np.linspace(0.15, 0.80, 40):
            k = int(frac * len(fpr))
            x, y = float(fpr[k]), float(tpr[k])
            gap = min([abs(y - py) + abs(x - px) * 0.4 for px, py in placed] or [9.0])
            gap = min(gap, abs(y - x) + 0.08)          # keep clear of the diagonal
            if gap > best_gap:
                best_x, best_y, best_gap = x, y, gap
        placed.append((best_x, best_y))
        ax.annotate(P.GENERATOR_LABELS.get(gen, gen), (best_x, best_y),
                    textcoords="offset points", xytext=(7, -3),
                    fontsize=8, color=P.TEXT_SECONDARY, zorder=4)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title(f"{P.ATTACK_LABELS.get(attack, attack)} — {dataset}",
                 loc="left", pad=10, fontsize=11)
    ax.legend(loc="lower right", fontsize=8, title="mean AUC",
              title_fontsize=8, labelcolor=P.TEXT_SECONDARY)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    _save(fig, f"roc_{dataset}_{attack}")


# ── Parameter sweep ──────────────────────────────────────────────────────────

def figure_sweep(dataset: str, param: str, metric: str = "auc",
                 generator: str | None = None) -> None:
    """Plot `metric` against one parameter, reading the value from each config."""
    import json

    rows = []
    for d in sorted(paths.RUNS_DIR.glob(f"{dataset}__*")):
        cfg_path, met_path = d / "config.json", d / "metrics.json"
        if not (cfg_path.exists() and met_path.exists()):
            continue
        cfg = json.loads(cfg_path.read_text())
        value = cfg.get("params", {}).get(param)
        if value is None:
            continue
        if generator and cfg["generator"] != generator:
            continue
        rows.append({"attack": cfg["attack"], "generator": cfg["generator"],
                     param: value,
                     metric: json.loads(met_path.read_text()).get(metric)})
    if not rows:
        print(f"  [skip] no runs vary {param}")
        return

    df = pd.DataFrame(rows).dropna()
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    for attack in _order(df.attack.unique(), ATTACK_ORDER):
        sub = df[df.attack == attack].groupby(param)[metric].agg(["mean", "std"])
        color = P.ATTACK_COLORS.get(attack, P.CATEGORICAL[0])
        ax.plot(sub.index, sub["mean"], color=color, marker="o", zorder=3,
                label=P.ATTACK_LABELS.get(attack, attack))
        ax.fill_between(sub.index, sub["mean"] - sub["std"].fillna(0),
                        sub["mean"] + sub["std"].fillna(0),
                        color=color, alpha=0.13, linewidth=0, zorder=2)
        ax.annotate(P.ATTACK_LABELS.get(attack, attack),
                    (sub.index[-1], sub["mean"].iloc[-1]),
                    textcoords="offset points", xytext=(6, 0),
                    fontsize=8, color=P.TEXT_SECONDARY, va="center")

    if metric == "auc":
        ax.axhline(0.5, color=P.TEXT_MUTED, linewidth=0.9, linestyle=(0, (4, 3)))
    ax.set_xlabel(param.replace("_", " "))
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_title(f"{METRIC_LABELS.get(metric, metric)} vs {param.replace('_', ' ')}"
                 f" — {dataset}", loc="left", pad=10, fontsize=11)
    ax.legend(fontsize=8, labelcolor=P.TEXT_SECONDARY)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    _save(fig, f"sweep_{dataset}_{param}_{metric}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None,
                   help="experiment YAML whose attack variants define the rows")
    p.add_argument("--dataset", default="BRCA")
    p.add_argument("--figures", nargs="+", default=["grid", "roc"],
                   choices=["grid", "roc"])
    p.add_argument("--metrics", nargs="+", default=["auc", "tpr_at_fpr_0.1"])
    p.add_argument("--sweep", default=None, help="parameter name to sweep over")
    p.add_argument("--tag", default=None)
    args = p.parse_args()

    P.apply_style()

    pairs = labels = None
    if args.config:
        from make_tables import apply_selection, selection_from_config
        cfg_dataset, pairs, labels = selection_from_config(args.config)
        args.dataset = args.dataset or cfg_dataset

    df = R.load_index()
    df = df[df.dataset == args.dataset]
    if args.tag:
        df = df[df.tag == args.tag]
    for c in ("auc", "aupr", "tpr_at_fpr_0.01", "tpr_at_fpr_0.1"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["tag"] = df["tag"].fillna("")
    if pairs:
        df = apply_selection(df, pairs)
    if df.empty:
        raise SystemExit(f"No runs for {args.dataset}.")

    # One row per attack variant, so hyperparameter variants never merge.
    labels = labels or {}
    df = df.assign(row=[labels.get((a, t), a if not t or t in ("default", "aux")
                                   else f"{a} [{t}]")
                        for a, t in zip(df.attack, df.tag)])

    if args.sweep:
        for m in args.metrics:
            figure_sweep(args.dataset, args.sweep, m)
        return

    if "grid" in args.figures:
        for m in args.metrics:
            figure_grid(df, args.dataset, m)
    if "roc" in args.figures:
        for attack in _order(df.row.unique(), ATTACK_ORDER):
            figure_roc(df, args.dataset, attack)


if __name__ == "__main__":
    main()
