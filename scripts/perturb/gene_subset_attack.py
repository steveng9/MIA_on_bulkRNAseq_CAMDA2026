"""How many genes does an attacker need, and does it matter which genes?

Team request (2026-11 meeting): extend the random gene-subset result (Hakime's
experiment 3: the covariance attack needs most genes, nearest-neighbour
attacks saturate at ~50, which genes hardly matters for a RANDOM subset) with
biologically targeted subsets, to get at the central question -- the minimum
number of genes an attacker needs to reach a given membership-inference AUC,
and whether a realistic attacker (who already knows which genes matter for a
disease) needs far fewer than a naive one.

Rules, computed once per dataset over the whole candidate pool (a cohort-level
step: it sees members and non-members alike, so the selection itself carries
no membership signal -- see `mia/preprocessing.py`):

  random   k genes drawn uniformly (the existing control)
  hvg      the k highest-variance genes across the cohort
  hallmark a fixed, disease-relevant panel chosen independently of this
           cohort's data: `configs/gene_sets/brca_hallmark_27.ensembl.txt`,
           27 of the 978 landmark genes (PAM50 + core driver / EMT genes,
           symbol -> Ensembl via NCBI gene_info, BRCA only -- a pan-cancer
           cohort has no single "hallmark" panel)

The attack is the paper's default MahalaMIA (`mia/views.py` ROSTER), run
inside a `view(genes=...)` so nothing is retrained: the released synthetic
data and the real candidates are both restricted to the k columns, exactly
what a k-gene-limited adversary would see.

    python scripts/perturb/gene_subset_attack.py --dataset BRCA --jobs 6
    python scripts/perturb/gene_subset_attack.py --dataset COMBINED --jobs 6

Writes `results/perturb/gene_subset_attack_<dataset>.csv`.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mia import datasets as D  # noqa: E402
from mia import metrics as M  # noqa: E402
from mia import paths  # noqa: E402
from mia import targets as T  # noqa: E402
from mia import views as V  # noqa: E402

GENERATORS = ["mvn", "cvae", "nd"]
KS = [5, 10, 20, 30, 50, 98, 150, 300, 500, 978]
ATTACKS = ["MahalaMIA", "MahalaMIA (no reference)"]
HALLMARK = Path("configs/gene_sets/brca_hallmark_27.ensembl.txt")


def gene_sets(dataset: str) -> dict:
    """{(rule, k): array of column positions}, computed once over the whole cohort."""
    X = D.load_expression(dataset)
    cols = list(X.columns)
    var = X.var(axis=0)
    hvg_order = var.sort_values(ascending=False).index
    out = {}
    for k in KS:
        if k >= len(cols):
            out[("hvg", k)] = out[("random", k)] = np.arange(len(cols))
            continue
        out[("hvg", k)] = np.array([cols.index(g) for g in hvg_order[:k]])
        out[("random", k)] = np.sort(np.random.default_rng(k).choice(len(cols), k, replace=False))
    if dataset == "BRCA" and HALLMARK.exists():
        wanted = [g.strip() for g in HALLMARK.read_text().split() if g.strip()]
        missing = [g for g in wanted if g not in cols]
        if missing:
            raise KeyError(f"hallmark genes missing from {dataset}: {missing}")
        out[("hallmark", len(wanted))] = np.array([cols.index(g) for g in wanted])
    return out


def one_target(args):
    dataset, generator, split, out_dir = args
    part = Path(out_dir) / f"{dataset}__{generator}__s{split}.csv"
    if not T.exists(dataset, generator, split):
        return f"{dataset}/{generator}/s{split} not built"
    done = pd.read_csv(part) if part.exists() else pd.DataFrame(columns=["rule", "k"])
    t0 = time.time()
    y = D.membership_labels(dataset, split)
    sets = gene_sets(dataset)
    rows = []
    for (rule, k), idx in sets.items():
        if ((done.rule == rule) & (done.k == k)).any():
            continue
        with V.view(dataset, genes=idx):
            for label in ATTACKS:
                try:
                    s = V.build(label).score(dataset, generator, split)
                except Exception:
                    print(f"  FAILED {dataset}/{generator}/s{split} {label} {rule}:{k}\n"
                          + traceback.format_exc(limit=2), flush=True)
                    continue
                m = M.evaluate(y, s)
                rows.append(dict(dataset=dataset, generator=generator, split=split,
                                 rule=rule, k=k, attack=label,
                                 auc=m["auc"], tpr_at_fpr_0_01=m["tpr_at_fpr_0.01"],
                                 tpr_at_fpr_0_1=m["tpr_at_fpr_0.1"]))
    pd.concat([done, pd.DataFrame(rows)]).to_csv(part, index=False)
    return f"{part.name}: +{len(rows)} rows in {time.time() - t0:.0f}s"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--splits", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--generators", nargs="+", default=GENERATORS)
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args()
    out_dir = paths.RESULTS / "perturb" / "gene_subset_attack_parts"
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(args.dataset, g, s, str(out_dir)) for g in args.generators for s in args.splits]
    with ProcessPoolExecutor(args.jobs) as pool:
        for msg in pool.map(one_target, jobs):
            print(msg, flush=True)
    parts = [pd.read_csv(f) for f in sorted(out_dir.glob(f"{args.dataset}__*.csv"))]
    out = paths.RESULTS / "perturb" / f"gene_subset_attack_{args.dataset}.csv"
    pd.concat(parts).to_csv(out, index=False)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
