"""How should k genes be chosen when a generator cannot handle all of them?

Generator-independent: for each selection rule and each k, on the real cohort,

  utility   macro-F1 of a class predictor trained on the members' k genes and
            tested on the non-members (what a perfect generator restricted to
            these genes could still deliver), against all genes;
  coverage  share of the variance of *all* genes that a ridge regression on
            the k selected genes explains on held-out samples (how much of the
            transcriptome the subset still carries);
  stability Jaccard overlap between the set chosen on the whole cohort and the
            sets chosen on each split's members only (if high, choosing on the
            pooled cohort tells an attacker next to nothing about membership).

    python scripts/gene_subset_ablation.py --dataset BRCA
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mia import datasets as D, paths, preprocessing as pp   # noqa: E402

OUT = paths.RESULTS / "gene_subset_ablation.csv"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="BRCA")
    p.add_argument("--rules", nargs="+", default=["anova", "hvg", "random"])
    p.add_argument("--ks", nargs="+", type=int, default=[25, 50, 100, 200, 400])
    p.add_argument("--splits", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    args = p.parse_args()

    X = D.load_expression(args.dataset)
    y = pd.Series(D.encode_subtypes(args.dataset, D.load_subtypes(args.dataset).values), index=X.index)
    members = {s: D.membership_labels(args.dataset, s).astype(bool) for s in args.splits}

    def score(cols, s):
        m = members[s]
        Xa = X[cols].values
        clf = HistGradientBoostingClassifier(max_iter=150, random_state=0).fit(Xa[m], y.values[m])
        f1 = f1_score(y.values[~m], clf.predict(Xa[~m]), average="macro")
        full = X.values
        mu, sd = full[m].mean(0), full[m].std(0) + 1e-9
        reg = Ridge(alpha=1.0).fit((Xa[m] - Xa[m].mean(0)) / (Xa[m].std(0) + 1e-9), (full[m] - mu) / sd)
        pred = reg.predict((Xa[~m] - Xa[m].mean(0)) / (Xa[m].std(0) + 1e-9))
        resid = (((full[~m] - mu) / sd - pred) ** 2).sum()
        total = (((full[~m] - mu) / sd) ** 2).sum()
        return f1, 1 - resid / total

    rows = []
    all_f1 = np.mean([score(list(X.columns), s)[0] for s in args.splits])
    print(f"{args.dataset}: all {X.shape[1]} genes, real-on-real macro-F1 {all_f1:.3f}", flush=True)
    for rule in args.rules:
        for k in args.ks:
            spec = f"{rule}:{k}" + (":0" if rule == "random" else "")
            cols = list(pp.prepare_cohort(X, spec, labels=y).columns)
            f1, cov = np.mean([score(cols, s) for s in args.splits], axis=0)
            jac = np.mean([len(set(cols) & set(c)) / len(set(cols) | set(c))
                           for c in (pp.prepare_cohort(X[members[s]], spec, labels=y[members[s]]).columns
                                     for s in args.splits)])
            rows.append(dict(dataset=args.dataset, rule=rule, k=k, macro_f1=f1, f1_vs_all=f1 / all_f1,
                             variance_explained=cov, jaccard_members_only=jac))
            print("  " + "  ".join(f"{a}={b:.3f}" if isinstance(b, float) else f"{a}={b}"
                                   for a, b in rows[-1].items()), flush=True)
    df = pd.DataFrame(rows)
    if OUT.exists():
        df = pd.concat([pd.read_csv(OUT), df]).drop_duplicates(["dataset", "rule", "k"], keep="last")
    df.to_csv(OUT, index=False)


if __name__ == "__main__":
    main()
