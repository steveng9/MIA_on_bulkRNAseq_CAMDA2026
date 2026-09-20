#!/usr/bin/env python
"""Score the released synthetic datasets as *data*, not as leaks.

    python scripts/eval_fidelity.py --dataset BRCA
    python scripts/eval_fidelity.py --dataset BRCA --generators pgm --splits 1

Writes one row per (dataset, generator, split) to results/fidelity.csv, which is
the objective the DP-PGM tuning sweep optimises against.  Rows accumulate and
are de-duplicated on (dataset, generator, split, config), so re-running after a
generator change replaces that generator's rows and leaves the rest alone.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mia import datasets as D  # noqa: E402
from mia import fidelity as F  # noqa: E402
from mia import targets as T  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "results" / "fidelity.csv"


def one(dataset: str, generator: str, split: int, seed: int, max_genes: int) -> dict:
    X_all = D.load_expression(dataset).values.astype(np.float64)
    y_all = D.encode_subtypes(dataset, D.load_subtypes(dataset).values)
    member = D.membership_labels(dataset, split).astype(bool)

    tgt = T.load_target(dataset, generator, split)
    row = F.evaluate(tgt["X"], tgt["y_int"], X_all[~member], y_all[~member],
                     X_all[member], y_all[member], seed=seed, max_genes=max_genes)
    row.update(dataset=dataset, generator=generator, split=split,
               n_syn=len(tgt["X"]), n_train=int(member.sum()),
               config=json.dumps(tgt["meta"].get("params", {}), sort_keys=True))
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="BRCA")
    p.add_argument("--generators", nargs="+", default=list(T.GENERATORS))
    p.add_argument("--splits", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    p.add_argument("--max-genes", type=int, default=300,
                   help="variance-ranked subset for the O(g^2) metrics")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rows = []
    for gen in args.generators:
        for split in args.splits:
            if not T.exists(args.dataset, gen, split):
                print(f"  {gen} split {split}: not built, skipping", flush=True)
                continue
            row = one(args.dataset, gen, split, args.seed, args.max_genes)
            rows.append(row)
            print(f"  {gen} split {split}: "
                  f"TSTR-F1={row['tstr_macro_f1']:.3f} "
                  f"(real {row['real_macro_f1']:.3f}, ratio {row['utility_ratio']:.3f})  "
                  f"disc-AUC={row['discriminator_auc']:.3f}  "
                  f"W1={row['wasserstein_mean']:.3f}  "
                  f"corr-MAE={row['corr_mae']:.3f}", flush=True)

    if not rows:
        print("nothing to score")
        return
    df = pd.DataFrame(rows)
    if OUT.exists():
        df = pd.concat([pd.read_csv(OUT), df], ignore_index=True)
    key = ["dataset", "generator", "split", "config"]
    df = df.drop_duplicates(subset=key, keep="last").sort_values(key)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\n{len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
