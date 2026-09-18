#!/usr/bin/env python
"""Negative controls for recorded runs.

    python scripts/sanity_check.py --dataset BRCA

A membership attack that reports a high AUC is claiming something strong, and
the most common reason for an implausibly high number is not a good attack but a
leak — labels reaching the scorer, or scores accidentally aligned to the wrong
thing.  These checks read the row-level scores back off disk and look for the
signatures of that:

  shuffled       scores against permuted labels must fall to chance.  Averaged
                 over several permutations, because a single one has a standard
                 error around 0.02 at these sample sizes and would flag healthy
                 runs a few percent of the time.  If the average does not sit at
                 chance, the metric or the alignment is broken, not the attack.
  cross-split    scores from a target trained on split i, evaluated against
                 split j's labels.  Should be near or below chance: a sample
                 that is a member of split i is usually also a member of split
                 j, but the fifth that differ pull the ordering apart.  A high
                 value here means the attack is reading something about the
                 sample rather than about its membership.
  degenerate     constant or near-constant score vectors, which produce
                 meaningless metrics rather than an error.

Failures print with the run id.  Nothing here proves an attack is correct; it
only catches the ways a wrong one looks right.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mia import datasets as D  # noqa: E402
from mia import metrics as M  # noqa: E402
from mia import paths, runs as R  # noqa: E402

SHUFFLE_TOLERANCE = 0.03     # how far from 0.5 the mean permuted-label AUC may sit
N_PERMUTATIONS = 20
CROSS_SPLIT_CEILING = 0.70   # above this, the score is tracking the sample


def check_run(run_id: str, rng) -> list:
    d = paths.RUNS_DIR / run_id
    scores = pd.read_csv(d / "scores.csv")
    s = scores.score.values.astype(float)
    y = scores.y_member.values.astype(int)
    problems = []

    if np.allclose(s, s[0]):
        problems.append("degenerate: every score identical")
        return problems
    if len(np.unique(y)) < 2:
        problems.append("degenerate: labels are all one class")
        return problems

    shuffled = float(np.mean([M.evaluate(rng.permutation(y), s)["auc"]
                              for _ in range(N_PERMUTATIONS)]))
    if abs(shuffled - 0.5) > SHUFFLE_TOLERANCE:
        problems.append(f"mean shuffled-label AUC over {N_PERMUTATIONS} "
                        f"permutations is {shuffled:.3f}, not near chance")

    import json
    cfg = json.loads((d / "config.json").read_text())
    dataset, split = cfg["dataset"], cfg["split"]
    if split is not None:
        for other in D.load_target_splits(dataset):
            if other == split:
                continue
            auc = M.evaluate(D.membership_labels(dataset, other), s)["auc"]
            if auc > CROSS_SPLIT_CEILING:
                problems.append(
                    f"cross-split AUC {auc:.3f} against split {other} labels "
                    "-- scores may track the sample, not its membership"
                )
            break
    return problems


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=None)
    p.add_argument("--attack", default=None)
    args = p.parse_args()

    df = R.load_index()
    if args.dataset:
        df = df[df.dataset == args.dataset]
    if args.attack:
        df = df[df.attack == args.attack]
    if df.empty:
        raise SystemExit("No matching runs.")

    rng = np.random.default_rng(0)
    failures = 0
    for run_id in df.run_id:
        problems = check_run(run_id, rng)
        if problems:
            failures += 1
            print(f"\n{run_id}")
            for msg in problems:
                print(f"  ! {msg}")

    print(f"\nChecked {len(df)} runs, {failures} with findings.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
