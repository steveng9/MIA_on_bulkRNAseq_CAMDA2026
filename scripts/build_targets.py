#!/usr/bin/env python
"""Build the target synthetic datasets that every attack is scored against.

    python scripts/build_targets.py --dataset BRCA
    python scripts/build_targets.py --dataset BRCA --generators pgm --force

One target per (generator, split).  Idempotent: already-built targets are left
alone unless --force is given.  The ND column reuses the blue team's published
synthetic data unless --retrain-nd is passed.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mia import targets as TG  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="BRCA", choices=["BRCA", "COMBINED"])
    p.add_argument("--generators", nargs="+", default=list(TG.GENERATORS))
    p.add_argument("--splits", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    p.add_argument("--device", default="cuda")
    p.add_argument("--force", action="store_true")
    p.add_argument("--retrain-nd", action="store_true",
                   help="train our own ND targets instead of reusing the published ones")
    args = p.parse_args()

    for gen in args.generators:
        for split in args.splits:
            t = time.time()
            TG.build_target(args.dataset, gen, split, device=args.device,
                            force=args.force, retrain_nd=args.retrain_nd)
            print(f"  [{gen} split {split}] {time.time() - t:.1f}s", flush=True)


if __name__ == "__main__":
    main()
