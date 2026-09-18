#!/usr/bin/env python
"""Run an experiment defined by a YAML config.

    python scripts/run_experiment.py configs/experiments/grid_brca.yaml
    python scripts/run_experiment.py configs/.../grid_brca.yaml --only mahalamia mamamia
    python scripts/run_experiment.py configs/.../grid_brca.yaml --build-targets

Results land in results/runs/<run_id>/ and results/index.csv.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mia.experiment import Experiment  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config")
    p.add_argument("--only", nargs="+", help="attack labels to run (default: all)")
    p.add_argument("--generators", nargs="+")
    p.add_argument("--splits", nargs="+", type=int)
    p.add_argument("--build-targets", action="store_true")
    p.add_argument("--prepare-only", action="store_true",
                   help="build the attack's shadow stack and stop before scoring")
    p.add_argument("--shadows", nargs="+", type=int, default=None,
                   help="with --prepare-only: build only these shadow indices, so "
                        "several processes can share one cache across GPUs")
    p.add_argument("--no-meta", action="store_true",
                   help="with --prepare-only: skip meta-classifier training "
                        "(use when another process will do it)")
    p.add_argument("--device", default=None)
    p.add_argument("--no-save", action="store_true")
    args = p.parse_args()

    exp = Experiment.load(args.config)
    if args.device:
        exp.device = args.device

    if args.build_targets:
        exp.build_targets()

    if args.prepare_only:
        for label in (args.only or list(exp.attacks)):
            print(f"\n=== preparing {label} ===", flush=True)
            exp.build_attack(label).prepare(
                exp.dataset, shadows=args.shadows, meta=not args.no_meta
            )
        return

    exp.run(only=args.only, generators=args.generators, splits=args.splits,
            save=not args.no_save)


if __name__ == "__main__":
    main()
