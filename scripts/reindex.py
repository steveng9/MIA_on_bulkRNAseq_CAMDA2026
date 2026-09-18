#!/usr/bin/env python
"""Rebuild results/index.csv from the run directories, or prune stale runs.

    python scripts/reindex.py
    python scripts/reindex.py --prune-attack mahalamia --dataset BRCA
    python scripts/reindex.py --prune-attack mahalamia --dataset BRCA --yes

Run ids are hashes of the attack configuration, so adding a parameter to an
attack orphans its earlier runs.  Prune them rather than letting the grid
average over two different versions of the same attack.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mia import runs as R  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prune-attack", default=None)
    p.add_argument("--dataset", default=None)
    p.add_argument("--yes", action="store_true", help="actually delete")
    args = p.parse_args()

    if args.prune_attack:
        hits = R.prune(args.dataset, args.prune_attack, dry_run=not args.yes)
        verb = "Deleted" if args.yes else "Would delete"
        print(f"{verb} {len(hits)} runs")
        for h in hits[:20]:
            print(f"  {h}")
        if not args.yes:
            print("\nRe-run with --yes to delete.")
            return

    df = R.rebuild_index()
    print(f"Indexed {len(df)} runs -> {R.paths.INDEX_CSV}")


if __name__ == "__main__":
    main()
