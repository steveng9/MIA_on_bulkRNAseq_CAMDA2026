#!/usr/bin/env python
"""Drop the fitted graphical model from finished PGM target checkpoints.

    python scripts/slim_pgm_checkpoints.py            # dry run
    python scripts/slim_pgm_checkpoints.py --apply

Keeps every other field of generator.pt (bin edges, chosen tables, rho
breakdown, selection diagnostics), which is all any attack or diagnostic reads.
Only touches targets whose synthetic data and meta.json exist and whose
checkpoint is at least 10 minutes old (so fits in flight are left alone).
Appends each slimmed path to artifacts/targets/SLIMMED.txt.
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mia import paths  # noqa: E402
from mia.generators.pgm import _import_upstream  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--min-mb", type=float, default=5.0)
    a = ap.parse_args()
    _import_upstream()
    log = paths.TARGETS_DIR / "SLIMMED.txt"
    freed = n = 0
    for p in sorted(paths.TARGETS_DIR.glob("*/pgm/*/split_*/generator.pt")):
        size = p.stat().st_size
        d = p.parent
        if (size < a.min_mb * 1e6 or time.time() - p.stat().st_mtime < 600
                or not (d / "synthetic_data.csv").exists() or not (d / "meta.json").exists()):
            continue
        n += 1
        if a.apply:
            g = pickle.load(open(p, "rb"))
            g._joint_fitter = None
            g._class_fitters = {}
            tmp = p.with_suffix(".slim.tmp")
            with open(tmp, "wb") as f:
                pickle.dump(g, f)
            tmp.replace(p)
            with open(log, "a") as f:
                f.write(f"{time.strftime('%F %T')}\t{size}\t{p.relative_to(paths.TARGETS_DIR)}\n")
            freed += size - p.stat().st_size
        else:
            freed += size
    print(f"{'slimmed' if a.apply else 'would slim'} {n} checkpoints, {freed / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
