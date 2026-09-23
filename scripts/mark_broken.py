#!/usr/bin/env python
"""Stamp every artifact built from a known-broken target, so it cannot be misread.

    python scripts/mark_broken.py            # idempotent

The rule lives in `mia.targets.broken_reason` (currently: DP-PGM targets with
binning=dp_* and the biased 'clip' edge estimator, all built before generator
commit cd5d1d5 on 2026-09-23).  This script only makes it visible on disk:

  * target dirs:  BROKEN_DP_EDGES.txt, and meta.json gains "status"/"broken_reason";
  * run dirs:     BROKEN_DP_EDGES.txt, and config.json's notes gain a
                  "BROKEN_DP_EDGES; " prefix (so scripts/reindex.py keeps it);
  * results/index.csv: the same notes prefix, rewritten under the index lock;
  * results/pgm_eps_sweep.csv, results/pgm_attack_binning.csv: a `status` column.

`mia.targets.load_target`/`build_target` already refuse these targets, and
`mia.runs.load_index()` adds a computed `status` column; this is belt and braces
for anyone reading the files directly.
"""
import csv
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from mia import csvlock, paths  # noqa: E402
from mia import targets as T  # noqa: E402

TAG = T.BROKEN_DP_EDGES
PREFIX = TAG + "; "


def stamp_dir(d: Path, reason: str) -> None:
    (d / f"{TAG}.txt").write_text(
        f"{reason}\n\nDo not use. See results/BROKEN.md and FINDINGS 10i.\n")


def main():
    n_t = 0
    for meta in sorted((ROOT / "artifacts" / "targets").glob("*/pgm/*/split_*/meta.json")):
        name = "pgm@" + meta.parent.parent.name
        reason = T.broken_reason(name)
        if not reason:
            continue
        m = json.loads(meta.read_text())
        m["status"], m["broken_reason"] = TAG, reason
        meta.write_text(json.dumps(m, indent=2, default=str))
        stamp_dir(meta.parent, reason)
        n_t += 1

    n_r = 0
    for cfg in sorted((ROOT / "results" / "runs").glob("*/config.json")):
        c = json.loads(cfg.read_text())
        reason = T.broken_reason(c.get("generator", ""))
        if not reason:
            continue
        if not str(c.get("notes", "")).startswith(PREFIX):
            c["notes"] = PREFIX + str(c.get("notes", ""))
            cfg.write_text(json.dumps(c, indent=2, default=str))
        stamp_dir(cfg.parent, reason)
        n_r += 1

    n_i = 0
    with csvlock.atomic_update(paths.INDEX_CSV) as tmp:
        with open(paths.INDEX_CSV, newline="") as f:
            rd = csv.DictReader(f)
            cols, rows = rd.fieldnames, list(rd)
        for r in rows:
            if T.broken_reason(r["generator"]) and not r["notes"].startswith(PREFIX):
                r["notes"] = PREFIX + r["notes"]
                n_i += 1
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)

    for name in ("pgm_eps_sweep.csv", "pgm_attack_binning.csv"):
        p = ROOT / "results" / name
        with csvlock.atomic_update(p) as tmp:
            df = pd.read_csv(p)
            df["status"] = [TAG if T.broken_reason(t) else "ok" for t in df["target"]]
            df.to_csv(tmp, index=False)
        print(f"{name}: {(df.status == TAG).sum()} of {len(df)} rows {TAG}")
    print(f"targets stamped: {n_t}; runs stamped: {n_r}; index notes updated: {n_i}")


if __name__ == "__main__":
    main()
