"""Record the search protocol in MeLoMIA meta.json files that predate protocol records.

A cached meta-classifier is keyed by its tag, and the tag does not include the search
budget.  From 2026-10-03 `MeLoMIA._ensure_meta` records the protocol at fit time and
refuses a cache fitted under another.  For older caches the protocol is recovered from
the EARLIEST recorded run with that (dataset, attack, tag) -- the run that fitted it --
and marked as backfilled.  Later runs under the same tag that asked for different
settings (and silently got the cached classifier) are listed.

    python scripts/melomia_backfill_protocol.py            # write
    python scripts/melomia_backfill_protocol.py --dry-run
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("n_shadows", "classifiers", "optuna_enabled", "optuna_trials", "cv_folds",
          "ensemble_temperature", "ensemble_min_auc", "internal_proxy_selection",
          "synth_shadow", "reference_calibration", "per_record_calibration", "seed")
# fields added after the run was recorded, with the value the code then used
DEFAULTS = {"per_record_calibration": False, "internal_proxy_selection": False}
dry = "--dry-run" in sys.argv

idx = pd.read_csv(ROOT / "results" / "index.csv").sort_values("timestamp")


def protocol(run_id):
    p = json.loads((ROOT / "results" / "runs" / run_id / "config.json").read_text())["params"]
    return {k: p.get(k, DEFAULTS.get(k)) for k in FIELDS}


for mp in sorted((ROOT / "artifacts" / "attacks").glob("melomia_*/*/*/meta/*/meta.json")):
    attack, ds, _, _, tag = mp.relative_to(ROOT / "artifacts" / "attacks").parts[:5]
    meta = json.loads(mp.read_text())
    rel = f"{attack}/{ds}/{tag}"
    if meta.get("protocol"):
        print(f"ok        {rel}: {meta.get('protocol_source')}")
        continue
    runs = idx[(idx.dataset == ds) & (idx.attack == attack) & (idx.tag == tag)]
    runs = runs[[(ROOT / "results" / "runs" / r / "config.json").exists() for r in runs.run_id]]
    if runs.empty:
        print(f"NO RUN    {rel}: no recorded run, protocol unknown")
        continue
    first = runs.iloc[0]
    proto = protocol(first.run_id)
    other = {}
    for exp, r in runs.groupby("experiment").run_id.first().items():
        d = {k: v for k, v in protocol(r).items() if v != proto[k]}
        if d:
            other[exp] = d
    print(f"backfill  {rel}: trials={proto['optuna_trials']} from {first.experiment} "
          f"({first.timestamp[:10]})" + (f"  LATER RUNS ASKED FOR {other}" if other else ""))
    if not dry:
        meta["protocol"] = proto
        meta["protocol_source"] = f"backfilled from run {first.run_id} ({first.experiment})"
        mp.write_text(json.dumps(meta, indent=2, default=str))
