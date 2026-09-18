"""Experiment record keeping.

Every attack evaluation writes exactly one run directory:

    results/runs/<run_id>/
        config.json    fully resolved parameters (what was run)
        scores.csv     sample_id, score, y_member  -- row level, never overwritten
        metrics.json   auc, aupr, tpr@fpr, ...

plus one line appended to results/index.csv, which is the flat table to load
with pandas when building figures or paper tables.

`run_id` is deterministic: the same configuration always maps to the same
directory, so re-running an experiment updates it in place instead of
accumulating near-duplicates.  The trailing hash covers every parameter that is
not already in the readable prefix, so two runs that differ only in, say, shadow
count get different directories.
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import paths

INDEX_COLUMNS = [
    "run_id", "timestamp", "dataset", "attack", "generator", "split",
    "auc", "aupr", "tpr_at_fpr_0.01", "tpr_at_fpr_0.1",
    "precision_at_5pct", "n", "n_members", "tag", "notes",
]


def _hash(params: dict) -> str:
    blob = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:8]


def make_run_id(dataset: str, attack: str, generator: str, split, params: dict) -> str:
    split_part = "all" if split is None else f"s{split}"
    return f"{dataset}__{attack}__{generator}__{split_part}__{_hash(params)}"


def save_run(
    *,
    dataset: str,
    attack: str,
    generator: str,
    split,
    params: dict,
    sample_ids,
    scores,
    y_member,
    metrics: dict,
    tag: str = "",
    notes: str = "",
) -> Path:
    """Persist one attack evaluation and index it.  Returns the run directory."""
    run_id = make_run_id(dataset, attack, generator, split, params)
    out = paths.RUNS_DIR / run_id
    out.mkdir(parents=True, exist_ok=True)

    record = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": dataset,
        "attack": attack,
        "generator": generator,
        "split": split,
        "tag": tag,
        "notes": notes,
        "params": params,
    }
    (out / "config.json").write_text(json.dumps(record, indent=2, default=str))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    pd.DataFrame(
        {
            "sample_id": list(sample_ids),
            "score": np.asarray(scores, dtype=float),
            "y_member": np.asarray(y_member, dtype=int),
        }
    ).to_csv(out / "scores.csv", index=False)

    _append_index(record, metrics)
    return out


def _append_index(record: dict, metrics: dict) -> None:
    paths.RESULTS.mkdir(parents=True, exist_ok=True)
    row = {c: "" for c in INDEX_COLUMNS}
    row.update({k: record.get(k, "") for k in
                ("run_id", "timestamp", "dataset", "attack", "generator", "split", "tag", "notes")})
    for k in ("auc", "aupr", "tpr_at_fpr_0.01", "tpr_at_fpr_0.1",
              "precision_at_5pct", "n", "n_members"):
        if k in metrics:
            row[k] = metrics[k]

    existing = []
    if paths.INDEX_CSV.exists():
        with open(paths.INDEX_CSV, newline="") as f:
            existing = [r for r in csv.DictReader(f) if r.get("run_id") != row["run_id"]]

    with open(paths.INDEX_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_COLUMNS)
        w.writeheader()
        for r in existing:
            w.writerow({c: r.get(c, "") for c in INDEX_COLUMNS})
        w.writerow(row)


def load_index() -> pd.DataFrame:
    if not paths.INDEX_CSV.exists():
        return pd.DataFrame(columns=INDEX_COLUMNS)
    return pd.read_csv(paths.INDEX_CSV)


def load_run(run_id: str) -> dict:
    out = paths.RUNS_DIR / run_id
    return {
        "config": json.loads((out / "config.json").read_text()),
        "metrics": json.loads((out / "metrics.json").read_text()),
        "scores": pd.read_csv(out / "scores.csv"),
    }
