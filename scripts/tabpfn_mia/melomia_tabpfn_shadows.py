"""MeLoMIA-ND with TabPFN base shadows: set up the BRCA_G200 arm.

The ablation separates what ICL-MIA changes relative to MeLoMIA-ND:

    shadow generator    probe          attack
    NoisyDiffusion      NoisyDiffusion MeLoMIA-ND (base_generator=None)
    TabPFN (target's)   NoisyDiffusion MeLoMIA-ND (base_generator="tabpfn")   <- this
    TabPFN (target's)   TabPFN, in context  ICL-MIA + LiRA

all on BRCA_G200 (the first 200 genes; configs/datasets/BRCA_G200.yaml).

This script
  1. writes the target releases cut to their first 200 columns as BRCA_G200
     targets (exactly what the generator would have produced for 200 genes);
  2. writes the 30 TabPFN shadow releases from shadows.py into MeLoMIA's cache
     for base_generator="tabpfn": shadow_splits.json (who was in each shadow
     context) and internal_synth/shadow_<k>.npz (the shadow release).  MeLoMIA
     builds a base shadow only when that file is missing, so it then trains its
     NoisyDiffusion synth-shadows, features and meta-classifier on these.
Source artifacts/pgg/env.sh first.  Main environment (camda-mia).
"""

from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from mia import datasets as D, paths  # noqa: E402
from mia.attacks import build  # noqa: E402

DS = "BRCA_G200"
PGG = Path("~/privacy/Pro-GENE-GEN-MPC/models/results_copula/TABPFN/BRCA/tabpfn_35").expanduser()


def write_targets(name="tabpfn"):
    genes = list(D.load_expression(DS).columns)
    cfg = json.loads((PGG / "config.json").read_text())
    for k in range(1, 6):
        d = paths.target_dir(DS, name, k)
        d.mkdir(parents=True, exist_ok=True)
        X = pd.read_csv(PGG / f"synthetic_data_split_{k}.csv")
        assert list(X.columns[:200]) == genes
        X[genes].to_csv(d / "synthetic_data.csv", index=False)
        pd.read_csv(PGG / f"synthetic_labels_split_{k}.csv").to_csv(d / "synthetic_labels.csv", index=False)
        fold = cfg["folds"][str(k)]
        (d / "meta.json").write_text(json.dumps({
            "params": fold.get("params"), "seed": fold.get("params", {}).get("seed"),
            "source": "Pro-GENE-GEN-MPC tabpfn_35, first 200 genes"}, indent=1, default=str))
    print(f"targets: {paths.target_dir(DS, name, 1).parent}")


def inject_shadows(shadow_dir=REPO / "results/tabpfn_mia/shadows"):
    atk = build("melomia_nd", base_generator="tabpfn", n_shadows=30, n_noise=600)
    cache = atk.cache(DS)
    files = sorted(glob.glob(f"{shadow_dir}/shadow_BRCA_g200_k*.npz"))
    assert len(files) == 30, len(files)
    ids_cohort = list(D.load_expression(DS).index)
    splits = {}
    (cache / "internal_synth").mkdir(parents=True, exist_ok=True)
    for f in files:
        z = np.load(f)
        k = int(f[-7:-4]) + 1                         # MeLoMIA numbers shadows from 1
        ids, inn = list(z["ids"]), z["member"].astype(bool)
        assert set(ids) == set(ids_cohort)
        splits[f"shadow_{k}"] = {"member_ids": sorted(np.array(ids)[inn].tolist()),
                                 "nonmember_ids": sorted(np.array(ids)[~inn].tolist())}
        np.savez(cache / "internal_synth" / f"shadow_{k}.npz",
                 X=z["S"].astype(np.float32), y=z["yS"].astype(np.int64))
    (cache / "shadow_splits.json").write_text(json.dumps(splits, indent=2))
    print(f"injected {len(files)} TabPFN shadow releases into {cache}")


if __name__ == "__main__":
    write_targets()
    inject_shadows()
