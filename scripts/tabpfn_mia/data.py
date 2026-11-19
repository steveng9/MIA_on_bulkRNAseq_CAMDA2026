"""Load a TabPFN target as built in the Pro-GENE-GEN-MPC tree.

That tree holds the canonical TCGA expression files and the gene-by-gene
TabPFN-3.5 targets (built with this repository's `TabPFNGenerator`), in its
own layout rather than this repository's `artifacts/`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("PGG_ROOT", "~/privacy/Pro-GENE-GEN-MPC")).expanduser()
COHORT = {"BRCA": "TCGA-BRCA", "COMBINED": "TCGA-COMBINED"}
TARGET_DIR = {"BRCA": "BRCA", "COMBINED": "COMB"}
SUBTYPES = Path("~/PRO-GENE-GEN/data/aml").expanduser()


@dataclass
class Target:
    X: np.ndarray            # (n_candidates, n_genes) real candidates
    y: np.ndarray            # (n_candidates,) subtype codes
    member: np.ndarray       # (n_candidates,) 1 = in the generator's context
    S: np.ndarray            # (n_syn, n_genes) release
    yS: np.ndarray           # (n_syn,) release subtype codes
    ids: list
    genes: list
    classes: list


def load(dataset: str, split: int, run: str = "tabpfn_35") -> Target:
    full = COHORT[dataset]
    expr = pd.read_csv(ROOT / f"data/aml/BLUE_{full}/{full}_primary_tumor_star_deseq_VST_lmgenes.tsv",
                       sep="\t", index_col=0).T
    sub = pd.read_csv(SUBTYPES / f"BLUE_{full}/{full}_primary_tumor_subtypes.csv",
                      index_col="samplesID")["Subtype"]
    tdir = ROOT / "models/results_copula/TABPFN" / TARGET_DIR[dataset] / run
    mem = pd.read_csv(tdir / f"membership_labels_split_{split}.csv", index_col="sample_id")
    ids = list(mem.index)
    S = pd.read_csv(tdir / f"synthetic_data_split_{split}.csv")
    assert list(S.columns) == list(expr.columns), "release genes differ from the cohort's"
    yS_str = pd.read_csv(tdir / f"synthetic_labels_split_{split}.csv").iloc[:, 0].astype(str).values
    classes = sorted(sub.unique())
    code = {c: i for i, c in enumerate(classes)}
    return Target(
        X=expr.loc[ids].values.astype(np.float32),
        y=np.array([code[s] for s in sub.loc[ids]]),
        member=mem["membership_label"].values.astype(int),
        S=S.values.astype(np.float32),
        yS=np.array([code[s] for s in yS_str]),
        ids=ids, genes=list(expr.columns), classes=classes,
    )
