"""Make the Pro-GENE-GEN-MPC data and TabPFN releases readable by `mia.*`.

This host has neither ~/data/CAMDA26 nor the NoisyDiffusion repo nor
`artifacts/`.  The repository's loaders only need files under the expected
names, so this builds a link tree under artifacts/pgg/ (git-ignored):

    shim/data/RED_TCGA-BRCA/<expression tsv>     -> Pro-GENE-GEN-MPC copy
    shim/data/BLUE_TCGA-BRCA.zip                 (holds the subtype CSV)
    shim/nd/TCGA-BRCA/TCGA-BRCA_splits.yaml      -> Pro-GENE-GEN-MPC copy
    targets/BRCA/<name>/split_k/synthetic_{data,labels}.csv, meta.json

and prints the environment to export.  Run records then go to their own
results directory (CAMDA_RESULTS), not the repository's results/index.csv.

    python scripts/tabpfn_mia/pgg_shim.py [--run tabpfn_35] [--name tabpfn]
"""

from __future__ import annotations

import argparse
import json
import os
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PGG = Path("~/privacy/Pro-GENE-GEN-MPC").expanduser()
SUB = Path("~/PRO-GENE-GEN/data/aml/BLUE_TCGA-BRCA/TCGA-BRCA_primary_tumor_subtypes.csv").expanduser()


def link(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="tabpfn_35", help="run directory under TABPFN/BRCA")
    ap.add_argument("--name", default="tabpfn", help="target name inside the shim")
    ap.add_argument("--root", default=str(REPO / "artifacts" / "pgg"))
    ap.add_argument("--results", default=str(REPO / "results" / "tabpfn_mia" / "melomia_runs"))
    a = ap.parse_args()
    root = Path(a.root)

    data = root / "shim" / "data"
    link(PGG / "data/aml/BLUE_TCGA-BRCA/TCGA-BRCA_primary_tumor_star_deseq_VST_lmgenes.tsv",
         data / "RED_TCGA-BRCA" / "TCGA-BRCA_primary_tumor_star_deseq_VST_lmgenes.tsv")
    with zipfile.ZipFile(data / "BLUE_TCGA-BRCA.zip", "w") as zf:
        zf.write(SUB, "BLUE_TCGA-BRCA/TCGA-BRCA_primary_tumor_subtypes.csv")
    link(PGG / "models/results_copula/splits/TCGA-BRCA_splits.yaml",
         root / "shim" / "nd" / "TCGA-BRCA" / "TCGA-BRCA_splits.yaml")

    src = PGG / "models/results_copula/TABPFN/BRCA" / a.run
    cfg = json.loads((src / "config.json").read_text())
    for k in range(1, 6):
        d = root / "targets" / "BRCA" / a.name / f"split_{k}"
        link(src / f"synthetic_data_split_{k}.csv", d / "synthetic_data.csv")
        link(src / f"synthetic_labels_split_{k}.csv", d / "synthetic_labels.csv")
        fold = cfg["folds"][str(k)]
        (d / "meta.json").write_text(json.dumps({
            "params": fold.get("params"), "seed": fold.get("params", {}).get("seed"),
            "source": f"Pro-GENE-GEN-MPC {a.run}", **{x: v for x, v in fold.items() if x != "params"}},
            indent=1, default=str))

    env = {"CAMDA_ARTIFACTS": str(root), "CAMDA_DATA": str(data),
           "CAMDA_ND_REPO": str(root / "shim" / "nd"), "CAMDA_RESULTS": a.results}
    (root / "env.sh").write_text("".join(f"export {k}={v}\n" for k, v in env.items()))
    print(f"wrote {root}/env.sh:\n" + (root / "env.sh").read_text())


if __name__ == "__main__":
    main()
