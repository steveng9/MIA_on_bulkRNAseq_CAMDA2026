"""GSE58135 (Varley et al. 2014) as an external auxiliary set for TCGA-BRCA.

84 primary breast tumours (42 triple-negative, 42 ER+/HER2-) from the NCBI
recount of the series, cut to the 978 landmark genes in the challenge's gene
order.  Run with the pydeseq2 virtualenv (~/.venvs/pydeseq2/bin/python).

Writes samples x genes TSVs to ~/data/GSE58135/processed/:

  aux_tpm.tsv         NCBI TPM as distributed (Hakime's condition 1)
  aux_log2tpm.tsv     log2(TPM + 1)
  aux_vst_own.tsv     a VST fitted on these 84 samples alone, with the
                      challenge's recipe (>=1 count in >=10% of samples,
                      design ~ subtype, parametric trend) -- condition 2
  aux_vst_frozen.tsv  the TCGA cohort's own fitted VST applied to these counts
                      (only with --tcga-fit, see tcga_vst.py)
  labels.csv          GSM, receptor group
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vst as VST  # noqa: E402

ROOT = Path("~/data/GSE58135").expanduser()
CHALLENGE = Path("~/data/CAMDA26/RED_TCGA-BRCA/"
                 "TCGA-BRCA_primary_tumor_star_deseq_VST_lmgenes.tsv").expanduser()


def sample_table() -> pd.DataFrame:
    rows, cur = [], None
    for line in open(ROOT / "GSE58135_samples_soft.txt"):
        if line.startswith("^SAMPLE"):
            cur = {"gsm": line.split("=")[1].strip()}
            rows.append(cur)
        elif line.startswith("!Sample_source_name_ch1"):
            cur["source"] = line.split("=", 1)[1].strip()
    m = pd.DataFrame(rows).set_index("gsm")
    m = m[m.source.isin(["ER+ Breast Cancer Primary Tumor",
                         "Triple Negative Breast Cancer Primary Tumor"])]
    m["group"] = np.where(m.source.str.startswith("ER+"), "ERpos_HER2neg", "TNBC")
    return m


def own_vst(counts: pd.DataFrame, groups: pd.Series) -> tuple:
    """pydeseq2 fit on genes x samples counts; returns (vst frame, fit record)."""
    from pydeseq2.dds import DeseqDataSet
    keep = VST.gene_filter(counts.values)
    c = counts.loc[keep]
    dds = DeseqDataSet(counts=c.T, metadata=pd.DataFrame({"group": groups}),
                       design="~group", quiet=True)
    dds.vst_fit(use_design=True)
    out = pd.DataFrame(dds.vst_transform(), index=c.columns, columns=c.index)
    coeffs = np.asarray(dds.uns["vst_trend_coeffs"], dtype=float)
    return out, {"genes_kept": int(keep.sum()), "trend_coeffs": coeffs.tolist(),
                 "size_factors": dict(zip(c.columns, map(float, dds.obs["size_factors"])))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tcga-fit", default=None, help="JSON written by tcga_vst.py")
    args = ap.parse_args()
    out = ROOT / "processed"
    out.mkdir(exist_ok=True)

    meta = sample_table()
    assert len(meta) == 84 and (meta.group == "TNBC").sum() == 42
    lm = pd.read_csv(ROOT / "f1000_lm_ensembl.tsv")
    entrez = dict(zip(lm["ENSEMBL.ID"], lm["Entrez.ID"]))
    genes = list(pd.read_csv(CHALLENGE, sep="\t", index_col=0, usecols=[0]).index)

    counts = pd.read_csv(ROOT / "GSE58135_raw_counts_GRCh38.p13_NCBI.tsv.gz", sep="\t", index_col=0)
    tpm = pd.read_csv(ROOT / "GSE58135_norm_counts_TPM_GRCh38.p13_NCBI.tsv.gz", sep="\t", index_col=0)
    counts, tpm = counts[meta.index], tpm[meta.index]
    ids = [entrez[g] for g in genes]

    def landmark(df):  # genes x samples (Entrez) -> samples x genes (Ensembl, challenge order)
        sub = df.loc[ids].T
        sub.columns = genes
        return sub

    meta[["group"]].to_csv(out / "labels.csv")
    landmark(tpm).to_csv(out / "aux_tpm.tsv", sep="\t")
    np.log2(landmark(tpm) + 1).to_csv(out / "aux_log2tpm.tsv", sep="\t")

    vst_all, fit = own_vst(counts, meta.group)
    missing = [g for g in ids if g not in vst_all.columns]
    if missing:
        raise SystemExit(f"{len(missing)} landmark genes fail the low-count filter")
    own = vst_all[ids]
    own.columns = genes
    own.to_csv(out / "aux_vst_own.tsv", sep="\t")
    (out / "aux_vst_own_fit.json").write_text(json.dumps(fit, indent=1))
    print(f"own VST: {fit['genes_kept']} genes kept, trend {fit['trend_coeffs']}, "
          f"range {own.values.min():.2f}..{own.values.max():.2f}")

    if args.tcga_fit:
        f = json.loads(Path(args.tcga_fit).read_text())
        # Size factor against the TCGA cohort's geometric means, over the genes
        # the two annotations share (Entrez id -> Ensembl id, NCBI gene_info).
        ref = pd.Series(f["loggeo"])
        info = pd.read_csv(ROOT / "Homo_sapiens.gene_info.gz", sep="\t", usecols=["GeneID", "dbXrefs"])
        ens = info.set_index("GeneID").dbXrefs.str.extract(r"Ensembl:(ENSG\d+)")[0].dropna()
        ens = ens[~ens.duplicated(keep=False)]
        shared = [e for e in counts.index if ens.get(e) in ref.index]
        lg = ref.loc[[ens[e] for e in shared]].values
        sf = VST.size_factors(counts.loc[shared].values, lg)
        q = landmark(counts).values / sf[:, None]
        frozen = pd.DataFrame(VST.vst(q, f["asympt_disp"], f["extra_pois"]),
                              index=meta.index, columns=genes)
        frozen.to_csv(out / "aux_vst_frozen.tsv", sep="\t")
        print(f"frozen VST: {len(shared)} shared genes for size factors, "
              f"range {frozen.values.min():.2f}..{frozen.values.max():.2f}")


if __name__ == "__main__":
    main()
