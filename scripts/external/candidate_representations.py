"""The challenge's own samples in other representations than the release.

For the experiment in which the attacked profiles themselves arrive in another
normalisation or from another pipeline (scripts/perturb/candidate_mismatch.py).
Every distributed sample of a cohort (BRCA: the 1,089 candidates; COMBINED:
the 4,323 candidates and the 824 reference samples), 978 landmark genes:

  <COHORT>_all_gdc_tpm.tsv       GDC's STAR pipeline, TPM (`tpm_unstranded`):
                                 the same reads and counts as the release,
                                 another normalisation
  <COHORT>_all_toil_log2tpm.tsv  UCSC TOIL recompute (STAR + RSEM, GENCODE
                                 v23), log2(TPM + 1): another pipeline
  <COHORT>_all_toil_vst_own.tsv  TOIL's RSEM expected counts through a VST
                                 fitted on these samples with the challenge's
                                 recipe (what the adversary computes when it
                                 knows the recipe but not the fitted
                                 transform; run with ~/.venvs/pydeseq2)

The TOIL files need ~/data/TOIL_Xena/raw (TcgaTargetGtex_rsem_gene_tpm.gz,
TcgaTargetGtex_gene_expected_count.gz).  TOIL names samples by their 15-character
barcode; a distributed sample TOIL lacks is left out of the TOIL files.

    python scripts/external/candidate_representations.py gdc
    python scripts/external/candidate_representations.py toil
    ~/.venvs/pydeseq2/bin/python scripts/external/candidate_representations.py toil_vst
"""

from __future__ import annotations

import gzip
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vst as VST  # noqa: E402

ROOT = Path("~/data/TCGA_GDC").expanduser()
TOIL = Path("~/data/TOIL_Xena/raw").expanduser()
OUT = ROOT / "processed"
CH = Path("~/data/CAMDA26").expanduser()


def distributed(name: str) -> tuple:
    """(sample ids, labels, 978 landmark gene ids) of a challenge cohort."""
    def read(suffix):
        f = CH / f"RED_TCGA-{name}" / f"TCGA-{name}_primary_tumor_star_deseq_VST_lmgenes{suffix}.tsv"
        return pd.read_csv(f, sep="\t", index_col=0, nrows=0).columns.tolist() if f.exists() else []
    ids = read("") + read("_reference")
    genes = pd.read_csv(CH / f"RED_TCGA-{name}" / f"TCGA-{name}_primary_tumor_star_deseq_VST_lmgenes.tsv",
                        sep="\t", index_col=0, usecols=[0]).index.tolist()
    man = pd.read_csv(ROOT / "gdc_star_counts_manifest.csv").set_index("aliquot")
    return ids, man.loc[ids, "project"].values, genes


def gdc() -> None:
    import tcga_vst
    gtab, meta, _, tpm = tcga_vst.assemble()
    ens = gtab.gene_id.str.replace(r"\..*$", "", regex=True).values
    col = {a: i for i, a in enumerate(meta.aliquot)}
    for name in ("BRCA", "COMBINED"):
        ids, _, genes = distributed(name)
        lm = [int(np.where(ens == g)[0][0]) for g in genes]
        T = pd.DataFrame(tpm[lm][:, [col[a] for a in ids]].T.astype(np.float64), index=ids, columns=genes)
        T.to_csv(OUT / f"{name}_all_gdc_tpm.tsv", sep="\t")
        print(f"{name}: {T.shape} GDC TPM", flush=True)


def _read_toil(path: Path, want: dict, rows=None) -> pd.DataFrame:
    """Rows (genes, version stripped) x wanted samples of a TOIL matrix, values as stored."""
    with gzip.open(path, "rt") as f:
        head = f.readline().rstrip("\n").split("\t")
        pos = [(i, want[s]) for i, s in enumerate(head) if s in want]
        idx, names = [i for i, _ in pos], [n for _, n in pos]
        genes, vals = [], []
        for line in f:
            parts = line.rstrip("\n").split("\t")
            g = parts[0].split(".")[0]
            if rows is not None and g not in rows:
                continue
            genes.append(g)
            vals.append(np.array([parts[i] for i in idx], dtype=np.float32))
    return pd.DataFrame(np.vstack(vals), index=genes, columns=names)


def toil() -> None:
    cohorts = {name: distributed(name) for name in ("BRCA", "COMBINED")}
    want = {a[:15]: a for ids, _, _ in cohorts.values() for a in ids}
    lm = {g for _, _, genes in cohorts.values() for g in genes}
    L_all = _read_toil(TOIL / "TcgaTargetGtex_rsem_gene_tpm.gz", want, rows=lm)
    C_all = _read_toil(TOIL / "TcgaTargetGtex_gene_expected_count.gz", want)
    for name, (ids, _, genes) in cohorts.items():
        L = L_all.loc[:, [a for a in ids if a in L_all.columns]]
        L = L[~L.index.duplicated()]
        # stored as log2(TPM + 0.001); the release-comparable form is log2(TPM + 1)
        X = np.log2(np.exp2(L.astype(np.float64)) - 0.001 + 1).clip(lower=0)
        X = X.reindex(genes).T
        lost = X.isna().any(axis=1)
        print(f"{name}: TOIL has {len(X) - int(lost.sum())} of {len(ids)} samples, "
              f"{int(L.index.isin(genes).sum())} of {len(genes)} genes", flush=True)
        X.loc[~lost].to_csv(OUT / f"{name}_all_toil_log2tpm.tsv", sep="\t")
        C = C_all.loc[:, [a for a in ids if a in C_all.columns]]
        C = C[~C.index.duplicated()]
        C = np.rint(np.exp2(C.astype(np.float64)) - 1).clip(lower=0).astype(np.int64)
        np.savez_compressed(OUT / f"{name}_all_toil_counts.npz", counts=C.values,
                            gene_ids=np.array(C.index), samples=np.array(C.columns))


def toil_vst() -> None:
    from pydeseq2.dds import DeseqDataSet
    for name in ("BRCA", "COMBINED"):
        ids, proj, genes = distributed(name)
        lab = dict(zip(ids, proj))
        d = np.load(OUT / f"{name}_all_toil_counts.npz", allow_pickle=True)
        counts = pd.DataFrame(d["counts"], index=d["gene_ids"], columns=d["samples"])
        keep = VST.gene_filter(counts.values)
        c = counts.loc[keep]
        if name == "BRCA":     # the release's BRCA VST used PAM50 subtype as its design
            sub = pd.read_csv(Path("~/MIA_on_bulkRNAseq_CAMDA2026/artifacts/cache/TCGA-BRCA_subtypes.csv")
                              .expanduser(), index_col=0)["Subtype"]
            groups = [str(sub[s]).replace(".", "_") for s in c.columns]
        else:
            groups = [lab[s].replace("-", "_") for s in c.columns]
        meta = pd.DataFrame({"group": groups}, index=c.columns)
        dds = DeseqDataSet(counts=c.T, metadata=meta, design="~group", quiet=True)
        dds.vst_fit(use_design=True)
        V = pd.DataFrame(dds.vst_transform(), index=c.columns, columns=c.index)
        miss = [g for g in genes if g not in V.columns]
        if miss:
            print(f"{name}: {len(miss)} landmark genes fail the count filter in TOIL; set to their minimum",
                  flush=True)
            for g in miss:
                V[g] = float(np.nanmin(V.values))
        V[genes].to_csv(OUT / f"{name}_all_toil_vst_own.tsv", sep="\t")
        print(f"{name}: {V.shape[0]} samples, {int(keep.sum())} genes in the fit, "
              f"trend {list(map(float, dds.uns['vst_trend_coeffs']))}", flush=True)


if __name__ == "__main__":
    {"gdc": gdc, "toil": toil, "toil_vst": toil_vst}[sys.argv[1]]()
