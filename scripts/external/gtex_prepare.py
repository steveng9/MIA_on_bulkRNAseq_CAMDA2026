"""GTEx v10 as a third cohort: normal tissue of the organs in TCGA-COMBINED.

Hakime's suggestion (2026-10-22): GTEx bulk tissue expression as a complement
to TCGA-COMBINED, with the same tissues.  Cohort `GTEX`: the ten GTEx tissues
that match COMBINED's twelve TCGA projects (both kidney projects map to kidney
cortex, both lung projects to lung), 978 landmark genes, the tissue as label.

Donors, not samples, are the unit.  GTEx takes several tissues from each donor
(4.5 of these ten on average), so all of a donor's samples fall on one side:
the reference set is whole donors (16%, COMBINED's share), and the five
member/non-member splits are a donor-grouped, tissue-stratified 5-fold
partition of the rest (the challenge's splits are a 5-fold partition too).

The transform is the challenge's recipe (genes with >= 1 count in >= 10% of
samples; DESeq2 VST, parametric trend, tissue as design; then the 978 landmark
genes), fitted on candidates and reference together as for COMBINED, and kept
frozen so the same donors' other tissues are placed in the release's space
(the extras, for the donor-linked attack: same organ at another site, or
another organ).

Writes ~/data/GTEx/processed/GTEX_{candidates,reference,extras}_vst.tsv,
GTEX_meta.csv, GTEX_extras_meta.csv, GTEX_splits.yaml, GTEX_vst_fit.json.
Run with ~/.venvs/pydeseq2/bin/python (about 30 minutes, 25 GB of memory).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vst as VST  # noqa: E402

RAW = Path("~/data/GTEx/raw").expanduser()
OUT = Path("~/data/GTEx/processed").expanduser()
CH = Path("~/data/CAMDA26").expanduser()
COUNTS = RAW / "GTEx_Analysis_v10_RNASeQCv2.4.2_gene_reads.gct.gz"

# GTEx tissue -> label, with the TCGA-COMBINED projects it stands for
TISSUES = {
    "Breast - Mammary Tissue": "Breast",          # BRCA
    "Kidney - Cortex": "Kidney",                  # KIRC, KIRP
    "Lung": "Lung",                               # LUAD, LUSC
    "Prostate": "Prostate",                       # PRAD
    "Colon - Transverse": "Colon",                # COAD (transverse samples include mucosa)
    "Ovary": "Ovary",                             # OV
    "Liver": "Liver",                             # LIHC
    "Esophagus - Mucosa": "Esophagus",            # ESCA
    "Pancreas": "Pancreas",                       # PAAD
    "Skin - Sun Exposed (Lower leg)": "Skin",     # SKCM
}
# other GTEx sites of the same ten organs: second samples closest to a training sample
SIBLING = {"Colon - Sigmoid", "Esophagus - Muscularis", "Esophagus - Gastroesophageal Junction",
           "Skin - Not Sun Exposed (Suprapubic)", "Kidney - Medulla"}
REF_SHARE = 824 / 5147
SEED = 42


def donor(sample: str) -> str:
    return "-".join(sample.split("-")[:2])


def grouped_folds(meta: pd.DataFrame, k: int, seed: int) -> list:
    """Donor-grouped folds, greedily balanced on tissue counts."""
    rng = np.random.default_rng(seed)
    tab = pd.crosstab(meta.donor, meta.tissue)
    order = rng.permutation(tab.index.values)
    order = sorted(order, key=lambda d: -tab.loc[d].sum())       # big donors first
    load = np.zeros((k, tab.shape[1]))
    target = tab.values.sum(0) / k
    folds = [[] for _ in range(k)]
    for d in order:
        v = tab.loc[d].values
        # marginal increase of the squared relative deviation from an even share
        cost = [(((load[f] + v) / target - 1) ** 2 - (load[f] / target - 1) ** 2).sum()
                for f in range(k)]
        f = int(np.argmin(cost))
        folds[f].append(d)
        load[f] += v
    return folds


def main() -> None:
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.default_inference import DefaultInference
    OUT.mkdir(parents=True, exist_ok=True)
    genes = pd.read_csv(CH / "RED_TCGA-BRCA" / "TCGA-BRCA_primary_tumor_star_deseq_VST_lmgenes.tsv",
                        sep="\t", index_col=0, usecols=[0]).index.tolist()

    att = pd.read_csv(RAW / "GTEx_Analysis_v10_Annotations_SampleAttributesDS.txt", sep="\t",
                      index_col=0, low_memory=False)
    sub = pd.read_csv(RAW / "GTEx_Analysis_v10_Annotations_SubjectPhenotypesDS.txt", sep="\t",
                      index_col=0)
    head = pd.read_csv(COUNTS, sep="\t", skiprows=2, nrows=0).columns.tolist()
    samples = head[2:]
    att = att.loc[samples]
    att["donor"] = [donor(s) for s in att.index]

    cohort = att[att.SMTSD.isin(TISSUES)].copy()
    cohort["tissue"] = cohort.SMTSD.map(TISSUES)
    dup = cohort.duplicated(["donor", "tissue"], keep="first")
    cohort = cohort[~dup]
    donors = np.array(sorted(cohort.donor.unique()))
    rng = np.random.default_rng(SEED)
    ref_donors = set(rng.choice(donors, int(round(REF_SHARE * len(donors))), replace=False))
    cohort["role"] = np.where(cohort.donor.isin(ref_donors), "reference", "candidate")
    extras = att[att.donor.isin(set(donors)) & ~att.index.isin(cohort.index)].copy()
    extras["tier"] = np.where(extras.SMTSD.isin(SIBLING), "same organ, other site",
                              np.where(extras.SMTSD.str.startswith("Cells"), "cell line",
                                       "other organ"))
    extras["donor_role"] = np.where(extras.donor.isin(ref_donors), "reference", "candidate")
    print(f"cohort {len(cohort)} samples ({int((cohort.role == 'candidate').sum())} candidates), "
          f"{len(donors)} donors ({len(ref_donors)} reference); {int(dup.sum())} repeated "
          f"donor-tissue samples dropped; extras {len(extras)}", flush=True)

    use = list(cohort.index) + list(extras.index)
    dtype = {s: np.float64 for s in use}
    dtype.update(Name=str, Description=str)
    raw = pd.read_csv(COUNTS, sep="\t", skiprows=2, usecols=["Name"] + use, dtype=dtype,
                      index_col=0)
    raw.index = raw.index.str.replace(r"\..*$", "", regex=True)
    raw = raw[~raw.index.duplicated()]
    print(f"read {raw.shape[0]} genes x {raw.shape[1]} samples", flush=True)
    missing = [g for g in genes if g not in raw.index]
    if missing:
        raise SystemExit(f"{len(missing)} landmark genes missing from GTEx v10, e.g. {missing[:3]}")

    K = np.rint(raw[cohort.index].values).astype(np.int64)
    keep = VST.gene_filter(K)
    lost = [g for g in genes if not keep[raw.index.get_loc(g)]]
    if lost:
        print(f"{len(lost)} landmark genes fail the count filter; kept anyway", flush=True)
        keep[[raw.index.get_loc(g) for g in lost]] = True
    fit_genes = raw.index[keep]
    c = pd.DataFrame(K[keep].T, index=cohort.index, columns=fit_genes)
    meta = pd.DataFrame({"tissue": cohort.tissue.values}, index=cohort.index)
    dds = DeseqDataSet(counts=c, metadata=meta, design="~tissue", quiet=True,
                       inference=DefaultInference(n_cpus=8))
    dds.vst_fit(use_design=True)
    Vc = pd.DataFrame(dds.vst_transform(), index=c.index, columns=fit_genes)[genes]
    a0, a1 = map(float, dds.uns["vst_trend_coeffs"])
    print(f"VST fitted on {len(c)} samples x {len(fit_genes)} genes; trend a0={a0:.4g} a1={a1:.4g}",
          flush=True)

    Ke = np.rint(raw.loc[fit_genes, extras.index].values.T).astype(np.int64)
    Ve = pd.DataFrame(dds.vst_transform(counts=Ke), index=extras.index, columns=fit_genes)[genes]
    # the frozen transform on the cohort itself must give back the fitted values
    Vchk = pd.DataFrame(dds.vst_transform(counts=c.values[:50]), index=c.index[:50],
                        columns=fit_genes)[genes]
    err = float(np.abs(Vchk.values - Vc.values[:50]).max())
    print(f"frozen-transform check on 50 cohort samples: max abs error {err:.2e}", flush=True)

    cand = cohort[cohort.role == "candidate"]
    Vc.loc[cand.index].to_csv(OUT / "GTEX_candidates_vst.tsv", sep="\t")
    Vc.loc[cohort.index[cohort.role == "reference"]].to_csv(OUT / "GTEX_reference_vst.tsv", sep="\t")
    Ve.to_csv(OUT / "GTEX_extras_vst.tsv", sep="\t")
    sexes = {1: "male", 2: "female"}
    for frame, name in ((cohort, "GTEX_meta.csv"), (extras, "GTEX_extras_meta.csv")):
        out = frame[[c for c in ("donor", "tissue", "SMTSD", "role", "tier", "donor_role", "SMRIN",
                                 "SMTSISCH") if c in frame.columns]].copy()
        out["sex"] = sub.loc[out.donor, "SEX"].map(sexes).values
        out["age"] = sub.loc[out.donor, "AGE"].values
        out.index.name = "sample"
        out.to_csv(OUT / name)

    write_splits(cand)
    (OUT / "GTEX_vst_fit.json").write_text(json.dumps(dict(
        n_cohort=len(cohort), n_candidates=len(cand), n_reference=int((cohort.role == "reference").sum()),
        n_donors=len(donors), n_reference_donors=len(ref_donors), n_extras=len(extras),
        genes_in_fit=int(keep.sum()), landmark_genes_below_filter=lost, trend_a0=a0, trend_a1=a1,
        frozen_check_max_abs_error=err, seed=SEED, tissues=TISSUES)))
    print(cand.tissue.value_counts().to_string(), flush=True)
    print(extras.tier.value_counts().to_string(), flush=True)


def write_splits(cand: pd.DataFrame) -> None:
    folds = grouped_folds(cand, 5, SEED)
    splits = {"splits": {f"split_{i + 1}": {"test_index": sorted(cand.index[cand.donor.isin(f)])}
                         for i, f in enumerate(folds)}}
    with open(OUT / "GTEX_splits.yaml", "w") as f:
        yaml.safe_dump(splits, f)
    sizes = [len(v["test_index"]) for v in splits["splits"].values()]
    print(f"non-members per split: {sizes}", flush=True)
    print(pd.crosstab(cand.tissue, [next(i for i, f in enumerate(folds) if d in f) + 1
                                    for d in cand.donor]).to_string(), flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["--splits-only"]:      # redraw the splits from the written metadata
        meta = pd.read_csv(OUT / "GTEX_meta.csv", index_col=0)
        write_splits(meta[meta.role == "candidate"])
    else:
        main()
