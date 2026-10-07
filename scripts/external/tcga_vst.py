"""Recover the challenge's fitted VST from GDC counts, verify it, and apply it
to TCGA samples the challenge did not distribute.

Needs the output of gdc_download.py.  For each challenge cohort (BRCA,
COMBINED) the transform is identified from the distributed values themselves:

  1. (asymptDisp, extraPois) are the pair for which, after inverting the
     transform, count / normalised-count is the same number across the 978
     genes of a sample -- that number is the sample's size factor.  No
     assumption about which samples the organisers fitted on.
  2. Gene-wise geometric means are taken over the cohort's distributed samples
     (for COMBINED: candidates plus reference set) on genes passing the
     challenge's low-count filter; a sample's size factor recomputed from them
     is compared with the one recovered in step 1.
  3. Every distributed sample is re-derived from its counts with the frozen
     transform and compared with the distributed values.

Writes to ~/data/TCGA_GDC/processed/:

  <COHORT>_vst_fit.json          coefficients, geometric means, verification
  <COHORT>_extras_vst.tsv        every non-distributed aliquot, samples x 978 genes
  extras_meta.csv                aliquot, patient, sample type, project, relation
  COMBINED_reference_tpm.tsv / _log2tpm.tsv
  COMBINED_reference_counts.npz  all-gene counts of the 824, for a VST refit
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vst as VST  # noqa: E402

ROOT = Path("~/data/TCGA_GDC").expanduser()
CH = Path("~/data/CAMDA26").expanduser()
OUT = ROOT / "processed"


def challenge(name: str, reference: bool = False) -> pd.DataFrame:
    suffix = "_reference" if reference else ""
    f = CH / f"RED_TCGA-{name}" / f"TCGA-{name}_primary_tumor_star_deseq_VST_lmgenes{suffix}.tsv"
    return pd.read_csv(f, sep="\t", index_col=0).T          # samples x genes


def assemble():
    man = pd.read_csv(ROOT / "gdc_star_counts_manifest.csv").set_index("file_id")
    genes = pd.read_csv(ROOT / "genes.tsv", sep="\t")
    parts = sorted((ROOT / "parts").glob("[0-9]*.npz"))
    counts, tpm, fids = [], [], []
    for p in parts:
        if ".tmp" in p.name:
            continue
        d = np.load(p, allow_pickle=True)
        counts.append(d["counts"])
        tpm.append(d["tpm"])
        fids += list(d["file_ids"])
    counts, tpm = np.concatenate(counts, 1), np.concatenate(tpm, 1)
    meta = man.loc[fids].reset_index()
    print(f"assembled {counts.shape[1]} files x {counts.shape[0]} genes "
          f"({len(man) - len(fids)} files missing)", flush=True)
    return genes, meta, counts, tpm


def invert_u(v, a, e):
    """a * q for transformed values v."""
    y = 4 * a * np.exp2(v)
    return (y - (1 + e)) ** 2 / (4 * y)


def identify(K: np.ndarray, Vt: np.ndarray):
    """(a, e, per-sample size factors) from counts K and distributed values Vt, samples x genes."""
    from scipy.optimize import minimize
    rng = np.random.default_rng(0)
    sub = rng.choice(len(K), min(300, len(K)), replace=False)
    Ks, Vs = K[sub].astype(np.float64), Vt[sub]
    ok = Ks >= 10

    def spread(p):
        a, e = np.exp(p)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(ok, np.log(Ks) - np.log(invert_u(Vs, a, e) / a), np.nan)
        return float(np.nanmean(np.nanvar(r, axis=1)))

    best = min((minimize(spread, np.log(s), method="Nelder-Mead",
                         options={"xatol": 1e-12, "fatol": 1e-18, "maxiter": 4000})
                for s in [(0.5, 3.0), (0.1, 1.0), (1.0, 10.0)]), key=lambda r: r.fun)
    a, e = np.exp(best.x)
    Kf = K.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(Kf >= 10, np.log(Kf) - np.log(invert_u(Vt, a, e) / a), np.nan)
    return float(a), float(e), np.exp(np.nanmedian(r, axis=1)), float(best.fun)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    genes, meta, counts, tpm = assemble()
    ens = genes.gene_id.str.replace(r"\..*$", "", regex=True).values
    col = {a: i for i, a in enumerate(meta.aliquot)}

    brca, comb, ref = challenge("BRCA"), challenge("COMBINED"), challenge("COMBINED", True)
    lm = [int(np.where(ens == g)[0][0]) for g in brca.columns]
    cohorts = {"BRCA": list(brca.index), "COMBINED": list(comb.index) + list(ref.index)}
    values = {"BRCA": brca, "COMBINED": pd.concat([comb, ref])}
    distributed = set(cohorts["BRCA"]) | set(cohorts["COMBINED"])

    # ── extras: what else GDC holds, and how it relates to the challenge donors
    pat = {"BRCA": {a[:12] for a in brca.index}, "COMBINED": {a[:12] for a in comb.index},
           "REF": {a[:12] for a in ref.index}}
    samp = {a[:16] for a in distributed}
    ex = meta[~meta.aliquot.isin(distributed)].copy()
    for k, s in pat.items():
        ex[f"donor_in_{k}"] = ex.patient.isin(s)

    def tier(r):
        if r.sample_type == "Solid Tissue Normal":
            return "matched normal"
        if r.sample_type == "Primary Tumor":
            return "same tumour sample, other aliquot" if r["sample"] in samp \
                else "same tumour, other vial"
        return "other lesion (metastatic / recurrent / new primary)"
    ex["tier"] = ex.apply(tier, axis=1)
    ex.drop(columns=["file_name", "size"]).to_csv(OUT / "extras_meta.csv", index=False)
    print(ex.groupby(["tier", "donor_in_COMBINED"]).size().to_string(), flush=True)

    for name, ids in cohorts.items():
        missing = [a for a in ids if a not in col]
        if missing:
            raise SystemExit(f"{name}: {len(missing)} distributed aliquots not downloaded yet")
        j = [col[a] for a in ids]
        K_all = counts[:, j]
        K = K_all[lm].T                                    # samples x 978
        Vt = values[name].loc[ids].values
        a, e, sf, spread = identify(K, Vt)

        keep = VST.gene_filter(K_all)
        loggeo = np.full(len(ens), -np.inf)
        loggeo[keep] = VST.log_geo_means(K_all[keep])
        sf_mor = VST.size_factors(K_all, loggeo)
        err_sf = float(np.abs(sf_mor / sf - 1).max())
        recon = VST.vst(K / sf_mor[:, None], a, e)
        err = np.abs(recon - Vt)
        fit = dict(cohort=name, n_samples=len(ids), asympt_disp=a, extra_pois=e,
                   genes_passing_filter=int(keep.sum()),
                   genes_in_geometric_mean=int(np.isfinite(loggeo).sum()),
                   size_factor_max_rel_error=err_sf,
                   reconstruction_max_abs_error=float(err.max()),
                   reconstruction_mean_abs_error=float(err.mean()),
                   reconstruction_p999_abs_error=float(np.quantile(err, 0.999)),
                   loggeo={g: float(v) for g, v in zip(ens, loggeo) if np.isfinite(v)})
        (OUT / f"{name}_vst_fit.json").write_text(json.dumps(fit))
        print({k: v for k, v in fit.items() if k != "loggeo"}, flush=True)

        jx = [col[a] for a in ex.aliquot]
        sfx = VST.size_factors(counts[:, jx], loggeo)
        Xx = VST.vst(counts[lm][:, jx].T / sfx[:, None], a, e)
        pd.DataFrame(Xx, index=ex.aliquot.values, columns=brca.columns).to_csv(
            OUT / f"{name}_extras_vst.tsv", sep="\t")

    jr = [col[a] for a in ref.index]
    T = pd.DataFrame(tpm[lm][:, jr].T.astype(np.float64), index=ref.index, columns=brca.columns)
    T.to_csv(OUT / "COMBINED_reference_tpm.tsv", sep="\t")
    np.log2(T + 1).to_csv(OUT / "COMBINED_reference_log2tpm.tsv", sep="\t")
    proj = meta.set_index("aliquot").loc[ref.index, "project"].values
    np.savez_compressed(OUT / "COMBINED_reference_counts.npz", counts=counts[:, jr],
                        gene_ids=ens, aliquots=np.array(ref.index), project=proj)
    print("done", flush=True)


if __name__ == "__main__":
    main()
