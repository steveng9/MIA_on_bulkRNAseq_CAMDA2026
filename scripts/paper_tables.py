#!/usr/bin/env python
"""LaTeX rows for the paper's tables, from the result files.

    python scripts/paper_tables.py            # every table
    python scripts/paper_tables.py quality bio

Each function prints the body rows of one table of current_draft/main.tex or
supplement.tex.  MahalaMIA numbers are the paper's default attack
(configs/experiments/paper_grid_*.yaml; the `_default` outputs of
scripts/perturb/).  The extended attack grid is `paper_figures.py --latex`
and the MahalaMIA ablation is `mahalamia_ablation.py --latex`.

  quality     fidelity, utility, biological plausibility and exposure (main text)
  quality_all the same for the three cohorts (supplement)
  bio         differential expression and co-expression at both thresholds (supplement)
  aux         auxiliary data that does not match the release
  candidates  the attacked profiles in another representation
  genes       fewer genes
  donor       a second sample of a training donor (TCGA)
  tissue      another tissue of a training donor (GTEx)
  epsilon     the corrected DP-PGM across epsilon
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
RES = ROOT / "results"
TABSYN = ("tabsyn@class_freq=train,diff_schedule=steps,latent_scale=std,"
          "preprocess=clip:0.001:0.999+standard")
PGM = "pgm@binning=dp_quantile,edge_estimator=threshold,n_bins=16"
GEN = {"mvn": "MVN", "cvae": "CVAE", "nd": "NoisyDiffusion", TABSYN: "TabSyn", "tabpfn": "TabPFN",
       "pgg": "DP-PGM (challenge)", PGM: "DP-PGM (corrected)"}
DP = {"pgg", PGM}
COHORT = {"BRCA": "TCGA-BRCA", "COMBINED": "TCGA-COMBINED", "GTEX": "GTEx"}


def f2(v, d=2):
    return "--" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"


def fidelity() -> pd.DataFrame:
    f = pd.read_csv(RES / "fidelity.csv")
    f = f[f.generator.isin(GEN)].drop_duplicates(["dataset", "generator", "split"], keep="last")
    return f.groupby(["dataset", "generator"]).agg(
        utility=("utility_ratio", "mean"), w1=("wasserstein_mean", "mean"),
        corr=("corr_mae", "mean"), disc=("discriminator_auc", "mean"),
        fid_splits=("split", "nunique"))


def bio(lfc=0.0, cutoff=0.5) -> pd.DataFrame:
    b = pd.read_csv(RES / "bio.csv")
    b = b[(b.lfc == lfc) & (b.cutoff == cutoff)]
    return b.groupby(["dataset", "generator"])[
        ["de_tpr", "de_precision", "de_fpr", "coex_recall", "coex_precision",
         "coex_edges_real", "coex_edges_syn", "n"]].mean()


def exposure() -> pd.DataFrame:
    """Strongest of our attacks per cohort and generator (AUC and TPR chosen separately)."""
    e = pd.read_csv(ROOT / "paper" / "Fig" / "data" / "exposure.csv")
    e = e[e.group == "ours"]
    back = {v: k for k, v in GEN.items()}
    e = e.assign(generator=e.generator.map(back))
    return e.pivot_table(index=["dataset", "generator"], columns="metric", values="value")


def quality(cohorts=("BRCA",), extra=("COMBINED",)) -> None:
    fid, b, ex = fidelity(), bio(), exposure()
    for ds in cohorts:
        print(f"% ---- {COHORT[ds]}: utility, W1, corr. error, discriminator, DE recall / precision, "
              "co-expression recall / precision, attack AUC, TPR at 1% FPR"
              + "".join(f"; {COHORT[x]} utility, attack AUC" for x in extra))
        for g, name in GEN.items():
            if (ds, g) not in fid.index:
                continue
            if g == "pgg":
                print("  \\midrule")
            r, q, x = fid.loc[(ds, g)], b.loc[(ds, g)], ex.loc[(ds, g)]
            cells = [name, f2(r.utility), f2(r.w1), f2(r["corr"], 3), f2(r.disc),
                     f"{f2(q.de_tpr)} / {f2(q.de_precision)}",
                     f"{f2(q.coex_recall)} / {f2(q.coex_precision)}",
                     f2(x["auc"]), f2(x["tpr_at_fpr_0.01"])]
            for o in extra:
                mark = ""
                if (o, g) in fid.index:
                    if fid.loc[(o, g)].fid_splits < 5:
                        mark = "$^{\\dagger}$"
                    cells += [f2(fid.loc[(o, g)].utility) + mark, f2(ex.loc[(o, g)]["auc"]) + mark]
                else:
                    cells += ["--", "--"]
            print("  " + " & ".join(cells) + " \\\\")
        q = b.loc[(ds, "real_heldout")]
        print("  \\midrule")
        print("  Real held-out donors & & & & & "
              f"{f2(q.de_tpr)} / {f2(q.de_precision)} & {f2(q.coex_recall)} / {f2(q.coex_precision)}"
              " & & " + "& & " * len(extra) + "\\\\")
        print(f"% splits: " + ", ".join(f"{GEN[g]} {int(fid.loc[(ds, g)].fid_splits)}"
                                        for g in GEN if (ds, g) in fid.index))


def quality_all() -> None:
    for ds in COHORT:
        print(f"  \\multicolumn{{9}}{{l}}{{\\emph{{{COHORT[ds]}}}}} \\\\")
        quality(cohorts=(ds,), extra=())


def bio_table() -> None:
    a, s = bio(0.0, 0.5), bio(1.0, 0.7)
    for ds in COHORT:
        print(f"  \\multicolumn{{9}}{{l}}{{\\emph{{{COHORT[ds]}}} "
              f"({a.loc[(ds, 'real_heldout')].coex_edges_real:,.0f} real edges at 0.5, "
              f"{s.loc[(ds, 'real_heldout')].coex_edges_real:,.0f} at 0.7)}} \\\\")
        for g, name in list(GEN.items()) + [("real_heldout", "Real held-out donors")]:
            if (ds, g) not in a.index:
                continue
            x, y = a.loc[(ds, g)], s.loc[(ds, g)]
            print(f"  {name} & {f2(x.de_tpr)} & {f2(x.de_precision)} & {f2(y.de_tpr)} & "
                  f"{f2(y.de_precision)} & {f2(x.coex_recall)} & {f2(x.coex_precision)} & "
                  f"{x.coex_edges_syn:,.0f} & {f2(y.coex_recall)} & {f2(y.coex_precision)} \\\\")
        print("  \\midrule")


def _mean(csv, where: dict, value="auc"):
    d = pd.read_csv(RES / "perturb" / csv)
    for k, v in where.items():
        d = d[d[k] == v]
    return d.groupby("generator")[value].mean()


def aux() -> None:
    rows = {"COMBINED": [("None", "none"), ("VST (matched)", "ref_vst"), ("TPM", "ref_tpm"),
                         ("TPM, aligned to the release", "ref_tpm_aligned")],
            "BRCA": [("None", "none"), ("GSE58135, TPM", "gse_tpm"),
                     ("GSE58135, VST", "gse_vst_frozen"),
                     ("GSE58135, aligned to the release", "gse_vst_frozen_aligned"),
                     ("TCGA-BRCA non-members (matched)", "tcga_heldout84")]}
    for ds, items in rows.items():
        print(f"% ---- {COHORT[ds]}: MVN, CVAE, ND, TabSyn, TabPFN")
        for label, cond in items:
            m = _mean("aux_mismatch_default.csv", dict(dataset=ds, condition=cond))
            print(f"  {label} & " + " & ".join(f2(m.get(g, np.nan)) for g in list(GEN)[:5]) + " \\\\")


def candidates() -> None:
    items = [("VST, as distributed", "vst"), ("GDC TPM", "gdc_log2tpm"),
             ("GDC TPM, aligned", "gdc_log2tpm_aligned"), ("TOIL TPM", "toil_log2tpm"),
             ("TOIL TPM, aligned", "toil_log2tpm_aligned"),
             ("TOIL, adversary's VST", "toil_vst_own"),
             ("TOIL, adversary's VST, aligned", "toil_vst_own_aligned")]
    for label, cond in items:
        cells = []
        for ds, ref in (("BRCA", "none"), ("COMBINED", "ref_vst")):
            m = _mean("candidate_mismatch_default.csv", dict(dataset=ds, condition=f"{cond}|{ref}"))
            cells += [f2(m.get(g, np.nan)) for g in ("mvn", "cvae", "nd")]
        print(f"  {label} & " + " & ".join(cells) + " \\\\")
    t = _mean("candidate_mismatch_default.csv", dict(dataset="BRCA"), "tpr_at_fpr_0_01")
    d = pd.read_csv(RES / "perturb" / "candidate_mismatch_default.csv")
    d = d[(d.dataset == "BRCA") & (d.generator == "mvn")]
    print("% TPR at 1% FPR, MVN, TCGA-BRCA: "
          + ", ".join(f"{c.split('|')[0]} {v:.2f}" for c, v in
                      d.groupby("condition").tpr_at_fpr_0_01.mean().items()))


def genes() -> None:
    ks = [10, 100, 400, 800, 978]
    new = pd.read_csv(RES / "perturb" / "gene_subsets_default.csv")
    old = pd.read_csv(RES / "perturb" / "gene_subsets.csv")

    def row(d, ds, g, attack):
        d = d[(d.dataset == ds) & (d.generator == g) & (d.attack == attack)
              & d.rule.isin(["random", "all"])]
        m = d.groupby("k").auc.mean()
        return " & ".join(f2(m.get(k, np.nan)) for k in ks)

    print("% ---- TCGA-BRCA")
    for g in ("mvn", "cvae", "nd"):
        print(f"  \\mahala{{}} / {GEN[g] if g != 'nd' else 'ND'} & {row(new, 'BRCA', g, 'MahalaMIA')} \\\\")
    print(f"  GAN-leaks / TabSyn & {row(old, 'BRCA', TABSYN, 'GAN-leaks')} \\\\")
    print("% ---- TCGA-COMBINED")
    print(f"  \\mahala{{}} / MVN & {row(new, 'COMBINED', 'mvn', 'MahalaMIA')} \\\\")
    print(f"  \\mama{{}} / DP-PGM & {row(old, 'COMBINED', 'pgg', 'MAMA-MIA v2')} \\\\")


def donor() -> None:
    new = pd.read_csv(RES / "perturb" / "donor_linked_default.csv")
    old = pd.read_csv(RES / "perturb" / "donor_linked.csv")
    mel = pd.read_csv(RES / "perturb" / "donor_linked_melomia.csv")
    sep, nor = "separate tumour sample (other vial or other lesion)", "matched normal"

    def row(d, ds, g, attack, label):
        d = d[(d.dataset == ds) & (d.generator == g) & (d.attack == attack)].set_index("tier")
        s, n = d.loc[sep], d.loc[nor]
        print(f"  {label} & {s.overlap_auc:.2f} & {s.linked_auc:.2f} ({s.linked_lo:.2f}--{s.linked_hi:.2f})"
              f" & {n.linked_auc:.2f} \\\\")

    for ds in ("COMBINED", "BRCA"):
        print(f"% ---- {COHORT[ds]}: sample A, separate tumour (95% interval), matched normal")
        for g, short in (("mvn", "MVN"), ("cvae", "CVAE"), ("nd", "ND")):
            row(new, ds, g, "MahalaMIA", f"\\mahala{{}} / {short}")
        row(mel, ds, "cvae", "MeLoMIA-CVAE", "\\melo{} / CVAE")
        row(mel, ds, "nd", "MeLoMIA-ND", "\\melo{} / ND")
        row(old, ds, "pgg", "MAMA-MIA v2", "\\mama{} / DP-PGM (challenge)")
    for ds in ("COMBINED", "BRCA"):
        r = pd.read_csv(RES / "perturb" / f"donor_linked_resplit_{ds}_mvn_default.csv").set_index("group")
        for grp in (sep, nor, "same tumour sample, other aliquot"):
            x = r.loc[grp]
            print(f"% calibrated, {ds}, {grp}: {x.auc:.2f} -> {x.calibrated_auc:.2f} "
                  f"({x.calibrated_lo:.2f}--{x.calibrated_hi:.2f}), shuffled {x.calibrated_null:.2f}")
    d = new[(new.attack == "MahalaMIA") & (new.generator == "mvn") & (new.dataset == "COMBINED")]
    print("% MVN, COMBINED, by tier: " + "; ".join(f"{t}: {a:.2f} (A {o:.2f})" for t, a, o in
                                                   zip(d.tier, d.linked_auc, d.overlap_auc)))


def tissue() -> None:
    d = pd.read_csv(RES / "perturb" / "gtex_donor_linked_default.csv")
    d = d[d.attack == "MahalaMIA"]
    m = d.groupby(["tissue", "generator"]).auc.mean().unstack()
    cal = pd.read_csv(RES / "perturb" / "gtex_donor_linked_resplit_mvn_default.csv").set_index("group")
    tr = d.groupby("generator").trained_auc.mean()
    print(f"  The trained-on samples & {tr['mvn']:.2f} & {tr['cvae']:.2f} & {tr['nd']:.2f} & "
          f"{cal.loc['cohort samples (trained on)'].calibrated_auc:.2f} \\\\")
    for t in m.index:
        print(f"  {t} & {m.loc[t, 'mvn']:.2f} & {m.loc[t, 'cvae']:.2f} & {m.loc[t, 'nd']:.2f} & "
              f"{cal.loc[t].calibrated_auc:.2f} \\\\")
    tp = d[d.generator == "mvn"].groupby("tissue").tpr_at_fpr_0_01.mean()
    print("% TPR at 1% FPR, MVN: " + ", ".join(f"{t} {v:.3f}" for t, v in tp.items()))
    nr = pd.read_csv(RES / "perturb" / "gtex_donor_linked_default.csv")
    nr = nr[(nr.attack == "MahalaMIA (no reference)") & (nr.generator == "mvn")]
    print("% no reference set, MVN: " + ", ".join(f"{t} {v:.2f}" for t, v in
                                                   nr.groupby("tissue").auc.mean().items()))


def epsilon() -> None:
    import paper_figures as PF
    t = PF.epsilon_table()
    for ds in ("BRCA", "COMBINED"):
        print(f"% ---- {COHORT[ds]}: epsilon, bound, MAMA-MIA AUC, TPR at 1% FPR, utility, W1")
        for r in t[t.dataset == ds].itertuples():
            print(f"  {r.epsilon:g} & {r.bound:.2f} & {r.auc:.2f} & {r.tpr_at_fpr_0_01:.2f} & "
                  f"{r.utility:.2f} & {r.w1:.2f} \\\\")


TABLES = {"quality": quality, "quality_all": quality_all, "bio": bio_table, "aux": aux,
          "candidates": candidates, "genes": genes, "donor": donor, "tissue": tissue,
          "epsilon": epsilon}

if __name__ == "__main__":
    for name in (sys.argv[1:] or TABLES):
        print(f"\n%% ===== {name} =====")
        TABLES[name]()
