"""Markdown tables for the donor-linked and bio-perturbation experiments.

Reads the CSVs in results/perturb/ and writes results/perturb/TABLES.md.
Every number is a mean over the five canonical splits unless a table says
otherwise.

    python scripts/perturb/report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mia import paths  # noqa: E402

R = paths.RESULTS / "perturb"
GEN = {"mvn": "MVN", "cvae": "CVAE", "nd": "NoisyDiffusion", "pgg": "DP-PGM (CAMDA-26)",
       "pgm": "DP-PGM (new)", "tabsyn": "TabSyn (tuned)", "tabpfn": "TabPFN"}
ORDER = list(GEN.values())


def gen_name(s: pd.Series) -> pd.Series:
    return s.str.split("@").str[0].map(GEN)


def md(df: pd.DataFrame, digits: int = 3) -> str:
    return df.round(digits).to_markdown(floatfmt=f".{digits}f")


def cols(df):
    return df[[c for c in ORDER if c in df.columns]]


def aux_tables(out: list) -> None:
    f = R / "aux_mismatch.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    d["generator"] = gen_name(d.generator)
    names = {
        "none": "no auxiliary set",
        "tcga_heldout84": "TCGA-BRCA, 84 held-out non-members (matched control)",
        "gse_vst_frozen": "GSE58135, TCGA's fitted VST",
        "gse_vst_own": "GSE58135, VST fitted on itself",
        "gse_log2tpm": "GSE58135, log2(TPM+1)",
        "gse_tpm": "GSE58135, TPM",
        "gse_tpm_aligned": "GSE58135, any of the above, quantile-aligned to the release",
        "ref_vst": "reference set as distributed (VST, joint fit)",
        "ref_vst_own": "reference set, VST refitted on itself",
        "ref_log2tpm": "reference set, log2(TPM+1)",
        "ref_tpm": "reference set, TPM",
        "ref_tpm_aligned": "reference set, TPM, quantile-aligned to the release",
        "ref_vst_n84": "reference set as distributed, 84 samples only",
    }
    for ds, title in (("BRCA", "Experiment 1: TCGA-BRCA, external auxiliary cohort GSE58135"),
                      ("COMBINED", "Experiment 2: TCGA-COMBINED, auxiliary set re-normalised")):
        x = d[(d.dataset == ds) & d.condition.isin(names)]
        if x.empty:
            continue
        n = x.groupby("generator").split.nunique()
        out.append(f"## {title}\n")
        out.append("AUC, mean over splits (" + ", ".join(f"{g}: {k}" for g, k in n.items()) + ").\n")
        for attack in ["MahalaMIA (ridge 1e-4)", "MahalaMIA (as submitted)", "RedSigma",
                       "DOMIAS-KDE", "GAN-leaks cal."]:
            keep = x[x.attack == attack]
            if attack.startswith("MahalaMIA"):
                noaux = x[x.attack == attack.replace("(", "(no aux, ").replace(
                    "no aux, as submitted", "no aux")]
                keep = pd.concat([keep, noaux])
            t = keep.pivot_table(index="condition", columns="generator", values="auc")
            t = cols(t.reindex([c for c in names if c in t.index]))
            t.index = [names[c] for c in t.index]
            out.append(f"**{attack}**\n\n{md(t)}\n")
        base = x[x.condition == "none"].pivot_table(index="attack", columns="generator", values="auc")
        out.append(f"**Attacks that use no auxiliary set (unchanged by these conditions)**\n\n"
                   f"{md(cols(base))}\n")


def subset_tables(out: list) -> None:
    for tag, label in (("", ""), ("_gse", " (BRCA with GSE58135 as the auxiliary set)")):
        f = R / f"gene_subsets{tag}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        d["generator"] = gen_name(d.generator)
        out.append(f"## Experiment 3: reduction dimension and gene subsets{label}\n")
        for ds in sorted(d.dataset.unique()):
            x = d[d.dataset == ds]
            b = x[x.block == "baseline_dr"]
            if not b.empty:
                out.append(f"### {ds}: challenge baselines, reduction dimension swept "
                           "(the baseline fixes it at 100)\n")
                for attack in ["DOMIAS-KDE", "GAN-leaks cal.", "LOGAN-D1"]:
                    t = b[b.attack == attack].pivot_table(index=["generator", "rule"],
                                                          columns="k", values="auc")
                    if not t.empty:
                        out.append(f"**{attack}**\n\n{md(t)}\n")
            s = x[x.block == "gene_subset"]
            out.append(f"### {ds}: our attacks when the adversary holds only k genes\n")
            out.append("Rows are generators; the gene-selection rule is `random` "
                       "(the rules agree to within 0.02, see the last table).\n")
            for attack in ["MahalaMIA (ridge 1e-4)", "MahalaMIA (as submitted)", "RedSigma",
                           "GAN-leaks", "MAMA-MIA v2", "MAMA-MIA v1"]:
                t = s[(s.attack == attack) & s.rule.isin(["random", "all"])].pivot_table(
                    index="generator", columns="k", values="auc")
                t = t.reindex([g for g in ORDER if g in t.index])
                out.append(f"**{attack}**\n\n{md(t)}\n")
            rule = s[s.rule != "all"].pivot_table(index=["attack", "rule"], columns="k", values="auc")
            out.append(f"**{ds}: effect of the selection rule, averaged over generators**\n\n"
                       f"{md(rule)}\n")


def donor_tables(out: list) -> None:
    f = R / "donor_linked.csv"
    if not f.exists():
        return
    link = pd.read_csv(R / "donor_linked_linkability.csv")
    out.append("## Donor-linked membership inference\n")
    out.append("### How recognisable is a donor's second sample?\n")
    out.append("For each second sample: where does its own donor's challenge sample rank "
               "among all candidates by correlation (no generator involved).\n")
    out.append(link.round(3).to_markdown(index=False) + "\n")
    lf = R / "donor_linkability.csv"
    if lf.exists():
        ll = pd.read_csv(lf)
        t = ll.pivot_table(index=["dataset", "tier"], columns="space", values="own_is_nearest")
        order = [c for c in ll.space.unique() if c in t.columns]
        out.append("Fraction whose own donor is the single most correlated candidate, after "
                   "removing class means and leading principal components:\n")
        out.append(md(t[order]) + "\n")

    for name, title in (("donor_linked.csv", "Attack results"),
                        ("donor_linked_melomia.csv",
                         "MeLoMIA (second samples read through the existing shadow stacks)"),
                        ("donor_linked_adapted.csv",
                         "Attack results after the adversary aligns the second samples to the release")):
        if not (R / name).exists():
            continue
        d = pd.read_csv(R / name)
        d["generator"] = gen_name(d.generator)
        out.append(f"### {title}\n")
        out.append("`linked` = AUC for the donor's membership from the second sample (never "
                   "trained on); `overlap` = the same donors' trained-on samples; brackets are "
                   "95% donor-bootstrap intervals; `within donor` = how often a second sample "
                   "scores higher in a split where its donor is a member than in the split "
                   "where it is not (0.5 = no effect).\n")
        for ds in sorted(d.dataset.unique()):
            for tier in d[d.dataset == ds].tier.unique():
                x = d[(d.dataset == ds) & (d.tier == tier)]
                rows = []
                for (g, a), r in x.set_index(["generator", "attack"]).iterrows():
                    rows.append(dict(
                        generator=g, attack=a,
                        linked=f"{r.linked_auc:.3f} [{r.linked_lo:.2f}, {r.linked_hi:.2f}]",
                        overlap=f"{r.overlap_auc:.3f} [{r.overlap_lo:.2f}, {r.overlap_hi:.2f}]",
                        cohort=f"{r.cohort_auc:.3f}",
                        **{"within donor": f"{r.within_donor:.3f} [{r.within_lo:.2f}, "
                                           f"{r.within_hi:.2f}]"}))
                t = pd.DataFrame(rows)
                t["o"] = t.generator.map({g: i for i, g in enumerate(ORDER)})
                t = t.sort_values(["o", "attack"]).drop(columns="o")
                n = int(x.n_B.iloc[0])
                out.append(f"**{ds}, {tier} (n = {n} second samples, "
                           f"{int(x.n_donors.iloc[0])} donors)**\n\n{t.to_markdown(index=False)}\n")


def main() -> None:
    out = ["# Donor-linked attacks and bio-perturbation experiments: tables\n",
           "Generated by `scripts/perturb/report.py`; the reading of these tables is in "
           "`docs/DONOR_LINKED_AND_PERTURBATIONS.md`.\n"]
    donor_tables(out)
    aux_tables(out)
    subset_tables(out)
    (R / "TABLES.md").write_text("\n".join(out))
    print(f"wrote {R / 'TABLES.md'} ({sum(len(o) for o in out) // 1000} kB)")


if __name__ == "__main__":
    main()
