# GTEx as a third cohort

Hakime's suggestion (2026-10-22): GTEx bulk tissue expression
(https://gtexportal.org/home/downloads/adult-gtex/bulk_tissue_expression) as a
complement to TCGA-COMBINED, with the same tissues. It also answers the question
TCGA cannot, which the paper lists as a limitation: does a sample of *another
tissue* of a training donor reveal that the donor was in the training data?

## Data

GTEx v10 (RNA-SeQC 2.4.2 gene read counts, sample and subject annotations),
open access, in `~/data/GTEx/raw` (outside the repository): 19,788 samples,
946 donors, 68 tissue types, about 21 samples per donor.

## Cohort `GTEX`

`scripts/external/gtex_prepare.py`, registered by `configs/datasets/GTEX.yaml`.

- **Tissues.** The ten GTEx tissues that match COMBINED's twelve TCGA projects:

  | GTEx tissue | label | TCGA project(s) |
  |---|---|---|
  | Breast - Mammary Tissue | Breast | BRCA |
  | Kidney - Cortex | Kidney | KIRC, KIRP |
  | Lung | Lung | LUAD, LUSC |
  | Prostate | Prostate | PRAD |
  | Colon - Transverse | Colon | COAD |
  | Ovary | Ovary | OV |
  | Liver | Liver | LIHC |
  | Esophagus - Mucosa | Esophagus | ESCA |
  | Pancreas | Pancreas | PAAD |
  | Skin - Sun Exposed (Lower leg) | Skin | SKCM |

  4,168 samples from 923 donors (COMBINED: 4,323 candidates, 824 reference).
- **Genes and transform.** The 978 landmark genes and the challenge's recipe:
  genes with at least one count in 10% of samples (42,533), DESeq2's VST
  (parametric trend, tissue as design; pydeseq2), fitted on candidates and
  reference together as for COMBINED. Trend coefficients 0.416 / 2.763. The
  fitted transform is kept frozen and reused for the donors' other tissues;
  applied to cohort samples it reproduces their fitted values exactly.
- **Donors are the unit.** GTEx takes 4.5 of these ten tissues per donor on
  average, so a donor's samples never sit on both sides. The reference set is
  148 whole donors (671 samples, COMBINED's 16% share); the five
  member/non-member splits are a donor-grouped 5-fold partition of the other
  775 donors (3,497 samples), balanced on tissue: about 700 non-members and
  2,800 members per split. Checked: no donor has samples on both sides of a
  split, and no reference donor is a candidate.
- **Second samples.** Every other GTEx sample of the cohort's donors (15,439)
  went through the same frozen VST: 2,039 of the same organ at another site
  (sigmoid colon, oesophageal muscularis and gastro-oesophageal junction,
  unexposed skin, kidney medulla), 12,445 of other organs, 955 cell lines.

The membership game is therefore donor-level: a member is a donor whose
samples of these ten tissues trained the generator.

## Runs

`scripts/queue_gtex.sh` (GPU 0, niced, pinned to cores 12-23 and 36-47):

1. targets: MVN, CVAE, NoisyDiffusion, the CAMDA-26 DP-PGM (`pgg`) and the
   corrected DP-PGM, five splits;
2. every attack that needs no shadow models, at the COMBINED grid's settings
   (`configs/experiments/grid_gtex.yaml`);
3. the donor-linked attack with another tissue (`scripts/perturb/gtex_donor_linked.py`);
4. MeLoMIA-CVAE, then MeLoMIA-ND, with per-record calibration and 20
   synth-shadows (`configs/experiments/melomia_prc_gtex.yaml`).

Not planned yet: TabSyn and TabPFN targets (hours to days per split), and
MeLoMIA-TabSyn.

## Results

First check (MVN, split 1): MahalaMIA with the ridge reaches AUC 0.976 with the
reference set and 0.816 without one (COMBINED, five splits: 0.89 and 0.68).
