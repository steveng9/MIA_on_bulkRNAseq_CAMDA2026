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

### Another tissue of a training donor (MVN, 100 redrawn splits)

`scripts/perturb/gtex_donor_linked_resplit.py`, the design of
`donor_linked_resplit.py` (TCGA) on GTEx: MVN retrained on 100 donor-level
splits (each candidate donor in with probability 0.8, all of its ten-tissue
samples together), every other-tissue sample of the 775 candidate donors scored
in every repetition. MahalaMIA with the ridge and the reference set. The
calibrated column z-scores each sample against its own scores in the
repetitions where its donor was out: ideal shadow models, so an upper bound on
what shadow modelling could reach. 95% donor-bootstrap intervals are within
0.02 of every number below; a label-shuffled control scores 0.50 in every row.

| attacked sample | samples | AUC | calibrated AUC |
|---|---|---|---|
| the trained-on samples themselves | 3,497 | 0.975 | 1.000 |
| same organ, another site (all) | 1,714 | 0.590 | 0.697 |
| Skin, not sun-exposed (trained: sun-exposed) | 543 | 0.620 | 0.771 |
| Colon, sigmoid (trained: transverse) | 350 | 0.604 | 0.725 |
| Oesophagus, gastro-oesophageal junction (trained: mucosa) | 340 | 0.571 | 0.656 |
| Oesophagus, muscularis (trained: mucosa) | 471 | 0.566 | 0.643 |
| another organ (all) | 10,524 | 0.520 | 0.554 |
| Adipose, subcutaneous | 586 | 0.612 | 0.730 |
| Nerve, tibial | 554 | 0.542 | 0.582 |
| Artery, tibial | 576 | 0.530 | 0.579 |
| Thyroid | 571 | 0.528 | 0.568 |
| Whole blood | 665 | 0.507 | 0.508 |
| Muscle, skeletal | 678 | 0.502 | 0.509 |
| Testis | 356 | 0.503 | 0.506 |
| Cultured fibroblasts (cell line) | 531 | 0.508 | 0.516 |

Readings:

1. **A different sample of the same organ identifies the donor**, as a second
   tumour sample did on TCGA (COMBINED, same design, separate tumour sample:
   0.64 uncalibrated, 0.78 calibrated). Unexposed skin against a generator
   trained on sun-exposed skin: 0.62, and 0.77 with calibration. This is a
   stronger leak than TCGA's matched normal tissue (0.50 / 0.53), the other
   normal-tissue case we had.
2. **The leak follows tissue similarity, not organ labels.** Subcutaneous
   adipose leaks as much as a same-organ sample (0.61 / 0.73), presumably
   because breast tissue, one of the ten trained tissues, is mostly fat (not
   checked). Nerve, artery and thyroid leak a little (0.57 to 0.58
   calibrated). Blood, muscle, testis and cultured fibroblasts stay at chance
   even with ideal calibration.
3. What carries the signal is not separated: shared biology of the donor
   (genotype, sex, age), shared handling (ischaemic time, death circumstances)
   or shared processing batches. For the privacy question it does not matter
   which: each links the sample to the donor.

The same question on the canonical five splits, for every generator and the
shadow-free attacks without calibration, is `scripts/perturb/gtex_donor_linked.py`
(queued; `results/perturb/gtex_donor_linked.csv`).
