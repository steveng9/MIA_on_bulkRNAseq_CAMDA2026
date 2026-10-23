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

### Shadow-free attacks, MVN / CVAE / NoisyDiffusion (2026-10-22)

`configs/experiments/grid_gtex.yaml`, five donor-grouped splits, AUC (mean over
splits; split-to-split standard deviation 0.02 or less except where noted) and
TPR at 1% FPR. COMBINED alongside, same attacks and settings
(`results/perturb/aux_mismatch.csv`, `grid_slides_extra_combined`,
`redsigma_combined`). DOMIAS is the corrected implementation (one PCA, 100
components, fitted on the reference set; `aux_mismatch.domias_baseline`); the
benchmark one in `mia/attacks/generic.py` scores 0.50 on every target here, as
on COMBINED (tracker 0e).

| attack | MVN | CVAE | NoisyDiffusion | COMBINED: MVN | CVAE | NoisyDiffusion |
|---|---|---|---|---|---|---|
| MahalaMIA, ridge 1e-4, reference set | **0.974** (0.61) | **0.870** (0.43) | **0.760** (0.14) | 0.893 (0.45) | 0.798 (0.33) | 0.769 (0.23) |
| MahalaMIA, ridge 1e-4, no reference set | 0.833 (0.22) | 0.866 (0.21) | 0.634 (0.08) | 0.682 (0.03) | 0.711 (0.02) | 0.592 (0.03) |
| MahalaMIA as submitted | 0.972 (0.57) | 0.653 +- 0.107 (0.11) | 0.756 (0.16) | 0.900 (0.41) | 0.619 (0.08) | 0.770 (0.20) |
| RedSigma | 0.829 (0.21) | 0.815 (0.31) | 0.504 (0.01) | 0.681 (0.03) | 0.847 (0.35) | 0.506 (0.01) |
| DOMIAS (corrected, 100 components) | 0.576 (0.03) | 0.762 (0.22) | 0.573 (0.05) | 0.59 | 0.69 | 0.54 |
| GAN-leaks | 0.528 (0.01) | 0.727 (0.14) | 0.539 (0.03) | 0.528 | 0.642 | 0.514 |
| GAN-leaks calibrated | 0.543 | 0.588 | 0.541 | 0.566 | 0.562 | 0.522 |
| MC | 0.512 | 0.557 | 0.522 | 0.507 | 0.531 | 0.507 |
| MAMA-MIA v1 | 0.512 | 0.516 | 0.523 | 0.512 | 0.517 | 0.528 |
| MAMA-MIA v2 | 0.509 | 0.523 | 0.531 | 0.508 | 0.517 | 0.547 |
| confidence (LR / RF), LOGAN | 0.50-0.52 | 0.50-0.52 | 0.50-0.51 | 0.50-0.51 | 0.50-0.51 | 0.50-0.51 |

Readings:

1. **GTEx is attacked about as easily as COMBINED, and more easily against MVN
   and CVAE.** The same attacks lead (MahalaMIA, RedSigma), the same ones fail
   (MAMA-MIA, the confidence attacks, LOGAN), and MahalaMIA as submitted is again
   unstable on CVAE (0.65 +- 0.11 here, 0.62 on COMBINED) where the ridge is not.
   One change of order: on CVAE, MahalaMIA now beats RedSigma (0.87 against
   0.82; COMBINED 0.80 against 0.85).
2. Part of the gap is cohort size: GTEx trains on about 2,800 samples, COMBINED
   on about 3,460, and MahalaMIA weakens as the training set grows towards the
   978 genes' covariance limit. COMBINED with its training set subsampled to
   2,800 gave MVN 0.93 and CVAE 0.84 (`scripts/cohort_size_sweep.py`, ridge
   1e-6), between COMBINED at full size and GTEx. The rest is not explained; normal tissue is more homogeneous than
   tumour, which would make a member stand out more, but that was not tested.
3. **The reference set matters less on GTEx.** Without it, MahalaMIA still
   reaches 0.83 on MVN and 0.87 on CVAE (COMBINED: 0.68 and 0.71); on CVAE the
   reference set adds nothing.
4. NoisyDiffusion is the hardest target on both cohorts, at the same level
   (0.76 against 0.77).

First check (MVN, split 1), before the grid: MahalaMIA with the ridge 0.976 with
the reference set and 0.816 without.

### Both DP-PGMs, and MeLoMIA-CVAE (2026-10-23)

The CAMDA-26 release (`pgg`) and the corrected DP-PGM (16 DP bins, epsilon 10),
five splits, AUC (TPR at 1% FPR); COMBINED in brackets.

| attack | DP-PGM, CAMDA-26 release | DP-PGM, corrected |
|---|---|---|
| MAMA-MIA v2 | **0.631** (0.03) [0.659] | **0.648** (0.02) [0.649] |
| MAMA-MIA v1 | 0.589 [0.620] | 0.517 [0.520] |
| every other shadow-free attack, corrected DOMIAS included | 0.50-0.51 | 0.50-0.51 |

The same picture as COMBINED: only MAMA-MIA reads a DP-PGM, at the same level.

MeLoMIA-CVAE (K=20, per-record calibration, `melomia_prc_gtex.yaml`): CVAE
**0.881** (TPR at 1% FPR 0.39; COMBINED 0.870), MVN 0.557, NoisyDiffusion 0.598.
It is the strongest attack on the GTEx CVAE, one point above MahalaMIA (0.870).

Corrected DOMIAS is `scripts/perturb/gtex_domias.py` (the grid's DOMIAS is the
uncorrected benchmark version).

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

### Another tissue of a training donor, canonical splits, every generator

`scripts/perturb/gtex_donor_linked.py`: the five canonical targets, the
shadow-free attacks, no calibration; each of a candidate donor's other-tissue
samples is scored against the target and labelled with its donor's membership.
Per-tissue tables for every attack: `results/perturb/TABLES.md`, section "GTEx:
another tissue of a training donor"; rows in
`results/perturb/gtex_donor_linked.csv`. MahalaMIA with the ridge and the
reference set, AUC, mean over splits:

| attacked sample | MVN | CVAE | NoisyDiffusion |
|---|---|---|---|
| the trained-on samples themselves | 0.974 | 0.870 | 0.760 |
| Skin, not sun-exposed | 0.621 | 0.578 | 0.556 |
| Colon, sigmoid | 0.605 | 0.588 | 0.555 |
| Oesophagus, gastro-oesophageal junction | 0.577 | 0.553 | 0.535 |
| Oesophagus, muscularis | 0.564 | 0.547 | 0.542 |
| Adipose, subcutaneous | 0.617 | 0.592 | 0.561 |
| Nerve / artery / thyroid | 0.53-0.54 | 0.52-0.53 | 0.52-0.53 |
| Blood, muscle, testis, fibroblasts | 0.50-0.51 | 0.51-0.52 | 0.50-0.51 |

Readings:

1. The MVN column reproduces the 100-redraw estimate (unexposed skin 0.621
   against 0.620, adipose 0.617 against 0.612), so five splits suffice here.
2. The leak is there for every generator and falls with how well the attack
   does on the trained-on samples: averaged over the four same-organ tissues,
   CVAE keeps about three quarters of MVN's excess over 0.5 (0.567 against
   0.592), NoisyDiffusion about half (0.547).
3. It is weak per donor: TPR at 1% FPR on unexposed skin is 0.05 (MVN), 0.03
   (CVAE, NoisyDiffusion). The other tissue shifts the whole distribution a
   little; it does not single out a few donors.
4. Without the reference set the leak halves (MVN unexposed skin 0.555); RedSigma
   sees less of it (0.55 at best), and GAN-leaks and MAMA-MIA none (0.49-0.52).
5. Against both DP-PGMs another tissue is at chance for every attack
   (0.49-0.51), including MAMA-MIA v2, which reaches 0.63 and 0.65 on the
   trained-on samples.
