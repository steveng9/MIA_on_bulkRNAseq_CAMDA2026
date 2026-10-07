# Donor-linked attacks and attacks under mismatched auxiliary data

Two experimental directions added on 2026-10-07. The first asks whether a
sample that was never in the training data can reveal that its donor was. The
second runs the three experiments Hakime proposed in
`Hakimes_experiment_ideas.txt` (auxiliary data from another cohort, auxiliary
data under another normalisation, and an adversary with only some of the
genes). Every number below is a mean over the five canonical splits. The full
tables are in `results/perturb/TABLES.md`; this file says how each experiment
was set up and how to read it.

Nothing here retrains a target. All four experiments change what the attacker
holds and leave the released synthetic data as it was.

## 0. Data added

**TCGA raw counts.** The challenge distributed only VST values for 978 genes.
Both directions need raw counts, so `scripts/external/gdc_download.py` pulled
the 6,121 open STAR-count files of the twelve TCGA projects from the GDC API
(26 GB read, kept as 4 GB of count and TPM matrices in `~/data/TCGA_GDC`).
Every aliquot of the challenge (1,089 BRCA; 4,323 + 824 COMBINED) was found by
its barcode.

**The challenge's VST, recovered exactly.** DESeq2's parametric VST is a closed
form with two fitted constants plus one size factor per sample.
`scripts/external/tcga_vst.py` identifies the two constants from the
distributed values themselves and recomputes the size factors from the counts.
Re-deriving every distributed sample from its counts then agrees with the
challenge files to 5.5e-14 on both cohorts. Two facts follow. BRCA was fitted
on its 1,089 samples alone. COMBINED was fitted on the 4,323 candidates and the
824 reference samples together, so the reference set as distributed shares the
candidates' transform exactly. The recovered transform can be frozen and
applied to a sample the challenge never distributed, which is what puts new
samples in the same space as the released data.

**GSE58135.** 84 primary breast tumours (42 triple-negative, 42 ER+/HER2-)
from Varley et al. 2014, NCBI's recount, all 978 landmark genes present after
mapping Entrez to Ensembl ids with the challenge's dictionary
(`scripts/external/gse58135_prepare.py`, `~/data/GSE58135`).

## 1. Donor-linked membership inference

### The question

In the single-cell project the "disjoint cells" experiment trained the
generator on some cells of a donor and attacked with other cells of the same
donor. The bulk analogue needs two samples per donor. TCGA has them: besides
the one primary-tumour aliquot per donor that the challenge distributed
(sample A), the GDC holds a second RNA-seq aliquot (sample B) for 465 of the
4,323 COMBINED donors and 119 of the 1,089 BRCA donors. No new dataset was
needed.

The generators trained on the A samples of the member donors. The adversary
holds a B sample and asks whether its donor was in the training data. B samples
fall into four tiers of biological distance from A.

| tier | what B is | COMBINED | BRCA |
|---|---|---|---|
| same tumour sample, other aliquot | the same piece of tissue, extracted or sequenced again | 25 | 3 |
| same tumour, other vial | another piece of the same primary tumour | 47 | 6 |
| other lesion | a metastasis, a recurrence or a new primary | 25 | 7 |
| matched normal | adjacent non-tumour tissue of the same organ | 410 | 111 |

### Setup

Each B sample is put in the cohort's VST space with the frozen transform,
appended to the candidate pool one tier at a time, and scored against the
existing targets (`scripts/perturb/donor_linked.py`). The class label given to
a B sample is its donor's. Three numbers are reported per cell
(`scripts/perturb/donor_linked_report.py`).

*Linked AUC* separates B samples of member donors from B samples of non-member
donors. The five splits are a 5-fold partition, so every donor is a non-member
in exactly one split and a member in the other four; the splits are pooled
after replacing each score by its percentile among that split's non-member
candidates.

*Overlap AUC* is the control: the same donors' A samples, member against
non-member. It is ordinary membership inference restricted to the donors that
have a B sample.

*Within donor* uses each donor as its own control. For one B sample it is the
share of its four member splits in which it scores above its one non-member
split, averaged over donors; 0.5 means the donor's membership does not move the
second sample's score. Intervals are 95% bootstrap intervals over donors.

A shuffled control (`donor_linked_control.py`) gives each B sample another
donor's membership pattern. Over 200 shuffles the linked AUC falls to 0.50 to
0.51, and every positive cell reported below exceeds the largest of its 200
shuffled values.

### Result 1: a second tumour sample reveals the donor's membership

TCGA-COMBINED, the three tumour tiers pooled (97 B samples, 70 donors).

| generator | attack | linked AUC | overlap AUC | within donor |
|---|---|---|---|---|
| MVN | MahalaMIA (ridge) | 0.65 [0.61, 0.69] | 0.87 [0.83, 0.92] | 0.86 [0.80, 0.90] |
| CVAE | MahalaMIA (ridge) | 0.65 [0.62, 0.69] | 0.82 [0.78, 0.88] | 0.78 [0.71, 0.84] |
| NoisyDiffusion | MahalaMIA (ridge) | 0.57 [0.53, 0.61] | 0.69 [0.66, 0.74] | 0.72 [0.65, 0.78] |
| DP-PGM (CAMDA-26) | MAMA-MIA v2 | 0.55 [0.52, 0.59] | 0.64 [0.59, 0.69] | 0.59 [0.52, 0.66] |
| DP-PGM (new, eps = 10) | MAMA-MIA v2 | 0.52 [0.50, 0.56] | 0.62 [0.60, 0.66] | 0.57 [0.49, 0.64] |

So about two fifths of the attack's advantage over chance survives when the
attacker's sample was never trained on (for MVN, 0.15 of 0.37). BRCA agrees on its 16 tumour B samples (MVN 0.68
[0.61, 0.77], CVAE 0.65 [0.57, 0.75], NoisyDiffusion 0.62 [0.57, 0.70]), with
wide intervals.

The linked AUC is much lower than the within-donor figure because the two
measure different things. A B sample's score depends mostly on how typical that
sample is, and the donor's membership moves it by less than samples differ from
one another. Within one donor the move is consistent: in COMBINED/MVN a second
tumour sample scores higher in a split where its donor is a member 86% of the
time. An attack that calibrates each record against itself would see more of
this, which is what MeLoMIA's per-record calibration does (section 1, last
part).

By tier (COMBINED, MahalaMIA ridge, linked AUC):

| tier | n | MVN | CVAE | NoisyDiffusion |
|---|---|---|---|---|
| same tumour sample, other aliquot | 25 | 0.74 [0.67, 0.83] | 0.76 [0.67, 0.84] | 0.60 [0.57, 0.66] |
| other lesion | 25 | 0.74 [0.66, 0.83] | 0.71 [0.63, 0.80] | 0.69 [0.62, 0.77] |
| same tumour, other vial | 47 | 0.58 [0.55, 0.63] | 0.62 [0.57, 0.68] | 0.51 [0.43, 0.58] |
| matched normal | 410 | 0.51 [0.50, 0.52] | 0.50 [0.49, 0.52] | 0.50 [0.50, 0.51] |

### Result 2: the leak follows how recognisable the second sample is

Without any generator, ask where a B sample's own A sample ranks among all
4,323 COMBINED candidates by correlation.

| tier | own donor is the nearest candidate | own donor in the top 10 |
|---|---|---|
| same tumour sample, other aliquot | 68% | 92% |
| other lesion | 68% | 76% |
| same tumour, other vial | 19% | 47% |
| matched normal | 1.7% | 5% |

The tiers order the same way in the attack table. A generator that memorises A
leaks about anything close to A, and a metastasis is as close to its primary as
a second aliquot is. The "other vial" tier is less recognisable than a
metastasis, which was not what I expected; these are mostly `01B` vials and may
differ in preservation, which I have not checked.

### Result 3: matched normal tissue does not leak, as attacked here

410 matched normals give tight intervals around 0.50 for every generator and
every attack (table above; the widest claim the data allow is an AUC below
0.52). A normal sample resembles other normals far more than it resembles its
donor's tumour.

That is a statement about these attacks and not yet about the data. Two further
checks say a weak donor signal is present in a normal sample.

First, after subtracting class means and projecting out the 400 leading
principal components of the candidate pool, a normal sample's own donor is the
single nearest of 4,323 candidates in 12% of cases and in the top ten in 27%
(`scripts/perturb/donor_linkability.py`). Chance is 0.02%. The shared programmes
(tissue, purity, proliferation) hide a donor-specific residue, presumably
germline regulation.

Second, an adaptive adversary who shifts the normals onto the release's class
means before attacking (`donor_linked.py --adapt`; it needs only a handful of
normals of the organ and the labelled release) moves two cells off chance on
COMBINED: MahalaMIA against NoisyDiffusion 0.52 [0.51, 0.53], within donor 0.58
[0.55, 0.61], and against MVN 0.52 [0.51, 0.53], within donor 0.56
[0.52, 0.59]. CVAE, both DP-PGMs and all of BRCA (111 normals) stay at chance.
I would call this suggestive. It is two cells out of many, at an AUC nobody
could act on.

### Result 4: DP-PGM

The geometric attacks see nothing through either DP-PGM, linked or not. MAMA-MIA
v2 keeps a small linked signal against the CAMDA-26 release (0.55), concentrated
in the re-sequenced aliquots (0.64 [0.58, 0.71], as high as its overlap AUC of
0.63). That release is not differentially private end to end. Against the
corrected generator at eps = 10 the linked interval reaches 0.50.

### MeLoMIA

MeLoMIA's stack is built around the candidate pool, so B samples are read
through the stack that already exists rather than appended
(`scripts/perturb/donor_linked_melomia.py`): every saved synth-shadow scores B,
a proxy is retrained on the release, B is calibrated against its own
signatures under the synth-shadows, and the saved meta-classifiers score it.
The same pass re-scores the candidates and reproduces the recorded AUCs (BRCA
NoisyDiffusion 0.97 to 0.99 per split, CVAE 0.78 to 0.83).

Results: still running on a shared GPU when this was written (BRCA is too
small to say anything: 16 tumour B samples). To be filled in from
`results/perturb/donor_linked_melomia.csv`.

### What this does and does not show

It shows that in bulk RNA-seq, as in single-cell, "the adversary must hold the
exact training record" is not a safe assumption: a second biopsy, a
re-sequenced aliquot or a later metastasis of a training donor carries about
two fifths of the membership signal. It does not show leakage through normal tissue,
blood, or a sample taken years apart, and TCGA cannot answer those. The numbers
rest on 70 donors for the tumour tiers.

Not done, in order of value:

1. More power. Every donor is a non-member in one split only. Redrawing many
   80/20 splits that place the donors with a B sample half in and half out, and
   retraining the cheap generators (MVN 6 s, CVAE 45 s per target), would
   measure each donor in both states many times.
2. A cohort with repeated samples by design. Candidates: GTEx (one donor, many
   tissues; 948 donors), longitudinal blood cohorts, CPTAC. This is the test of
   whether a sample of a different tissue links to a donor.
3. An attack built for the normal-tissue case: remove shared structure first,
   then score.
4. Training on B and attacking with A, and training on both.

## 2. Hakime's experiment 1: an external cohort as auxiliary data (BRCA)

### Setup

TCGA-BRCA ships no auxiliary set. Until now the repository either ran BRCA
attacks without one or, for the attacks that require one, let non-member
candidates stand in (marked "optimistic"). GSE58135 replaces that with a real
outside cohort (`scripts/perturb/aux_mismatch.py`). Conditions:

- no auxiliary set;
- GSE58135 as TPM (Hakime's condition 1), and as log2(TPM + 1);
- GSE58135 through a VST fitted on its own 84 samples with the challenge's
  recipe (condition 2);
- GSE58135 through TCGA-BRCA's own fitted VST, the closest an outside cohort
  can get to the release's representation;
- a matched control: 84 non-member candidates of the split, class-stratified,
  used as the auxiliary set and left out of every metric, so all conditions
  score the same candidates. Same cohort, same transform, same size as
  GSE58135;
- each GSE58135 variant after the adversary quantile-maps every gene onto the
  release's distribution of that gene. This repair needs only the release.

MAMA-MIA takes its population estimate from the candidate pool and MeLoMIA
trains on it, so neither reads the auxiliary set and neither changes here.

### Result

MahalaMIA with the ridge, AUC:

| auxiliary set | MVN | CVAE | NoisyDiffusion | TabSyn | TabPFN |
|---|---|---|---|---|---|
| none | 1.000 | 0.969 | 0.817 | 0.674 | 0.563 |
| GSE58135, TPM | 1.000 | 0.969 | 0.818 | 0.675 | 0.564 |
| GSE58135, log2(TPM+1) | 1.000 | 0.974 | 0.831 | 0.684 | 0.569 |
| GSE58135, own VST | 1.000 | 0.984 | 0.857 | 0.700 | 0.578 |
| GSE58135, TCGA's VST | 1.000 | 0.986 | 0.862 | 0.704 | 0.582 |
| GSE58135, quantile-aligned to the release | 1.000 | 0.996 | 0.951 | 0.742 | 0.645 |
| TCGA held-out 84 (matched control) | 1.000 | 0.998 | 0.971 | 0.762 | 0.702 |

Three readings. A mismatched auxiliary set never drives the attack below its
no-auxiliary level; in the worst case (raw TPM) it is ignored. Matching the
normalisation of an outside cohort recovers a little (NoisyDiffusion 0.817 to
0.862), and most of the remaining gap to the matched control is cohort
difference. And the adversary can close most of that gap alone: quantile-aligning
the outside cohort to the release takes NoisyDiffusion to 0.951 against 0.971
for the matched control, from any starting normalisation (the mapping is
rank-based, so TPM and log-TPM give identical results and VST nearly so).

RedSigma does not move at all (CVAE 0.80 to 0.81 in every condition). The
calibrated baselines are a different story: GAN-leaks calibrated does better
with the *worse* auxiliary set (CVAE 0.65 with log-TPM, 0.55 with TCGA's VST,
0.71 with no calibration at all), because a far-away reference cancels nothing
and the attack falls back to plain GAN-leaks. Its calibration hurts on this
cohort.

Against both DP-PGMs the auxiliary-using attacks stay at 0.50 to 0.53 in every
condition.

For BRCA this also settles the "optimistic" question for the attacks tested:
the headline MahalaMIA numbers do not depend on borrowing non-members.

## 3. Hakime's experiment 2: the same auxiliary samples, re-normalised (COMBINED)

### Setup

The 824 reference samples keep their identity and change representation: as
distributed (VST, fitted jointly with the candidates), VST refitted on the 824
alone, log2(TPM + 1), TPM (GDC's `tpm_unstranded`), each also quantile-aligned
to the release, and 84 of the 824 as distributed, to separate the effect of
size.

### Result

MahalaMIA with the ridge, AUC:

| auxiliary set | MVN | CVAE | NoisyDiffusion | TabPFN (1 split) |
|---|---|---|---|---|
| none | 0.682 | 0.711 | 0.592 | 0.527 |
| reference, VST as distributed | 0.893 | 0.798 | 0.769 | 0.601 |
| reference, VST refitted on itself | 0.906 | 0.807 | 0.777 | 0.601 |
| reference, log2(TPM+1) | 0.717 | 0.738 | 0.606 | 0.535 |
| reference, TPM | 0.685 | 0.714 | 0.593 | 0.528 |
| reference, TPM, quantile-aligned to the release | 0.899 | 0.803 | 0.763 | 0.611 |
| reference as distributed, 84 samples | 0.867 | 0.777 | 0.707 | 0.582 |

On COMBINED the auxiliary set matters far more than on BRCA (0.68 without,
0.89 with, against MVN), and a normalisation mismatch removes all of that gain:
with TPM the attack is back at its no-auxiliary level. Refitting the VST costs
nothing, so the joint fit of the distributed reference set gave the attacker no
advantage. The quantile alignment again restores everything (0.899 against
0.893). Eighty-four matched samples do better than 824 mismatched ones.

RedSigma (CVAE 0.85, MVN 0.68) is unchanged by normalisation. With 84 reference
samples its CVAE rule returns a constant (0.500): the rule zeroes the first 100
principal components and 84 samples supply only 83.

The answer to the reviewer question behind experiments 1 and 2 is therefore
that the attack is not fragile to the auxiliary data's representation. A
mismatch reduces the attack to its no-auxiliary form, which is already 1.00 /
0.97 on BRCA MVN / CVAE, and an attacker who notices the mismatch can undo it
with the release alone.

## 4. Hakime's experiment 3: reduction dimension and gene subsets

### Setup

Two blocks (`scripts/perturb/gene_subsets.py`).

*Baselines.* DOMIAS-KDE, GAN-leaks calibrated and LOGAN-D1 exactly as the
challenge baseline runs them (each set on its own StandardScaler, PCA fitted on
the reference set, and the three gene selections vDE, sDE, dDE), with the
dimension swept over 10 to 800 instead of fixed at 100.

*Our attacks.* The adversary holds only k genes of each candidate. The release
and the auxiliary set are cut to the same k genes; the generator still trained
on all 978. Selection rules: the baseline's three, the most variable genes in
the release, and a random set.

### Results

**The baseline's fixed dimension understates DOMIAS.** COMBINED, DOMIAS-KDE with
PCA:

| components | 10 | 25 | 50 | 100 | 200 | 400 | 600 | 800 |
|---|---|---|---|---|---|---|---|---|
| MVN | 0.512 | 0.528 | 0.551 | 0.590 | 0.653 | 0.806 | 0.727 | 0.543 |
| CVAE | 0.545 | 0.646 | 0.701 | 0.685 | 0.645 | 0.641 | 0.668 | 0.713 |
| NoisyDiffusion | 0.515 | 0.521 | 0.525 | 0.539 | 0.560 | 0.601 | 0.689 | 0.565 |

At the baseline's 100 components DOMIAS reads 0.59 against MVN; at 400 it
reads 0.81. The best dimension differs by generator (400 for MVN, 600 for NoisyDiffusion,
and two peaks for CVAE at 50 and 800), so no single setting is fair to all
three, and the baseline row of a results table depends on this choice as much
as on the method. LOGAN-D1 is within 0.03 of chance at every dimension.

**The vDE selection ranks numerical noise.** The baseline standardises the
release and the reference set separately and then ranks genes by the ratio of
their variances, which is 1 for every gene after standardisation. The order it
returns is floating-point residue. Its curves are still within 0.04 of sDE's
and dDE's at every dimension, because (next result) the choice of genes hardly
matters.

**Our attacks degrade smoothly, and which genes are kept does not matter.**
AUC with a random k genes:

| attack on generator | 10 | 50 | 100 | 200 | 400 | 600 | 800 | 978 |
|---|---|---|---|---|---|---|---|---|
| BRCA MahalaMIA (ridge) on MVN | 0.522 | 0.538 | 0.573 | 0.650 | 0.815 | 0.963 | 0.999 | 1.000 |
| BRCA MahalaMIA (ridge) on CVAE | 0.520 | 0.541 | 0.579 | 0.655 | 0.800 | 0.912 | 0.956 | 0.970 |
| BRCA MahalaMIA (ridge) on NoisyDiffusion | 0.522 | 0.537 | 0.567 | 0.620 | 0.703 | 0.764 | 0.804 | 0.821 |
| BRCA GAN-leaks on TabSyn | 0.594 | 0.742 | 0.764 | 0.768 | 0.770 | 0.772 | 0.772 | 0.773 |
| COMBINED MahalaMIA (ridge) on MVN | 0.510 | 0.542 | 0.589 | 0.671 | 0.770 | 0.812 | 0.805 | 0.893 |
| COMBINED MahalaMIA (ridge) on CVAE | 0.509 | 0.540 | 0.592 | 0.664 | 0.761 | 0.784 | 0.768 | 0.798 |
| COMBINED RedSigma on CVAE | 0.500 | 0.500 | 0.500 | 0.667 | 0.793 | 0.829 | 0.843 | 0.847 |
| COMBINED MAMA-MIA v2 on DP-PGM (CAMDA-26) | 0.523 | 0.548 | 0.567 | 0.589 | 0.620 | 0.638 | 0.650 | 0.659 |

The two kinds of attack behave differently. The covariance attack needs most of
the genes: its signal is in the low-variance directions of the full covariance
(FINDINGS sections 2 and 6), and those only exist once the gene count approaches the
sample count. With 100 genes it is at 0.57. The nearest-neighbour attacks
saturate early: GAN-leaks on TabSyn has 96% of its full-panel AUC at 50 genes.
So withholding genes protects against one family and not the other. The
selection rules differ by 0.01 in AUC on average (0.065 at most), and a random
set is never more than 0.022 below the best rule: an adversary gains little by
choosing genes cleverly and loses little by holding an arbitrary panel.

RedSigma's CVAE rule on COMBINED returns a constant below 200 genes for the
same reason as above (it zeroes 100 principal components).

## 5. Not done, and where Steven's call is needed

- **MeLoMIA under experiments 1 to 3.** Its shadows are trained on the
  candidate pool, so experiments 1 and 2 reach it only through an optional
  reference calibration that the per-record calibration has replaced; I expect
  no change and did not run it. Experiment 3 would need a full shadow stack per
  gene subset (30 base shadows and 30 synth-shadows; several GPU-hours for
  MeLoMIA-ND on BRCA, about four times that on COMBINED). Say if one or two
  subset sizes are worth that.
- **Candidates in a mismatched representation.** Hakime's text mismatches the
  auxiliary data only. The harder case, where the record under attack is itself
  TPM or from another pipeline, is not run. The donor-linked experiment is a
  version of it (the B samples come through a frozen transform), and the
  machinery is the same.
- **PAM50 labels for GSE58135.** The series gives receptor status only.
  Attacks that need a class for auxiliary samples label them with the nearest
  class of the release, as they already do for the COMBINED reference set.
- **TabPFN on COMBINED** has one split, so its column is a single run.
- **The donor-linked follow-ups** listed at the end of section 1.

## 6. For the paper

Suggested placement: the donor-linked result in the threat-model part of the
experiments section (one table: the five rows of Result 1, one sentence on
normals), experiments 1 to 3 as a robustness subsection with the two MahalaMIA
tables and the gene-count curve, the rest in the appendix.

Future-work text, if the donor-linked follow-ups are not run before submission:

> Our donor-linked analysis is limited to the second samples TCGA happens to
> hold: 97 tumour samples from 70 donors, and matched normal tissue of the same
> organ. Whether a sample of another tissue, or one taken years later, links a
> donor to a training set needs a cohort with repeated sampling by design, such
> as GTEx. Normal tissue did not leak under our attacks, but after removing
> shared expression programmes a normal sample's nearest neighbour among 4,323
> tumours is its own donor's tumour 12% of the time, so an attack built for
> that case may succeed where ours did not.

## Reproducing

```
python scripts/external/gdc_download.py                 # ~15 min, 26 GB read
python scripts/external/tcga_vst.py                     # recover + verify the VST, write B samples
~/.venvs/pydeseq2/bin/python scripts/external/gse58135_prepare.py \
    --tcga-fit ~/data/TCGA_GDC/processed/BRCA_vst_fit.json
~/.venvs/pydeseq2/bin/python scripts/external/reference_vst_own.py

python scripts/perturb/donor_linked.py --dataset COMBINED      # also BRCA; --adapt
python scripts/perturb/donor_linked_report.py                  # --suffix=_adapted | _melomia
python scripts/perturb/donor_linked_control.py
python scripts/perturb/donor_linkability.py
python scripts/perturb/aux_mismatch.py --dataset BRCA          # also COMBINED
python scripts/perturb/gene_subsets.py --dataset COMBINED --logan
python scripts/perturb/report.py                               # results/perturb/TABLES.md
```

`mia/views.py` is the piece the three perturbation scripts share: a context
manager that changes the auxiliary set, the gene set or the candidate pool for
one block and leaves the targets untouched. Only attacks that keep nothing on
disk may run inside it. Set `CAMDA_ARTIFACTS` to the main checkout's
`artifacts/` when running from a worktree. pydeseq2 lives in its own
virtualenv (`~/.venvs/pydeseq2`) so the project environments are unchanged.
