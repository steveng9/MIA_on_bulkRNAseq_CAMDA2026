# Reproducing the CAMDA 2026 abstract

What this repo's pipeline produces, next to the numbers in
`main_bulkRNA_CAMDA26_abstract.tex`.  The point of the comparison is to make the
differences visible rather than to hide them: some are expected consequences of
a cleaner setup, and at least one is a genuine disagreement worth chasing.

*Status: the BRCA grid is complete (all four attacks x all four generators,
as-submitted and tuned).  The synth-shadow ablation and the COMBINED MeLoMIA
rows are running.  This file is updated as cells land.*

## What is and is not the same

**Same.** The five 80/20 partitions, taken from the blue team's published splits
YAML, so membership labels match exactly.  The NoisyDiffusion column reuses the
blue team's published synthetic data, generated from those same partitions.
Metric definitions: AUC-ROC, AUPR, TPR at 1% and 10% FPR, averaged over splits.

**Different by necessity.**  The MVN, CVAE and DP-PGM targets are retrained here
rather than taken from the challenge platform, because the platform's synthetic
datasets come with no ground-truth membership labels for internal evaluation.
Hyperparameters follow the blue-team configs (MVN noise 0.7, CVAE 10k iterations
at batch 64 with z=128 and beta=0.001, DP-PGM at eps=10 with 4 bins in joint
mode), so the generators are the same *methods*, but not bit-identical models.
Numbers for those three columns should be close, not equal.

**Different by choice.**  MeLoMIA runs with K=30 shadows on BRCA rather than the
5 the abstract used — 5 was inherited from the blue team's published split count
rather than chosen, and it is thin for a meta-classifier.  See TODO item 2.

## BRCA

### MahalaMIA vs MVN — reproduced

| | AUC | AUPR | T@1 | T@10 |
|---|---|---|---|---|
| abstract | 0.922 | 0.981 | 0.698 | 0.825 |
| this repo, as submitted (pseudo-inverse) | 0.928 | 0.982 | 0.631 | 0.792 |
| **this repo, tuned (ridge 1e-4)** | **1.000** | — | — | — |

The as-submitted row reproduces the abstract within noise across all four
metrics.  The tuned row is new: replacing the pseudo-inverse of the synthetic
covariance with a lightly ridge-regularised full inverse gives essentially
perfect membership inference against the MVN generator.

That is a large enough jump to deserve suspicion, so it was checked:
permuting the membership labels drops the AUC to 0.48, and scoring split 1's
target against split 2's labels gives 0.37 — both what a leak-free attack should
do (`scripts/sanity_check.py`).  The mechanism is straightforward in hindsight.
The MVN generator's output is a draw from a Gaussian fitted to the training
split, so the synthetic covariance is an almost noiseless copy of the members'
covariance; the pseudo-inverse discards the near-null directions, and those
directions are where the fit is tightest.

### MAMA-MIA vs DP-PGM — reproduced

| | AUC | AUPR | T@1 | T@10 |
|---|---|---|---|---|
| abstract | 0.528 | 0.824 | 0.036 | 0.142 |
| this repo (4 bins) | 0.499 | 0.805 | 0.032 | 0.120 |
| this repo (2 bins) | 0.511 | — | — | — |

Both sit just above chance, which is the abstract's conclusion: DP at eps=10
leaves little for a marginal-ratio attack to find.  Ours is marginally lower;
our DP-PGM targets are our own, and the generator's marginal selection and
budget split differ in detail from the challenge's.

### Cross-generator results

The abstract reports MahalaMIA applied unmodified to the non-DP deep generators:
AUC 0.986 against CVAE on BRCA, and 0.489 against ND.  We reproduce the first
finding qualitatively and **not** the second.

| MahalaMIA vs | abstract | this repo (pinv) | this repo (ridge 1e-4) |
|---|---|---|---|
| CVAE | 0.986 | 0.891 | 0.970 |
| ND | 0.489 | **0.806** | **0.821** |

CVAE lines up: the generator preserves enough of the training covariance for a
purely statistical attack to beat the CVAE-specific loss attack.  ND does not.
The abstract has MahalaMIA at chance against NoisyDiffusion; we get 0.81, and
the same conclusion holds under every covariance variant we tried.  Since our ND
column *is* the blue team's published synthetic data, the target is not the
difference — the difference is upstream, in how the abstract's cross-model run
was set up.  Worth resolving before the paper repeats the "diffusion preserves
covariance less consistently" claim, because our numbers say the opposite.

### DP-PGM aside

MAMA-MIA's bin count was chosen to match the generator's own discretisation, and
against DP-PGM that is roughly right.  Against NoisyDiffusion it is badly wrong:
AUC rises monotonically with resolution (0.54 at 4 bins, 0.61 at 16, 0.68 at 32,
0.76 at 64).  At those bin counts the attack has stopped being a marginal attack
in the DP-PGM sense — with ~870 synthetic samples spread over 64 bins per gene,
the domain ratio is closer to a per-gene nearest-value detector — but it is a
real attack on a deep generative model, and a strong one.  See TODO.

### MeLoMIA on its own generator — both exceeded

| MeLoMIA-ND vs ND | AUC | AUPR | T@1 | T@10 |
|---|---|---|---|---|
| abstract (K=5) | 0.620 | 0.858 | 0.045 | 0.193 |
| **this repo (K=30)** | **0.858** | **0.959** | **0.329** | **0.622** |

| MeLoMIA-CVAE vs CVAE | AUC | AUPR | T@1 | T@10 |
|---|---|---|---|---|
| abstract (K=5) | 0.753 | 0.928 | 0.274 | 0.489 |
| **this repo (K=30)** | **0.798** | **0.943** | **0.329** | **0.518** |

Both exceed the abstract, and the shadow count is the only thing that changed —
same features, same splits, same metric definitions, and for ND the same
published synthetic data.  But the two gain very differently from it.  ND
improves by 0.238 AUC and multiplies TPR@1%FPR by 7.3x; CVAE improves by 0.045
and by 1.2x.

So K=5 was starving one attack rather than handicapping the method.  That is
worth saying explicitly in the paper, because the abstract's headline ordering
(CVAE attack stronger than ND attack, 0.753 vs 0.620) **reverses** at K=30:
0.798 vs 0.858.  The conclusion that the CVAE is the more vulnerable deep
generator does not survive giving the ND attack an adequate shadow budget.

### The full BRCA grid

Every attack against every generator, as-submitted configurations, five splits.
The abstract reports only the diagonal plus MahalaMIA's two cross-generator
cells; everything else here is new.

| AUC | MVN | CVAE | ND | DP-PGM |
|---|---|---|---|---|
| MahalaMIA | **0.928** | **0.891** | 0.806 | 0.497 |
| MAMA-MIA | 0.520 | 0.528 | 0.539 | 0.501 |
| MeLoMIA-CVAE | 0.553 | 0.798 | 0.563 | 0.491 |
| MeLoMIA-ND | 0.876 | 0.824 | **0.858** | 0.488 |

Two things in it are not in the abstract at all.

**MahalaMIA wins two of the four columns** — and after the covariance
conditioning of finding 2 it reaches 1.000 on MVN and 0.970 on CVAE.  A
closed-form distance with no shadow-model budget beats both learned attacks on
the two generators it beats them on, which makes it the more practically
alarming result.

**The two MeLoMIA backends behave completely differently off-diagonal.**
MeLoMIA-ND transfers (0.876 against MVN, a per-class Gaussian with no diffusion
process anywhere — above its own diagonal cell); MeLoMIA-CVAE does not (0.553
against MVN, 0.563 against ND).  `results/FINDINGS.md` section 3 works through
why.

Tables and figures for both the as-submitted and the tuned grid are in
`results/tables/grid_brca_BRCA.txt` and `results/tables/grid_brca_tuned_BRCA.txt`.

## COMBINED

The larger cohort (4,323 samples, 12 classes, plus a held-out auxiliary
reference set of 824) is not in the abstract, so there is nothing to reproduce
here -- it is the generalisation check.  Statistical attacks, five splits:

| AUC | MVN | CVAE | ND | DP-PGM |
|---|---|---|---|---|
| MahalaMIA (pinv, aux) | 0.900 | 0.619 | 0.770 | 0.500 |
| MAMA-MIA (4 bins) | 0.512 | 0.517 | 0.528 | 0.499 |

Two differences from BRCA matter.

**CVAE gets much harder; MVN barely does.**  MahalaMIA against the CVAE falls
from 0.891 to 0.619, while against the MVN generator it only moves from 0.928 to
0.900.  That is the expected direction and a useful separation: a VAE trained on
four times the data memorises far less per record, whereas a per-class Gaussian
fit is a summary statistic whose fidelity to its training split does not decay
with n.  The MVN generator's vulnerability is structural, not a small-sample
artefact, and that is the stronger claim for the paper.

**The NoisyDiffusion column stays high** (0.806 -> 0.770), which again contradicts
the abstract's 0.489 rather than explaining it.  The gap is not a cohort-size
effect.

**DP-PGM is at chance on both cohorts** for both statistical attacks, consistent
with the abstract.

MeLoMIA rows on COMBINED are queued at K=20 shadows.

## Reproducing this file

```bash
python scripts/build_targets.py --dataset BRCA
python scripts/run_experiment.py configs/experiments/grid_brca.yaml
python scripts/run_experiment.py configs/experiments/tune_statistical.yaml
python scripts/make_tables.py --config configs/experiments/grid_brca.yaml
python scripts/sanity_check.py --dataset BRCA
```
