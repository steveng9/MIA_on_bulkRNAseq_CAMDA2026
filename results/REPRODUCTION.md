# Reproducing the CAMDA 2026 abstract

What this repo's pipeline produces, next to the numbers in
`main_bulkRNA_CAMDA26_abstract.tex`.  The point of the comparison is to make the
differences visible rather than to hide them: some are expected consequences of
a cleaner setup, and at least one is a genuine disagreement worth chasing.

*Status: BRCA statistical attacks and the MeLoMIA-ND row complete; the
MeLoMIA-CVAE row and the COMBINED MeLoMIA rows are running.  This file is
updated as cells land.*

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

### MeLoMIA-ND vs NoisyDiffusion — exceeded

| | AUC | AUPR | T@1 | T@10 |
|---|---|---|---|---|
| abstract (K=5 shadows) | 0.620 | 0.858 | 0.045 | 0.193 |
| **this repo (K=30 shadows)** | **0.858** | **0.959** | **0.329** | **0.622** |

This is the headline change from more shadow models.  The abstract's K=5 was
inherited from the blue team's five published splits rather than chosen, and a
five-model pool is thin for a meta-classifier over 15 timesteps x 600 noise
draws.  Raising it to 30 moves TPR at 1% FPR from 0.045 to 0.329 — 7.3x — and at
10% FPR from 0.193 to 0.622.  The low-FPR end is where the improvement is
concentrated, which is the end that matters for a privacy claim: the attack goes
from flagging 1 in 22 members at a 1% false-positive budget to flagging 1 in 3.

Nothing else about the attack changed.  Same features, same target synthetic
data (the blue team's published ND datasets), same five splits, same metric
definitions.  K is the only knob that moved, so the abstract's ND number should
be read as a property of its shadow budget rather than of the generator.

### MeLoMIA-ND applied across generators — it is not diffusion-specific

| MeLoMIA-ND vs | MVN | CVAE | ND | DP-PGM |
|---|---|---|---|---|
| AUC | 0.876 | 0.824 | 0.858 | 0.488 |
| AUPR | 0.955 | 0.950 | 0.959 | 0.801 |
| T@1 | 0.219 | 0.402 | 0.329 | 0.017 |
| T@10 | 0.573 | 0.557 | 0.622 | 0.110 |

The off-diagonal cells are new and they undercut the "attack matched to
generator" framing.  MeLoMIA-ND scores its loss trajectory over the diffusion
timesteps of a *proxy* model that the attacker trains on the released synthetic
data; nothing in that pipeline requires the target to have been a diffusion
model.  Against the MVN generator — which has no diffusion process at all and no
neural network — it reaches 0.876, slightly *above* its own diagonal cell.
Against the CVAE it reaches 0.824 with the best T@1 in the row (0.402).

The reading is that the proxy's denoising loss is measuring how tightly the
released synthetic distribution concentrates around each candidate record, and
that quantity is informative for any generator that overfits, whatever
mechanism produced the overfitting.  The diffusion ladder is a good
multi-resolution probe, not a model-matched one.

**DP-PGM is the exception on this row too**: 0.488 AUC, T@10 of 0.110 against a
0.10 baseline.  It is the only cell in the BRCA grid where the strongest attack
in the suite is at chance, and it is at chance by a wide margin rather than
marginally.  Both statistical attacks and the strongest learned attack agree,
which is the cleanest positive result for DP in the whole grid.

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
