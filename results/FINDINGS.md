# Findings beyond the abstract

Results that came out of rebuilding the attacks and running them across the
full grid rather than only against their designated targets.  Each one passes
the negative controls in `scripts/sanity_check.py` — permuted labels fall to
chance, and one split's target does not predict another split's labels — which
is the reason they are reported rather than treated as bugs.

---

## 1. NoisyDiffusion leaks its training set's per-gene quantiles

**The strongest result here, and it has nothing to do with the diffusion model.**

MAMA-MIA was built for DP-PGM and is near chance against it (AUC ≈ 0.50).
Pointed at NoisyDiffusion with a fine enough discretisation, it reaches
**AUC 0.9996** on TCGA-BRCA.  Against MVN, CVAE and DP-PGM at the same settings
it stays at 0.48–0.51, so it is specific to this one generator.

| MAMA-MIA bins | MVN | CVAE | **ND** | DP-PGM |
|---|---|---|---|---|
| 4 | 0.520 | 0.528 | 0.539 | 0.499 |
| 64 | 0.515 | 0.524 | 0.757 | 0.493 |
| 256 | 0.508 | 0.509 | 0.870 | 0.486 |
| 1024 | 0.511 | 0.505 | **0.998** | 0.488 |
| 2048 | 0.508 | 0.507 | **0.9996** | 0.493 |

### Mechanism

NoisyDiffusion normalises with a scikit-learn `QuantileTransformer` fitted on
the training split, and inverts that transform to produce the values it
releases.  When `n_quantiles` is at least the training size — it is, by default
— the learned quantiles *are* the sorted training values.  Every synthetic value
is therefore an interpolation between two adjacent values that a training member
actually had, and the released dataset carries the training set's per-gene
empirical support almost exactly.

The evidence, from `scripts/analyse_quantile_leak.py` on BRCA split 1:

| | MVN | CVAE | **ND** | DP-PGM |
|---|---|---|---|---|
| genes where `min(synthetic) == min(member)` | 0/978 | 0/978 | **551/978** | 1/978 |
| genes where `min(synthetic) == min(non-member)` | 1/978 | 0/978 | 12/978 | 0/978 |
| real values within 1e-5 of a synthetic value, members | 0.006 | 0.008 | **0.040** | 0.002 |
| same, non-members | 0.006 | 0.010 | 0.008 | 0.003 |
| mean single-gene AUC | 0.502 | 0.496 | **0.649** | 0.499 |

For 551 of 978 genes the smallest value in the released data is *exactly* the
smallest value one of the training members had.  A single gene already separates
members from non-members at AUC 0.65; 978 near-independent weak signals pool to
near-certainty.

### Why it matters

* It is a property of the **output pipeline, not the model**.  Any generator
  that inverts an empirical quantile map fitted on its training data inherits
  it, no matter how private the model in between is.  A differentially private
  diffusion model with this post-processing would leak just as badly, which is a
  concrete way in which "DP-trained" is not the same as "DP".
* The mitigation is cheap: fit the quantile map on public or auxiliary data, fit
  it under DP, or use a smooth parametric transform (log/VST plus
  standardisation) whose inverse depends on the training set only through a
  couple of sufficient statistics.
* It is an attack a real adversary can run.  It needs the released synthetic
  data and the candidate records — nothing else.  No shadow models, no proxy
  generator, seconds of CPU.

### Caveat on naming

At 1024+ bins per gene over ~870 synthetic samples, MAMA-MIA's domain ratio has
stopped being a test on marginal *shape* and become a per-gene nearest-value
detector.  It is still the same estimator with the same aggregation, but calling
the result "MAMA-MIA" without qualification would overstate the continuity with
the marginal-based attack it was designed as.  The paper should probably present
it as its own baseline.

---

## 2. MahalaMIA is near-perfect once the covariance is conditioned properly

The submitted version inverts the 978×978 synthetic covariance — estimated from
~870 samples, hence rank-deficient — with a pseudo-inverse.  Replacing that with
a lightly ridge-regularised full inverse:

| MahalaMIA, BRCA | MVN | CVAE | ND | DP-PGM |
|---|---|---|---|---|
| pseudo-inverse (submitted) | 0.928 | 0.891 | 0.806 | 0.503 |
| Ledoit–Wolf shrinkage | 0.883 | 0.808 | 0.731 | 0.515 |
| ridge, α = 1e-3 | 0.999 | 0.922 | 0.802 | 0.538 |
| **ridge, α = 1e-6** | **1.000** | **0.996** | 0.826 | 0.503 |
| PCA to 50 components | 0.414 | 0.414 | 0.413 | 0.494 |
| PCA to 500 components | 0.651 | 0.661 | 0.627 | 0.523 |

Performance is flat for α below about 1e-4, so nothing delicate is being tuned;
what matters is that the near-null directions are kept rather than discarded.

The PCA rows say the same thing from the other side.  Projecting onto the
*leading* synthetic principal components drops the attack below chance, and it
recovers only as k approaches full rank.  The membership signal lives in the
low-variance directions: the high-variance ones carry biological structure that
members and non-members share, while a generator's over-fit to its training set
shows up where the synthetic covariance is nearly singular — exactly what a
pseudo-inverse throws away and Ledoit–Wolf shrinks toward zero.

This also re-orders the story about which generators are safe.  With a properly
conditioned covariance, CVAE is as exposed as MVN (0.996 vs 1.000), and a purely
statistical attack requiring no shadow models beats the tailored loss-trajectory
attack on both.

### One mechanism: the signal is in the low-variance directions, and `pinv` throws them away

The cohort-size sweep (`scripts/cohort_size_sweep.py`) holds everything fixed
but n.  One cohort, COMBINED; the training set is resampled to each size at
p = 978 genes, three trials each, so cohort composition, class count and tissue
heterogeneity cannot explain any trend.  Both generators, all three covariance
treatments:

| n/p | MVN pinv | MVN ridge 1e-6 | MVN ridge 1e-2 | CVAE pinv | CVAE ridge 1e-6 | CVAE ridge 1e-2 |
|---|---|---|---|---|---|---|
| 0.51 | 0.837 | **1.000** | 1.000 | 0.895 | **1.000** | 0.991 |
| 0.72 | 0.904 | **1.000** | 1.000 | 0.983 | **1.000** | 0.990 |
| 0.89 | 0.963 | **1.000** | 0.999 | 0.987 | **1.000** | 0.982 |
| 1.12 | 1.000 | 1.000 | 0.999 | 0.874 | 0.993 | 0.978 |
| 1.53 | 0.999 | 0.998 | 0.995 | 0.868 | 0.954 | 0.956 |
| 2.04 | 0.986 | 0.980 | 0.982 | 0.670 | 0.898 | 0.914 |
| 2.86 | 0.944 | 0.930 | 0.954 | 0.620 | 0.836 | 0.855 |
| 3.54 | 0.899 | 0.888 | 0.924 | 0.646 | 0.780 | 0.798 |

Figure: `results/figures/cohort_size_COMBINED_auc.pdf`.

**Both generators are perfectly attackable at n ≤ p.**  With a conditioned
covariance the attack reaches AUC 1.000 *and* TPR 1.000 at 10% FPR — every
member recovered without spending any false-positive budget — against MVN *and*
against the CVAE, everywhere below the crossover.  Exposure then decays smoothly
as n grows.  BRCA is not an unusual cohort; it sits at n/p = 0.89.

That is one mechanism, not two.  The membership signal lives in the
low-variance directions of the synthetic covariance — the directions where a
generator's fit to its particular training set is tightest — and a pseudo-inverse
discards them.  (The PCA rows above say the same thing from the other side:
projecting onto the *leading* components puts the attack below chance.)

**What differs between the generators is why the covariance is ill-conditioned.**
Measuring the synthetic covariance at n = 3458, well above p:

| source | n | effective rank | components for 99% var | condition number |
|---|---|---|---|---|
| real cohort | 4323 | 20.0 | 624 | 1.5e+02 |
| MVN synthetic | 3458 | 26.9 | 595 | 1.6e+02 |
| **CVAE synthetic** | 3458 | 11.5 | **268** | **6.2e+07** |
| ND synthetic | 3458 | 14.7 | 531 | 1.9e+02 |
| DP-PGM synthetic | 3458 | 97.4 | 881 | 2.8e+01 |

* **MVN is rank-limited by the data.**  Its covariance is a sample covariance,
  so it is singular exactly when n < p and well conditioned otherwise — 1.6e+02
  at n = 3458, essentially the real cohort's own conditioning.  The
  pseudo-inverse therefore fails only below the crossover, and the two curves
  converge above it.
* **The CVAE is rank-limited by its architecture.**  Its decoder maps a
  128-dimensional latent onto 978 genes, so the synthetic data lies near a
  low-dimensional manifold whatever n is: condition number 6.2e+07, five orders
  of magnitude worse than anything else, and 268 components carrying 99% of the
  variance against the real cohort's 624.  `np.linalg.pinv`'s default `rcond`
  truncates hundreds of directions, and those are the ones that carry membership.
  So the gap persists at every n and *widens* as n grows: +0.11 at n/p = 0.51,
  +0.23 at n/p = 2.04.

**The submitted estimator understates risk, and worst where risk is highest.**
Against MVN the pseudo-inverse is non-monotone — it peaks at n/p ≈ 1.1 and falls
away on both sides, reporting 0.837 at n/p = 0.51 where the true exposure is
1.000.  Against the CVAE it is low everywhere above the crossover (0.620 at
n/p = 2.86 against 0.855).  A blue team benchmarking against it and concluding a
small cohort was acceptably safe would have been reading the attack's linear
algebra rather than the generator's privacy.  For a privacy evaluation that is
the worst direction for an error to run, and it is the strongest argument in this
work for reporting attacks with a conditioned covariance.

All 224 COMBINED MahalaMIA runs, these 144 included, pass the shuffled-label and
cross-split negative controls.

---

## 3. Both grids: the cheapest attack wins, and only one attack transfers — on one cohort

Four attacks x four generators x five splits, on both cohorts, as-submitted
configurations.  Bold is the best attack in each column.

**BRCA** (1,089 samples, 871 train, 5 subtypes)

| AUC | MVN | CVAE | ND | DP-PGM |
|---|---|---|---|---|
| MahalaMIA | **0.928** | **0.891** | 0.806 | 0.497 |
| MAMA-MIA | 0.520 | 0.528 | 0.539 | 0.501 |
| MeLoMIA-CVAE | 0.553 | 0.798 | 0.563 | 0.491 |
| MeLoMIA-ND | 0.876 | 0.824 | **0.858** | 0.488 |

**COMBINED** (4,323 samples, 3,458 train, 12 classes, plus an 824-sample auxiliary set)

| AUC | MVN | CVAE | ND | DP-PGM |
|---|---|---|---|---|
| MahalaMIA | **0.900** | 0.619 | **0.770** | 0.500 |
| MAMA-MIA | 0.512 | 0.517 | 0.528 | 0.499 |
| MeLoMIA-CVAE | 0.528 | **0.731** | 0.525 | 0.494 |
| MeLoMIA-ND | 0.644 | 0.591 | 0.648 | 0.499 |

### 3a. MahalaMIA is the best attack on most columns, and the gap widens at the low-FPR end

On BRCA it takes MVN and CVAE; on COMBINED it takes MVN and ND.  At TPR@1%FPR —
the metric a privacy claim actually rests on — it takes **all three non-DP
columns on COMBINED**:

| TPR@1%FPR, COMBINED | MVN | CVAE | ND |
|---|---|---|---|
| MahalaMIA | **0.411** | **0.079** | **0.196** |
| MeLoMIA-CVAE | 0.015 | 0.052 | 0.011 |
| MeLoMIA-ND | 0.075 | 0.031 | 0.058 |

A closed-form Mahalanobis distance to the synthetic covariance — no shadow
models, no neural network, seconds of compute — beats both learned loss attacks
5x at the 1% budget on MVN and 3x on ND.  And this is before the covariance
conditioning of finding 2, which takes the BRCA MVN and CVAE cells to 1.000 and
0.970.

For a paper arguing that synthetic bulk RNA-seq leaks, this is the more alarming
result: the strongest practical attack needs only the released file, no shadow
budget, and no access to the generator's family.

### 3b. Transfer is a property of the cohort, not just of the attack

On BRCA, MeLoMIA-ND appeared to be generator-agnostic: a +0.008 diagonal
advantage, scoring 0.876 against MVN — a per-class Gaussian with no diffusion
process anywhere — slightly *above* its own diagonal cell.  MeLoMIA-CVAE was the
opposite, +0.240 and at chance off-diagonal.

That distinction does not survive the larger cohort.  Holding the attack fixed
at MeLoMIA-ND:

| MeLoMIA-ND vs | BRCA | COMBINED | drop |
|---|---|---|---|
| MVN | 0.876 | 0.644 | −0.232 |
| CVAE | 0.824 | 0.591 | −0.233 |
| ND | 0.858 | 0.648 | −0.210 |

The drop is **uniform to within 0.023 across three structurally unrelated
generators**.  MeLoMIA-ND does not become selective on COMBINED; it becomes
uniformly weaker.  So the honest statement is conditional: the denoising-loss
probe transfers across generator families *when the target has memorised enough
for a density probe to find*, and a 4x larger training set removes most of that
for every family at once.

MeLoMIA-CVAE, by contrast, keeps its shape exactly — 0.553/0.798/0.563 on BRCA
becomes 0.528/0.731/0.525 on COMBINED.  Only its diagonal moves.  Its ELBO probe
needs the target's own encoder-decoder geometry, which no amount of extra data
changes.

An earlier draft of this section generalised the BRCA behaviour into a claim
that the attack "is not diffusion-specific."  On one cohort that reads as a
property of the probe; on two it is visibly a property of how much the
generators memorised.

### 3c. AUC understates how much a larger cohort protects

| MeLoMIA-CVAE vs CVAE | AUC | T@1 | T@10 |
|---|---|---|---|
| BRCA | 0.798 | 0.329 | 0.518 |
| COMBINED | 0.731 | 0.052 | 0.319 |

AUC falls 0.067 — a modest-sounding number.  TPR at 1% FPR falls **6.3x**, from
flagging one member in three to one in nineteen.  The same pattern holds for
MeLoMIA-ND's diagonal (0.329 to 0.058, 5.7x) and MahalaMIA's CVAE column (0.339
to 0.079, 4.3x).

Reporting AUC alone would describe cohort size as a mild mitigation.  The
low-FPR metrics say it is close to an order of magnitude.  This is a concrete
reason for the paper to lead with TPR at fixed low FPR.

Note the exception: MahalaMIA against MVN barely moves (0.928 to 0.900 AUC,
0.631 to 0.411 at 1% FPR).  A per-class Gaussian fit is a summary statistic
whose fidelity to its own training split does not decay with n, so the MVN
generator's vulnerability is structural rather than a small-sample artefact —
the stronger claim for the paper.

### 3d. The shadow budget, not the generator, set the abstract's ND number

| MeLoMIA-ND vs ND (BRCA) | K=5 (abstract) | K=15 | K=30 |
|---|---|---|---|
| AUC | 0.620 | 0.848 | 0.858 |

K=5 was inherited from the blue team's five published splits rather than chosen.
The meta-classifier sees 15 timesteps x 600 noise draws per record, so a
five-model pool is a very small sample of the *between-model* variation it must
generalise over.  Raising K to 30 — nothing else changed — multiplies TPR at a
1% false-positive budget by 7.3x and at 10% by 3.2x.

The curve saturates early: almost all the gain is between K=5 and K=15, and
K=30 adds 0.010.  So the grid's K=30 sits past the knee, and the sweep in TODO
item 2 should spend its budget below K=15.

The same change is worth far less to the CVAE backend (0.753 to 0.798).  K=5 was
starving one attack specifically, and the abstract's headline ordering — CVAE
attack ahead of ND attack — **reverses** on BRCA once both have an adequate
budget.

### 3e. DP-PGM is the one generator that holds, on both cohorts, against everything

Its column spans 0.488-0.501 AUC on BRCA and 0.494-0.500 on COMBINED.  Best
TPR@10%FPR across all four attacks and both cohorts: 0.110, against a 0.100
baseline.

This is the cleanest positive result in the project: not "the attacks we tried
did poorly" but *four attacks spanning a closed-form distance, a marginal-ratio
test and two learned loss-trajectory classifiers, at two cohort sizes, all land
on chance.*

It should be read alongside finding 1, which is the reminder that this protects
the **model**.  NoisyDiffusion's quantile leak lives in a post-processing step
bolted to the generator's output, and no amount of DP in the training loop would
have stopped it.

---

## 4. Cross-validation score and attack strength point in opposite directions

TODO item 8 asked whether synth-shadow modelling pays for itself.  It does, and
the ablation also produced the most useful methodological result in the repo.

Both arms are identical except for where the feature-extraction shadows were
trained: `_synth` on internal synthetic data emitted by base shadows (the
method), `_real` directly on real splits (the ablation).  Same K=15, same
splits, same features, same meta-classifier, same proxy at inference.

| AUC | vs CVAE | vs ND |
|---|---|---|
| MeLoMIA-ND, real shadows | 0.704 | 0.686 |
| MeLoMIA-ND, synth-shadows | **0.810** | **0.848** |
| MeLoMIA-CVAE, real shadows | 0.685 | 0.532 |
| MeLoMIA-CVAE, synth-shadows | **0.793** | **0.561** |

| TPR@1%FPR | vs CVAE | vs ND |
|---|---|---|
| MeLoMIA-ND, real shadows | 0.137 | 0.104 |
| MeLoMIA-ND, synth-shadows | **0.368** | **0.295** |
| MeLoMIA-CVAE, real shadows | 0.209 | 0.026 |
| MeLoMIA-CVAE, synth-shadows | **0.329** | **0.045** |

Synth-shadow modelling wins every cell.  On the ND diagonal it is worth +0.161
AUC and 2.9x TPR@1%FPR; on the CVAE diagonal +0.108 and 1.6x.  The one cell
where it barely matters (+0.029) is MeLoMIA-CVAE against ND, where the attack is
near chance under both conditions and there is nothing for the extra layer to
lift.

### The part worth the paper's attention

Look at what cross-validation said about these same runs.

| | CV AUC (sample-grouped) | deployed, diagonal |
|---|---|---|
| real-data shadows | 1.000 / 0.969 | 0.685 / 0.686 |
| synth-shadows | 0.789 / 0.791 | 0.793 / 0.848 |

**The two orderings are opposite.**  Real-data shadows produce a near-perfect
cross-validation score and the weaker attack; synth-shadows produce a visibly
worse cross-validation score and the stronger attack.  An experimenter choosing
between the two designs on CV score — the obvious thing to do — would pick the
wrong one and give up 0.11 to 0.16 AUC.

The 1.000 is not a bug, it is the honest answer to the wrong question.  A shadow
trained on a real split has memorised its own training rows, so a classifier
reading that shadow's losses separates members from non-members perfectly.  The
`_real` arm's search noticed: it selected the *smallest* feature budget on offer
(2 temperatures, 12 noise draws) because nothing more was needed.  The ensemble
weights then came out exactly uniform — with every classifier at 1.000 there was
nothing left to discriminate on, so the blend carried no information while
looking like a considered choice.

The synth arm's CV number, by contrast, is roughly *honest*: 0.789 against 0.793
deployed on the CVAE diagonal.  Synth-shadow modelling does not only attack
better, it makes its own validation trustworthy, because the models the
meta-classifier learns from are in the same domain as the one it is applied to.

This is the same disease as finding 3's selection problem, on a second axis.
Sample-grouped CV fails to hold out *models*; real-data shadows fail to match
the inference *domain*.  Either one alone makes the CV number an unreliable
guide to deployment, and the repo had both at once.  `MeLoMIA(
internal_proxy_selection=True)` fixes the first; synth-shadow modelling, now
measured, fixes the second.

**Practical rule for the paper: never report a MeLoMIA cross-validation AUC as
an attack result.**  Report the grid cell, which is scored against real target
labels through the real deployment path.

### A shadow-count curve that saturates early

The `_synth` arm runs K=15 where the main grid runs K=30, which gives a third
point on the curve from finding 3c:

| MeLoMIA-ND vs ND | K=5 (abstract) | K=15 | K=30 |
|---|---|---|---|
| AUC | 0.620 | 0.848 | 0.858 |

Almost all of the gain is between K=5 and K=15; K=30 adds 0.010.  So the grid's
K=30 sits comfortably past the knee rather than being an arbitrary choice, and
the full sweep in TODO item 2 should spend its budget below K=15 rather than
above K=30.

---

## 5. Where we disagree with the abstract

The abstract reports MahalaMIA applied to NoisyDiffusion on BRCA at AUC 0.489 —
chance — and concludes that "diffusion-based generation preserves covariance
less consistently".  We get **0.806** with the submitted pseudo-inverse and
**0.826** with the ridge, on the blue team's own published ND synthetic data and
the blue team's own splits.

Since the target data is identical, the difference is upstream in how the
abstract's cross-model run was configured.  This should be resolved before the
paper repeats the claim, because our numbers point the opposite way: ND
preserves quite a lot of covariance structure, and (see finding 1) rather more
than that.

---

## Reproducing

```bash
python scripts/run_experiment.py configs/experiments/tune_statistical.yaml
python scripts/analyse_quantile_leak.py --dataset BRCA
python scripts/sanity_check.py --dataset BRCA
```
