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

(Both tables above are BRCA, where n < p.  Section 6 scans the same axes far more
finely and on both cohorts, and finds that the ranking of *conditioners* flips
above the crossover — Ledoit–Wolf is the worst choice here and the best on
COMBINED.  The PCA conclusion survives the finer scan; the Ledoit–Wolf one does
not generalise.)

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

## 5. Model-disjoint selection: the CV number was wrong, the hyperparameters were not

Finding 4 showed cross-validation ranking two *designs* backwards.  This is the
narrower question it raised: within the chosen design, was the meta-classifier
also selected on the wrong axis?

The submission selected features, hyperparameters and ensemble weights with
`StratifiedGroupKFold` grouped on `sample_id`, which holds *records* out while
leaving every shadow model on both sides of every fold.  At inference the model
is the new thing and every record has been seen K times under K different
membership labels — the opposite arrangement.  Rotating a hold-out over the
shadow pool instead is the internal-proxy role of `docs/FIVE_ROLES.md`, and
`MeLoMIA(internal_proxy_selection=True)` implements it.  Both arms share the
same K=30 stack and the same features, so only selection differs.

### The CV numbers do fall, in proportion to how much there is to overfit

| BRCA, mean over 5 classifiers | sample-grouped | model-disjoint | gap |
|---|---|---|---|
| MeLoMIA-ND | 0.7969 | 0.7738 | **+0.0231** |
| MeLoMIA-CVAE | 0.7924 | 0.7826 | **+0.0098** |

The gap is the model-specific component of the signal: what a classifier can
learn about *these thirty shadows* that does not transfer to a new model.  It is
2.4x larger for the ND backend, which is what you would expect from capacity —
its feature grid is 15 timesteps x 600 draws against the CVAE's 6 x 50.

The searches agree, unprompted.  Under model-disjoint folds both backends chose
*smaller* feature budgets — ND from 15 timesteps and 600 draws to 9 and 350,
CVAE from 5 temperatures and 44 draws to 4 and 32.  Once validation stops
rewarding quirks of models already seen, the search stops wanting the surplus
features.

### But the attack is unchanged

| BRCA AUC | grouped | block-CV | Δ |
|---|---|---|---|
| MeLoMIA-ND vs MVN | 0.8761 | 0.8678 | −0.008 |
| MeLoMIA-ND vs CVAE | 0.8239 | 0.8151 | −0.009 |
| MeLoMIA-ND vs ND | 0.8584 | 0.8538 | −0.005 |
| MeLoMIA-CVAE vs CVAE | 0.7980 | 0.7981 | +0.000 |
| MeLoMIA-CVAE vs ND | 0.5628 | 0.5604 | −0.002 |

Every difference is well inside the split-to-split spread (±0.02 to ±0.07).  So
the two claims separate, and only one survives:

* **Reporting** — the grouped CV AUC is inflated by up to 0.036 relative to an
  honest model-disjoint estimate, so it must not be quoted as an attack result.
  Confirmed.
* **Selection** — it was nonetheless not choosing materially worse
  hyperparameters.  Refuted.

This is a useful negative result.  The contamination was real and the fix is
correct, but the grid never depended on it.  Worth one paragraph in the paper as
methodology, not a headline.

### A correction to how the CV number should be described

It is tempting — and this write-up did it earlier — to call the grouped number
"optimistic" full stop.  That is wrong in an instructive way.  Set the CV
numbers beside the deployed diagonal cells:

| BRCA | grouped CV | model-disjoint CV | deployed diagonal |
|---|---|---|---|
| MeLoMIA-ND | 0.797 | 0.774 | **0.858** |
| MeLoMIA-CVAE | 0.792 | 0.783 | **0.798** |

Both CV numbers sit *below* the ND attack's actual grid cell.  The grouped
number is optimistic **relative to model-disjoint CV**, which is what the
inflation measures — but it is not an upper bound on deployment, and for the ND
backend it understates the real result by 0.06.

The reason is that the two quantities are different tasks.  CV scores shadow
membership on internal synthetic data emitted by base shadows; the grid scores
target membership on the released dataset, whose generator was fitted to more
data with different hyperparameters.  Nothing forces them to agree in either
direction.  Finding 4's real-data-shadow arm is the case where CV runs far
*ahead* of deployment (1.000 against 0.685); this is a case where it runs
behind.

**The rule that survives all of it:** a MeLoMIA cross-validation AUC is a
diagnostic of the meta-classifier, not an estimate of the attack.  Report the
grid cell, which is scored against real target labels through the deployment
path.

---

## 6. The geometry sweep: PCA is a worse ridge, and the right fix depends on n/p

Section 2 established that conditioning the covariance matters and that
aggressive PCA is harmful.  It scanned only k = 50 and k = 500, which skipped
the region where PCA could plausibly have helped, and it tested one cohort's
regime.  This is the full scan: 43 variants on BRCA (860 runs) and 47 on
COMBINED (1128 runs), five splits each, over five axes — covariance conditioning,
PCA dimension, *leading*-component removal, per-gene standardisation, and
class-conditional mean/covariance.  Configs in
`configs/experiments/mahala_geometry_{brca,combined}.yaml`.

The two cohorts sit on opposite sides of the crossover, and that turns out to
decide everything:

| cohort | train n | genes p | n/p | synthetic covariance |
|---|---|---|---|---|
| TCGA-BRCA | 871 | 978 | 0.89 | rank-deficient |
| TCGA-COMBINED | 3458 | 978 | 3.54 | full rank |

### 6a. PCA never wins, and on COMBINED it only ever loses

Each cell below is **one number averaged over three generators** — the mean AUC
across MVN, CVAE and ND, five splits each.  DP-PGM is at chance in every row and
is excluded from the average.  Section 6f breaks the same quantity out per
generator, so the `pinv` row here is exactly the mean of the `pinv` column
there (BRCA 0.928/0.880/0.806 -> 0.872; COMBINED 0.900/0.619/0.770 -> 0.763).

| variant | BRCA, mean AUC over 3 gens | COMBINED, mean AUC over 3 gens |
|---|---|---|
| ridge 1e-8 | **0.941** | 0.810 |
| Ledoit–Wolf | 0.807 | **0.868** |
| PCA 900 | 0.912 | 0.746 |
| PCA 850 | 0.929 | 0.696 |
| PCA 800 | 0.917 | 0.678 |
| PCA 700 | 0.825 | 0.731 |
| PCA 600 | 0.732 | 0.728 |
| PCA 400 | 0.565 | 0.683 |
| PCA 200 | 0.445 | 0.613 |
| pinv (submitted) | 0.872 | 0.763 |

The three-generator mean is a ranking convenience, not a quantity anyone should
quote on its own: it weights three unrelated generators equally and hides
variants that are excellent against one and harmful against another.  Section 6d
is exactly that case — class-conditional scoring looks mid-table here while
being the single best setting against MVN on both cohorts.  Rank with it, report
per generator.

On **BRCA** there is a real band — k ≈ 850 of 978, a 13% reduction — where PCA
beats the submitted pseudo-inverse by a wide margin (0.929 vs 0.872).  That band
is narrow and it is not where the earlier sweep looked.  But **a tiny ridge at
full rank beats every PCA setting in it** (0.941), and the same holds at the
low-FPR end: 0.721 vs 0.668 mean TPR at 1% FPR.

On **COMBINED**, PCA is harmful at every k tested.  Every projection is worse
than leaving the 978 dimensions alone.

Both facts follow from the same thing.  Mahalanobis distance is invariant to
any invertible linear map, so a full-rank PCA changes nothing at all; PCA only
has an effect *because the regulariser is applied after the projection and is
not equivariant*.  Truncating to k < p is therefore not a distinct idea — it is
an implicit, very aggressive regulariser, one that sets the discarded directions
to zero weight instead of down-weighting them.  Ridge does the same job while
keeping the low-variance directions that Section 2 showed carry the membership
signal.  PCA can only help where there is rank deficiency to fix (BRCA), and
even there it is strictly the cruder instrument.

**So the user-visible recommendation is: do not put PCA in front of MahalaMIA.**
Condition the covariance instead.

### 6b. Which conditioner, though, flips with the regime

Mean AUC over the same three generators:

| | BRCA (n/p 0.89) | COMBINED (n/p 3.54) |
|---|---|---|
| ridge 1e-8 | **0.941** | 0.810 |
| ridge 1e-2 | 0.852 | 0.841 |
| Ledoit–Wolf | 0.807 | **0.868** |

Below the crossover the covariance is singular and the near-null directions are
signal, so the correct move is the *smallest* regulariser that makes the inverse
exist — hence ridge 1e-8, and hence Ledoit–Wolf's shrinkage being actively
counterproductive (0.807, below even `pinv`).  Above the crossover the covariance
is invertible and the problem is no longer rank but *estimation noise* in 978×978
= 956,484 entries from 3458 samples.  There, shrinkage toward a structured
target is the right estimator and it wins clearly (0.868 vs 0.763 for `pinv`).

Section 2's Ledoit–Wolf row was measured on BRCA only, and its poor showing
there should not be read as a verdict on shrinkage in general.

### 6c. Removing the *leading* components separates ND from the CVAE

`drop_leading` keeps full rank but deletes the top components.  On BRCA:

| dropped | MVN | CVAE | ND |
|---|---|---|---|
| 0 (ridge 1e-8) | 1.000 | 0.996 | 0.826 |
| 5 | 0.990 | 0.995 | 0.705 |
| 20 | 0.988 | 0.995 | 0.684 |
| 50 | 0.986 | 0.995 | 0.666 |
| 200 | 0.979 | **0.996** | **0.631** |

Deleting the 200 highest-variance directions costs the CVAE *nothing* — 0.996
before and after — and costs NoisyDiffusion a third of its AUC.  The CVAE's leak
is entirely in the low-variance directions, consistent with its 128-dimensional
decoder bottleneck (Section 2).  ND's leak is substantially in the dominant
directions, consistent with the quantile-support mechanism of Section 1, which
reproduces the training set's per-gene marginals and therefore its principal
axes.  **The two generators leak in geometrically orthogonal places**, which is
why no single projection is right for both.

### 6d. Class-conditional scoring closes MVN completely

Fitting a per-class mean and covariance, and scoring each sample against its own
class:

| | MVN AUC | MVN TPR@1%FPR | CVAE AUC | ND AUC |
|---|---|---|---|---|
| COMBINED, pinv (submitted) | 0.900 | 0.411 | 0.619 | 0.770 |
| COMBINED, ridge 1e-4 | 0.893 | 0.449 | 0.798 | 0.769 |
| COMBINED, ridge 1e-4 + class-conditional | **1.000** | **1.000** | 0.776 | 0.666 |
| BRCA, ridge 1e-6 + class-conditional | **1.000** | **1.000** | 0.772 | 0.660 |

Every COMBINED member is recovered with no false-positive budget spent at all.
The reason is structural rather than statistical: our MVN generator *is* a set of
per-class Gaussians, so a per-class Mahalanobis distance is exactly its negative
log-likelihood.  Matching the attack's geometry to the generator's is worth more
than any amount of regulariser tuning.

It is also strictly generator-specific — it costs the CVAE 0.32 → 0.13 and ND
0.27 → 0.03 in TPR@1%FPR on COMBINED, since neither is class-structured in that
way.  This is an argument for reporting MahalaMIA as a small family parameterised
by an assumed generator structure, not as one fixed statistic.

Per-class PCA *bases* (`pca_per_class`) are a different matter and fail badly —
0.35–0.41 AUC, consistently below chance on both cohorts.  With as few as ~70
samples in a class, a per-class basis is fit to noise, and a consistently
below-chance score means the resulting distance is anti-correlated with
membership.

### 6e. The auxiliary reference set does not help the projection

COMBINED is the only cohort with an unlabelled auxiliary set (824 samples), so
it is the only place the PCA basis can be estimated from something other than
the synthetic data:

| basis | mean AUC over 3 gens |
|---|---|
| synthetic (default) | 0.810 |
| reference | 0.811 |
| pooled | 0.702 |

No gain.  Pooling actively hurts.  Whatever the auxiliary set is useful for, it
is not defining a better subspace.

### 6f. Recommended settings, and what they cost the submitted numbers

Per generator now, not averaged — these are the cells whose means appear in 6a:

| cohort / generator | submitted (`pinv`) AUC | best geometry AUC | setting |
|---|---|---|---|
| BRCA / MVN | 0.928 | **1.000** | ridge 1e-6 (+cc also 1.000) |
| BRCA / CVAE | 0.880 | **0.996** | ridge 1e-8 |
| BRCA / ND | 0.806 | **0.826** | ridge 1e-6 |
| COMBINED / MVN | 0.900 | **1.000** | ridge 1e-4 + class-conditional |
| COMBINED / CVAE | 0.619 | **0.838** | Ledoit–Wolf |
| COMBINED / ND | 0.770 | **0.820** | Ledoit–Wolf |

The largest single gain is COMBINED/CVAE, +0.22 AUC, and the largest at the
low-FPR end is COMBINED/MVN, TPR@1%FPR 0.411 → 1.000.  None of it comes from
PCA.

DP-PGM stays at 0.496–0.506 AUC in all 1988 runs across both cohorts — but see
`docs/PGM_ATTACK_SURFACE.md` before reading that as evidence of privacy.

---

## 7. DP-PGM's flat row was three defects, not a privacy result

Every DP-PGM cell in every table above sits at 0.488–0.506.  The abstract read
that as the generator being private.  Two defects, one ours and one upstream,
produce the same table for reasons that have nothing to do with privacy, and
both are now fixed.

### 7a. The adapter permuted the gene columns

`mia/generators/pgm.py` dropped the upstream generator's
`selected_gene_names` and returned its matrix as though column *j* were gene
*j*.  The upstream selects genes by variance and returns them in *that* order,
so every gene's distribution was silently transposed onto a different gene.

A column permutation leaves the multiset of columns unchanged, so every global
statistic survives it — synthetic mean 11.298 against a real 11.296.  Nothing
that checked totals could have caught it.  Per gene, on BRCA split 1:

| | before | after |
|---|---|---|
| mean per-gene mean error | 1.931 | 0.374 |
| per-gene Wasserstein (training SDs) | 2.898 | 0.727 |
| TSTR macro-F1 (real ceiling 0.811) | 0.092 | 0.185 |
| real-vs-synthetic discriminator AUC | 1.000 | 1.000 |

Fixed in `c29d6b5`, together with `mia/fidelity.py` and
`scripts/eval_fidelity.py` — the harness that caught it, and the first thing in
this repo to measure generator quality at all.  **Every DP-PGM number recorded
before 2026-09-20 is affected.**

### 7b. The budget was spent under basic composition

`src/pgm_fitter.py` in `steveng9/PrivateRNAseqGen` divided ε linearly across
marginals and calibrated each with the classical Gaussian bound, so σ grew
*linearly* in the number of measurements.  Gaussian mechanisms compose
additively in ρ = 1/(2σ²), which makes it **√k**.  At the submitted
configuration — 978 genes, joint mode, 1956 marginals, ε=10, δ=1e-5:

| | 1-way σ | 2-way σ |
|---|---|---|
| basic (as shipped) | 1435.8 | 707.2 |
| zCDP | 28.8 | 20.2 |
| reduction | **49.8×** | **35.0×** |

σ = 1436 on a cohort of **871 rows**: the injected noise exceeded the size of
the entire study, and was roughly seven times the ~218 counts in a one-way cell.

The penalty is √k, so it is a function of how many features you measure:

| marginals k | basic σ | zCDP σ | penalty |
|---|---|---|---|
| 5 | 2.4 | 1.2 | 2.0× |
| 20 | 9.7 | 2.4 | 4.1× |
| 50 | 24.2 | 3.7 | 6.5× |
| 100 | 48.5 | 5.3 | 9.1× |
| 978 | 473.8 | 16.6 | 28.6× |
| 1956 | 947.6 | 23.4 | 40.5× |

**This is the part that generalises beyond our repo.**  Below roughly 50
measurements basic composition costs a small constant, which is the regime
tabular DP synthesisers are usually demonstrated in.  Genomic feature counts —
hundreds to thousands of marginals — are what turn a tolerable constant into a
30–60× loss.  Any tabular DP-SDG method carried over to omics inherits this, and
the looseness is invisible because the output is still formally (ε, δ)-DP.

Fixed in `9e1fe87` as `composition="zcdp"`, now the default, with
`composition="basic"` retained byte-for-byte so earlier results reproduce.  The
accounting is McKenna's, transcribed rather than derived: MST writes the same
rule as `weights / np.linalg.norm(weights)` then `sigma / weight`, which for k
equal weights is exactly σ√k.  ρ comes from OpenDP's numerically-inverted
conversion — snsynth's `cdp_rho`, what MST and AIM call — with the closed form
ε = ρ + 2√(ρ ln(1/δ)) as a fallback that returns a *smaller* ρ and therefore more
noise, so it can only be conservative.  Sixteen tests in the generator repo's
`tests/test_composition.py` check σ against MST's formulation at k = 1, 10, 978,
1956 and assert the ρ accumulated across the measurement loop equals the budget
to 1e-9; the fitter raises if it ever exceeds it.

### 7c. Under basic composition, fidelity and coverage cannot both be had

Nine configurations, BRCA split 1, ε=10, all on the original basic accounting
(`scripts/pgm_fidelity_sweep.py`, `results/pgm_sweep.csv`):

| config | TSTR macro-F1 | vs ceiling | W1 (SDs) | corr-MAE | disc. AUC |
|---|---|---|---|---|---|
| n₁=978, k=4, joint *(submitted)* | 0.185 | 0.23 | **0.727** | 0.209 | 1.000 |
| n₁=400, k=4, joint | 0.165 | 0.20 | 1.680 | 0.251 | 1.000 |
| n₁=200, k=4, joint | 0.283 | 0.35 | 2.028 | 0.195 | 1.000 |
| n₁=100, k=4, joint | 0.470 | 0.58 | 2.307 | 0.169 | 1.000 |
| **n₁=50, k=4, joint** | **0.658** | **0.81** | 2.264 | 0.168 | 1.000 |
| n₁=200, k=8, joint | 0.182 | 0.22 | 1.964 | 0.197 | 1.000 |
| n₁=200, k=2, joint | 0.342 | 0.42 | 2.188 | 0.178 | 1.000 |
| n₁=200, n₂=50, k=4, strat | 0.234 | 0.29 | 2.100 | 0.189 | 1.000 |
| n₁=100, n₂=50, k=4, strat | 0.349 | 0.43 | 2.302 | 0.172 | 1.000 |

Utility rises monotonically as fewer genes are measured — at 50 genes the
synthetic data recovers **81% of the real-data classification ceiling** against
23% at the submitted setting — because σ falls linearly with the marginal count.

But the two metrics move in opposite directions.  W1 *worsens* from 0.73 to 2.26
SDs over the same sweep, because at n₁=50 only 50 of 978 genes are modelled at
all and the other 928 are filled with a constant.  The n₁=50 release is a good
classifier substrate and a bad RNA-seq dataset.  Discriminator AUC is pinned at
1.000 throughout; 928 constant columns are not subtle.  More bins costs utility
at fixed σ (k=8 → 0.182, k=2 → 0.342), since the same budget is spread over more
cells.

**No setting of this knob gives both, and the knob is not the problem.**  The
trade-off exists only because σ scales linearly in the number of marginals.
That is what 7b removes.

### 7c-ii. Re-accounted, the whole frontier lifts (single split; see 7c-iii)

Same cohort, same split, same seed, same ε and δ, same 978 genes — only the
composition theorem differs (`scripts/pgm_composition_compare.py`,
`results/pgm_composition.csv`):

| | basic | **zCDP** |
|---|---|---|
| 1-way σ | 1435.8 | **28.8** |
| 2-way σ | 707.2 | **20.2** |
| TSTR macro-F1 | 0.185 | **0.553** |
| vs real ceiling (0.811) | 0.23 | **0.68** |
| per-gene W1 (training SDs) | 0.727 | **0.511** |
| corr-MAE | 0.209 | **0.142** |
| discriminator AUC | 1.000 | 1.000 |

The `basic` arm reproduces 7c's first row exactly, so the two are directly
comparable.

**Correction to the numbers above.**  That table is a single split (split 1),
which is how it was first reported; split 1 happens to be a low one for `basic`.
The five-split means at this configuration are utility ratio 0.286±0.067
(`basic`) against 0.743±0.086 (`zCDP`).  The conclusion is unchanged — roughly
2.6× the utility at better per-gene fidelity — but the single-split figures
should not be quoted, and 7c-iii supersedes them.

**A claim made here was too strong.**  The first version of this finding said
the fidelity/coverage trade-off "disappears."  It does not; 7c-iii shows it
persists.  What changes is that the whole frontier lifts far enough that full
gene coverage becomes affordable.

### 7c-iii. The full grid: zCDP dominates, and `n_bins` becomes a live axis

11 configurations × {basic, zCDP} × 5 splits, BRCA, ε=10, δ=1e-5
(`scripts/pgm_grid.py`, `results/pgm_sweep.csv`; 101 fits, 3547 s wall on 16
workers).  Mean ± sd over splits.

Utility, as a fraction of the real-data ceiling (higher is better):

| configuration | basic | **zCDP** |
|---|---|---|
| n₁=50, k=4, joint | 0.674±0.097 | **0.961±0.030** |
| n₁=100, k=4, joint | 0.562±0.039 | **0.927±0.014** |
| n₁=200, k=4, joint | 0.434±0.060 | **0.840±0.065** |
| n₁=400, k=4, joint | 0.293±0.101 | **0.818±0.053** |
| n₁=100/50, k=4, strat | 0.398±0.057 | **0.800±0.076** |
| n₁=978, k=4, joint | 0.286±0.067 | **0.743±0.086** |
| n₁=978, k=8, joint | 0.256±0.047 | **0.712±0.050** |
| n₁=978, k=2, joint | 0.289±0.068 | **0.693±0.078** |

Per-gene Wasserstein distance, in training SDs (lower is better):

| configuration | basic | **zCDP** |
|---|---|---|
| n₁=978, k=8, joint | 0.498±0.005 | **0.257±0.003** |
| n₁=978, k=4, joint | 0.725±0.004 | **0.505±0.005** |
| n₁=978, k=2, joint | 1.230±0.011 | **1.116±0.011** |
| n₁=400, k=4, joint | 1.682±0.008 | **1.576±0.009** |
| n₁=50, k=4, joint | 2.213±0.032 | 2.221±0.033 |

Three things this establishes.

**(a) zCDP wins every cell on every metric**, and one configuration wins
outright.  `n₁=978, k=8, joint, zCDP` **Pareto-dominates every `basic`
configuration**: its utility (0.712) exceeds the best utility `basic` reaches
anywhere (0.674, at n₁=50), and its W1 (0.257) beats the best W1 `basic` reaches
anywhere (0.498, at n₁=978 k=8).  No amount of tuning under the old accounting
reaches that point.  It is the recommended setting.

**(b) The trade-off persists, but the frontier lifts.**  Utility still peaks at
n₁=50 (0.961) precisely where fidelity is worst (W1 2.221) — that configuration
models 5% of the genome and fills the other 95% with a constant, which no
accounting can repair.  What zCDP buys is that full coverage is no longer
unaffordable: 71% of the ceiling *and* the best per-gene fidelity in the study,
at once.

**(c) `n_bins` is newly worth tuning.**  Under `basic`, k = 2/4/8 were
indistinguishable at full coverage (utility 0.256–0.289) because noise dominated
the bin structure entirely.  Under zCDP, k=8 halves W1 relative to k=4 (0.257
against 0.505) for about four points of utility.  That axis only became visible
once σ fell from 1436 to 29.  `corr_mae` is flatter — most 978/200 configs land
at 0.141–0.152 under zCDP against 0.19–0.21 under basic — with the largest
single swing at n₁=400, from 0.251 (the worst in the grid) to 0.126 (the best).

**It is still not a good generator.**  Discriminator AUC is 1.000 nearly
everywhere; the only cells that dip are the three highest-fidelity zCDP configs
(n₁=978 k=8, n₁=400 k=4, n₁=978 k=4), all at 0.999±0.001.  Real and synthetic
remain essentially perfectly separable.  What changed is that DP-PGM now has
enough signal in it to be worth attacking, which is the precondition for the
DP-PGM column meaning anything.

### 7d. What the re-accounting does not fix

ε is still not an end-to-end guarantee.  Two steps read the private training
data and spend no budget:

1. **Marginal selection** ranks genes by the variance of the private data and
   pairs by its Spearman correlations, so which genes appear in the released
   model is a deterministic function of the training split.  MST spends a full
   **ρ/3** on exactly this step, via the exponential mechanism.
2. **Discretisation** takes bin edges from `np.percentile` of the private data,
   and `inverse_transform` releases values interpolated between them — the same
   structure as finding 1, where NoisyDiffusion's `QuantileTransformer` leaked
   its training support at AUC 0.9996.

Both are fixable and neither is fixed.  Until they are, the reported ε covers
the noisy marginals only, and the paper has to say so.  Full treatment in
`docs/PGM_ATTACK_SURFACE.md` §5.

### 7e. What this means for the DP-PGM column

**"All four attacks sit at chance against DP-PGM at ε=10" was not a safe
claim.**  Attacking column-scrambled data whose marginals carry seven times
their own signal in noise, and attacking a genuinely private generator, predict
exactly the same table.  The flat row was consistent with privacy but was not
evidence of it, because the experiment could not have distinguished the two.

The attack grid needs re-running against the corrected generator before the
DP-PGM column means anything, and the useful result is a privacy/utility curve
over ε rather than a single flat row.

---

### 7f. A third defect: the sensitivity and the row count disagreed

Found on re-verifying the accounting end to end, after 7b was already fixed.

The noise scale assumed **L2 sensitivity 1**, which is the sensitivity of a
marginal count vector under *add/remove* neighbours (one patient inserted or
deleted moves exactly one cell by 1).  That is the convention MST and AIM use.
But the fitter also passed **the exact training-set row count** to
`FactoredInference`, and under add/remove the row count is itself sensitive —
two neighbouring datasets have different sizes, so releasing n exactly
distinguishes them with certainty.  Stratified mode did it twice, additionally
sizing each class's synthetic sample from the exact per-class counts, which
releases the class histogram.

The alternative convention does not rescue it.  Under *replace* neighbours n is
public and safe to release, but one patient changing moves one cell down and
another up, so Δ₂ = √2 and **every σ must grow by √2** (ρ costs 2×).  The
implementation had neither guarantee: Δ₂ = 1 noise with a bounded-DP release of
n.

The fix costs nothing.  `mbi` will estimate the total itself from measurements
already paid for, as the minimum-variance unbiased combination of the noisy
1-way marginals (each of which sums to n).  At 978 marginals and σ = 28.8, the
estimate of n = 871 has sd **1.84, or 0.21%**.  We now pass `total=None` and let
it, exactly as MST does; stratified mode allocates from the submodels' noisy
totals.  `neighboring="replace"` is available for anyone who wants the bounded-DP
reading and will pay the √2.

Note that parallel composition over the per-class subsets in stratified mode —
which is what lets each class spend the *full* ρ — is also only valid under
add/remove.  Under replace a changed patient can move between classes and touch
two submodels, so their budgets would compose sequentially.  A third reason the
default is add/remove.

**And a defect in the legacy arm, left unfixed deliberately.**
`composition="basic"` uses δ = 1e-5 for *each* of the 1956 marginals without
dividing by k.  Basic composition adds δ as well as ε, so its σ really
corresponds to (10, **0.0196**)-DP, not (10, 1e-5) — δ under-reported by 1956×.
A correct basic accounting would use δ/k and be a further **1.28×** noisier.  We
have not changed it, because `basic` exists only to reproduce pre-2026-09-20
results.  But it means every comparison against `basic` in 7c-ii and 7c-iii
slightly **flatters** it: the honest zCDP advantage is a little larger than
measured.  The paper must describe `basic` as "the original implementation,"
never as "basic composition correctly applied."

Full derivation, in plain English and from first principles, in
`docs/ZCDP_ACCOUNTING.md`.

---

## 8. Where we disagree with the abstract

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

The second disagreement is about DP-PGM, and it is a disagreement with our own
submitted numbers as much as with the abstract's.  Both report the DP-PGM
column at chance and read it as the differentially private generator doing its
job.  Finding 7 shows the experiment could not have distinguished that from two
defects — permuted gene columns, and marginals carrying seven times their own
signal in noise — which predict the identical table.  The claim may well survive
re-running against the corrected generator, but it is not currently supported,
and the paper should not repeat it until the grid has been re-run.

---

## Reproducing

```bash
python scripts/run_experiment.py configs/experiments/tune_statistical.yaml
python scripts/analyse_quantile_leak.py --dataset BRCA
python scripts/sanity_check.py --dataset BRCA
```

---

## 9. DP-PGM at ε=10 is not at chance, and MAMA-MIA is what sees it

Everything here is against the **corrected** generator (all three defects in §7
fixed): zCDP composition, add/remove sensitivity with an estimated row count,
canonical gene order. Targets rebuilt 2026-09-20; every DP-PGM number recorded
before that date is superseded.

### 9a. The flat row was a ceiling, and it lifted

A member contributes exactly **+1 count** to each marginal cell they occupy.
Two noise layers stand against it: the Gaussian DP noise (σ, in absolute counts)
and multinomial sampling of `n_syn` synthetic rows. So the per-cell SNR is
`1/sqrt(σ² + n_syn·p(1−p))`, and over k roughly independent cells the separation
is `√k` times that. Treating PGM inference as lossless gives a **ceiling** on any
marginals-based attack:

| cohort | n | d (1-way) | d (2-way) | d total | **AUC ceiling** |
|---|---|---|---|---|---|
| BRCA | 871 | 0.992 | 1.437 | 1.746 | **0.892** |
| COMBINED | 3458 | 0.813 | 1.401 | 1.620 | **0.874** |

Under the *old* accounting (σ = 1435.8) the same calculation gives a ceiling of
**0.514**. So the flat DP-PGM row was not a weak attack and not evidence of
privacy — **no attack could have exceeded 0.514**, and MAMA-MIA measured 0.50.
The experiment could not have distinguished a private generator from a broken
one, which is exactly what §7e warned.

### 9b. Measured, before and after

Unmodified MAMA-MIA and MahalaMIA, five splits, ε=10, δ=1e-5:

| cohort | attack | before (n runs) | **after** |
|---|---|---|---|
| BRCA | mamamia | 0.4959±0.0088 (27) | **0.6041±0.0117** |
| BRCA | mahalamia | 0.5002±0.0114 (242) | 0.5037±0.0128 |
| COMBINED | mamamia | 0.4995±0.0050 (20) | **0.5297±0.0042** |
| COMBINED | mahalamia | 0.5000±0.0040 (255) | 0.5048±0.0044 |

**The effect is specific to the marginal attack.** MahalaMIA — the strongest
attack on every other generator — stays at chance on both cohorts. That is the
cleanest evidence we have that the signal genuinely lives in the released
marginals rather than in gene-space geometry, and it is a point worth making in
the paper: attack and mechanism have to match.

### 9c. Aggregation is worth more than anything else we changed

`scripts/mamamia_aggregation.py`, `results/mamamia_aggregation.csv`. Focal-point
selection is not a free parameter here: the cohorts carry exactly 978 genes and
the release sets `n_1way=978`, so "top 978 by variance" is *every* gene, and
with `n_2way=0` the clique set (978 one-way + 978 gene×label) is fixed by public
configuration. Every arm below targets the same known marginals.

| arm | BRCA AUC | BRCA T@1%FPR | COMBINED AUC |
|---|---|---|---|
| ratio (shipped) | 0.6042±0.0117 | 0.0161 | 0.5297±0.0042 |
| log | 0.6252±0.0096 | 0.0202 | 0.5312±0.0060 |
| log, inverse-variance | 0.6121±0.0116 | 0.0202 | 0.5292±0.0061 |
| log, 1-way only | 0.6437±0.0150 | 0.0484 | 0.5721±0.0062 |
| log, 1-way only, class-centred | 0.6663±0.0095 | **0.0503** | **0.5864±0.0052** |
| **log, class-centred** | **0.6706±0.0214** | 0.0402 | 0.5407±0.0032 |

Three findings, two of which contradict predictions made earlier in this file.

**(i) Log space beats ratio space**, by 2.1 points on BRCA and 6.6 points of
TPR@10%FPR. The mean of `p_syn/p_aux` is dominated by cells where `p_aux` is
small — the least reliable ones — while the sum of log-ratios is the
Neyman–Pearson statistic. This is the one prediction that held.

**(ii) Inverse-variance weighting HURTS** (0.6121 against 0.6252 unweighted),
contradicting the proposal in `docs/PGM_ATTACK_SURFACE.md` §6.3(ii). It
up-weights the gene×label family 2.2×, because that family gets 67% of the
budget and so carries lower σ. But σ is not what limits that family — see (iii).
**The quieter marginals are the less useful ones.** Noise-optimal weighting is
not signal-optimal.

**(iii) The gene×label family is a net negative, and the reason is subtype
confounding.** One-way marginals alone beat both families combined on *both*
cohorts (BRCA 0.6437 vs 0.6252; COMBINED 0.5721 vs 0.5312 — a 4-point loss from
adding them). Their internal correlation is low (mean |r| 0.045–0.089), so
redundancy is not the explanation. Subtype is: a candidate's 2-way score depends
on their label, and label predicts *class*, not *membership*, so the family adds
a within-class constant that dilutes the signal. Centring each score within
subtype confirms it — the 2-way family gains the most (BRCA 0.5992 → 0.6389),
the 1-way family gains less (0.6437 → 0.6663), and on BRCA the combination
finally beats 1-way alone (0.6706). On COMBINED, with 12 classes, even centred
the 2-way family never pays for itself (0.5864 for centred 1-way alone).

This is the same mechanism as finding 6d, where class-conditional scoring took
COMBINED/MVN to AUC 1.000: subtype is a confounder in every attack that scores
a mixed cohort, and removing it is nearly free.

Net: **0.604 → 0.671 on BRCA and 0.530 → 0.586 on COMBINED, from aggregation
alone** — no change to which marginals are targeted, no shadow models, no
recovery of the generator's bin edges. Against the 9a ceilings (0.892, 0.874)
there is still substantial headroom.

**Threat-model note.** Class-centring uses each candidate's subtype, which is
published with the challenge data, but it also uses *other candidates' scores*
— it is a transductive operation over the scored pool. That is consistent with
how the challenge is evaluated (the whole pool is scored at once) and with what
`sigmoid_calibrate` already does, but it should be stated rather than assumed.

### 9d. A calibration bug that cost 12 points

Worth recording because it nearly inverted the headline of 9c. `sigmoid_calibrate`
in `mia/attacks/mahalamia.py` began `np.log(np.maximum(raw, 1e-300))`, which
assumes strictly positive scores — true for ratio-space scores. Log-ratio scores
are *negative*, so every one of them was clamped to `1e-300`, collapsing the
entire ranking into one tie. The log arm measured **0.508** through that path
against **0.625** computed directly.

The docstring asserted "Rank-preserving, so AUC is untouched," which is true only
for positive input, and the first version of this finding reported that ratio
space beat log space "decisively" on the strength of it. The function now takes
`log_transform` and raises on non-positive input rather than silently clamping.

## 10. The binning leak, measured; and the two DP-safe fixes

**Every DP-PGM number before this section, including all of §9, used the legacy
`quantile` binning.** Its bin edges are percentiles of the private training
split and spend no budget, so those releases are not ε-DP end to end. Targets
built that way now carry `"binning": "quantile"` in `meta.json` (backfilled
2026-09-21); `results/pgm_eps_sweep.csv` has a `binning` column.

Two fixes, both in `~/private-pgm-rnaseq-camda2026/src/discretization.py`:

* `binning=uniform`: equal-width bins on the fixed range [0, 24] (log2
  expression), which reads no data and spends nothing.
* `binning=dp_quantile`: equal-depth edges from a Gaussian-noised histogram on a
  public 48-cell grid, spending 10% of ρ. The marginals get the other 90%.
  Tests check that the whole pipeline spends exactly ρ_total.

Data: 5 splits per cell. Attack AUC is the best of five fixed log-score arms
(log; 1-way; class-centred; 1-way class-centred; 2-way class-centred), taking
each arm's mean first. "Bound" is Φ(√ρ), the highest AUC *any* test can reach
if the release really is ρ-zCDP. The aux-edge rows are `results/pgm_eps_sweep.csv`.
The gen-edge rows are `results/pgm_aligned_edges.csv` (`scripts/pgm_aligned_edges.py`).

| cohort | binning | ε | bound | attack, aux edges | attack, gen edges | utility ratio | W1 |
|---|---|---|---|---|---|---|---|
| BRCA | quantile (legacy) | 0.1 | 0.508 | **0.554** ±.013 | 0.551 | 0.30 | 0.70 |
| BRCA | quantile (legacy) | 0.3 | 0.523 | **0.620** ±.014 | 0.524 | 0.28 | 0.79 |
| BRCA | quantile (legacy) | 1 | 0.569 | **0.606** ±.014 | 0.543 | 0.41 | 0.66 |
| BRCA | quantile (legacy) | 10 | 0.909 | 0.671 | 0.718 | 0.76 | 0.51 |
| BRCA | uniform k=4 | 0.3 | 0.523 | 0.502 | 0.510 ±.010 | 0.19 | 6.34 |
| BRCA | uniform k=16 | 0.3 | 0.523 | 0.504 | 0.515 ±.011 | 0.19 | 7.09 |
| BRCA | dp_quantile | 0.3 | 0.523 | 0.505 | 0.502 | 0.19 | 6.32 |
| BRCA | uniform k=4 | 10 | 0.909 | 0.503 | 0.568 | 0.63 | 3.60 |
| BRCA | uniform k=16 | 10 | 0.909 | 0.528 | 0.562 | 0.65 | 1.81 |
| BRCA | dp_quantile | 10 | 0.909 | 0.507 | **0.604** | 0.70 | 2.30 |
| COMBINED | quantile (legacy) | 0.3 | 0.523 | **0.547** ±.006 | 0.517 | 0.24 | 0.74 |
| COMBINED | quantile (legacy) | 10 | 0.909 | 0.586 | 0.632 | 0.93 | 0.62 |
| COMBINED | uniform k=4 | 0.3 | 0.523 | 0.502 | 0.503 | 0.13 | 3.75 |
| COMBINED | uniform k=16 | 0.3 | 0.523 | 0.504 | 0.507 | 0.10 | 3.93 |
| COMBINED | dp_quantile | 0.3 | 0.523 | 0.503 | 0.504 | 0.12 | 3.61 |
| COMBINED | uniform k=4 | 10 | 0.909 | 0.501 | 0.515 | 0.76 | 2.78 |
| COMBINED | uniform k=16 | 10 | 0.909 | 0.505 | 0.523 | 0.80 | 1.22 |
| COMBINED | dp_quantile | 10 | 0.909 | 0.509 | **0.560** | 0.90 | 2.51 |

### 10a. The legacy release breaks its own guarantee, and the leak is the edges

Under legacy binning the attack beats the ε bound at ε ≤ 1 on BRCA, by about
7 SE at ε=0.3, and at ε=0.3 on COMBINED. That is impossible for a ρ-zCDP
release, so something outside the accounting is leaking, and two things
identify it as the edges:

* On BRCA the aux-edge attack is **flat from ε=0.3 to ε=3** (0.62, 0.61, 0.63).
  The marginal noise changes by 10× over that range and the attack does not
  notice.
* Binning with the target's own edges instead *removes* most of the excess at
  low ε (BRCA ε=0.3: 0.620 → 0.524). The aux-edge attack wins because its
  cells are misaligned with the private percentiles. The synthetic mass in each
  auxiliary cell encodes where the member-dependent edges fell, and that is
  membership signal. MAMA-MIA was exploiting the binning leak without being
  designed to.

### 10b. With either fix, nothing exceeds the bound

At ε=0.3 every arm on both cohorts, with either edge choice, is at or below
0.515 against a bound of 0.523. That is consistent with an end-to-end DP
release, and it is what the legacy runs failed.

### 10c. Attacks must bin with the generator's cells

Against the fixed releases, aux-quantile edges understate the risk badly. At
ε=10 on BRCA dp_quantile they give 0.507, while the generator's own edges give
0.604. Both edge sets are legitimately the attacker's: uniform edges are a
public configuration, and dp_quantile edges are a DP output that can be read
off the release. **Report gen-edge numbers for fixed binning.** For legacy
binning, the gen-edge column is an oracle, since those edges are private.

### 10d. The price

At ε=10 the fixes cost utility. BRCA utility ratio falls from 0.76 to 0.70
(dp_quantile) or 0.63–0.65 (uniform); COMBINED from 0.93 to 0.90 or 0.76–0.80.
Per-gene W1 rises 3–7×, because uniform cells on [0, 24] are wide and
dithering within them smears the values. dp_quantile at 10% of ρ is the best
of the three at ε=10 on both cohorts. At ε=0.3 its edges carry about σ=1200
of noise per grid cell and are effectively random, so it ties uniform. The
discriminator AUC is 1.00 for every variant, legacy included.

The corrected DP-PGM column of §9 (MAMA-MIA 0.604 / 0.530) is therefore
partly a binning result. Against a DP-safe release at ε=10, the best measured
attack is **0.604 (BRCA) / 0.560 (COMBINED)**, with dp_quantile and aligned
edges. The shipped MAMA-MIA, with aux edges, is at chance.
