# Experiment backlog

Planned work toward the full conference paper.  Each item is also tracked as a
task in the session task list; this file is the durable copy.

Adding an experiment means copying a YAML file in `configs/experiments/`,
changing the fields in question, and running it — results land in the same
`results/index.csv` as everything else, so ablations stay comparable to the
headline grid by construction.

---

### 1. Each attack on each SDG method (4x4 grid) — *in progress*

Score all four attacks (MahalaMIA, MeLoMIA-ND, MeLoMIA-CVAE, MAMA-MIA) against
synthetic data from all four generators (MVN, CVAE, ND, DP-PGM), on TCGA-BRCA
and TCGA-COMBINED, five splits per cell.

The diagonal is each attack against the generator it was designed for and
reproduces the abstract's `Ours` row.  The off-diagonal is the interesting part:
it separates "this attack works because it knows the generative mechanism" from
"this attack works because the synthetic data leaks regardless".  The abstract
already hints that the separation is not clean — MahalaMIA, built for MVN,
scored AUC 0.986 against CVAE on BRCA, *beating* the CVAE-specific MeLoMIA.

Configs: `configs/experiments/grid_brca.yaml`, `grid_combined.yaml`.

### 2. MeLoMIA with many more than 5 shadow models

The abstract's ND numbers came from K=5 shadows, inherited from the blue team's
five published splits rather than chosen.  With K=5 the meta-classifier sees
only five independent membership draws per sample, which is very thin.

Sweep K over {5, 10, 15, 25, 50, 100, 200} for both backends and plot attack
performance against K, with wall-clock per shadow recorded so the compute
trade-off is explicit.  Expect monotone improvement with saturation; find the
knee.  The current grid runs at K=30 (BRCA) and K=20 (COMBINED), so this
extends in both directions.

### 3. Ablate the MeLoMIA sweep axis

Which part of the loss trajectory actually carries membership?

* **ND — timesteps.**  Ablate the 15-point timestep superset: single timesteps,
  low-t only, high-t only, coarse vs fine grids, versus the Optuna-selected
  subsets.  Low t should dominate (only a memorised sample denoises exactly
  from near-clean input), but that is a prediction, not a result.
* **CVAE — temperatures.**  Ablate alpha over {0, 0.5, 1, 1.5, 2, 3}: alpha=0
  alone (pure posterior-mean reconstruction) versus the full sweep, and the
  128-dimensional per-dimension KL block on its own.
* **Both — draw budget N.**  Cost is linear in N; find where it stops paying.

Deliverable: an AUC-versus-sweep-point figure for the paper.

### 4. PCA on genes before the attack — *first pass done for MahalaMIA, and it failed*

Project the 978 landmark genes onto a lower-dimensional PCA basis first, and
measure the effect on each attack.

For MahalaMIA the result is already in, and it is the opposite of what motivated
it.  Projecting onto the leading PCA directions of the synthetic data does not
merely fail to help — it drives the attack *below* chance on BRCA (AUC 0.41 at
k=50, 0.44 at k=200, 0.65 at k=500, versus 0.93 at full rank).  Meanwhile simply
ridge-regularising the full-rank covariance takes MVN from 0.93 to **1.00**
(`configs/experiments/tune_statistical.yaml`).

Read together, those two say the membership signal lives in the *low-variance*
directions — exactly the ones PCA discards and the pseudo-inverse down-weights.
That makes sense: the high-variance directions are dominated by biological
structure shared by members and non-members alike, while a generator's
over-fitting to its training set shows up in the directions where the synthetic
covariance is nearly singular.  Worth writing up, and worth testing directly by
scoring on the *trailing* components instead of the leading ones.

Still open: PCA for MeLoMIA, where it would shrink the shadow generators and may
denoise the loss features rather than discard the signal.  Fit the basis only on
data the adversary legitimately holds — the released synthetic set, or the
auxiliary reference — never on `D_real`.

### 5. More cohorts and more cohort sizes

Leakage should fall as the training set grows, and the abstract already hints at
it (MVN and CVAE attacks are weaker on the larger COMBINED).  Turn the hint into
a curve: subsample COMBINED to n in {500, 1000, 2000, 4323} with genes and
generator hyperparameters held fixed, so the only thing varying is n.  Doing it
by subsampling one cohort rather than comparing BRCA to COMBINED avoids
confounding size with tissue composition.

Optionally add further TCGA cohorts, or the CAMDA Track II single-cell data.

### 6. Vine copula in place of MahalaMIA's covariance

MahalaMIA summarises the synthetic distribution by a mean and a covariance,
which assumes joint Gaussianity and sees only linear dependence.  That is exactly
right for the MVN generator and increasingly wrong for everything else.

Replace it with a vine copula fitted to `D_synth` (`pyvinecopulib` or
`rvinecopulib`): per-gene marginals plus a regular vine of bivariate copulas for
the dependence structure, scoring candidates by copula log-density instead of
Mahalanobis distance, keeping the same aux/no-aux ratio structure.  Should be
strictly more expressive for tail and non-linear gene-gene dependence, and is
the natural route to making MahalaMIA competitive against non-Gaussian targets.

Main obstacle is 978 dimensions — likely needs item 4 first, or a truncated vine.

### 7. White-box versus black-box

The CAMDA setting is black-box: no target parameters, so MeLoMIA must train a
proxy on `D_synth` as a stand-in for the unobserved target.  We train the targets
ourselves, so we can also run the attack against the *actual* target model and
measure what the substitution costs.

That gap bounds how much worse a realistic adversary is than an idealised one
and is a headline number for the paper.  Worth adding the middle rung too:
query access to the target without parameter access.

Needs `--retrain-nd` targets (so the ND column's weights are ours) and a
white-box mode in the MeLoMIA backends that extracts from the target rather
than from a proxy.

### 8. With and without synth-shadow modelling

The head-to-head that justifies the central methodological claim.  Identical
splits, identical features, identical meta-classifier; only the feature-extraction
shadows differ:

  (a) trained directly on real-data splits;
  (b) trained on internal synthetic datasets produced by base shadows.

Earlier notes in this repo record ND real-data shadows reaching validation
TPR@10%FPR of 0.58 and collapsing to about 0.14 when scored through a
synthetic-trained proxy — the domain gap synth-shadow modelling exists to close.
Reproduce that cleanly for both backends on both cohorts.

---

### 9. Matched base shadows for the off-diagonal grid cells

A wrinkle in how the 4x4 grid is currently run.  MeLoMIA's base shadows are
generators of the *probe's* family — ND base shadows for MeLoMIA-ND — which is
what the CAMDA submission did and is right on the diagonal.  Off the diagonal it
is not: attacking MVN-generated data with MeLoMIA-ND trains the meta-classifier
on features from ND-synthetic data and then applies it to a proxy fitted to
MVN-synthetic data.  That is precisely the train/inference domain mismatch
synth-shadow modelling was introduced to remove, so a weak off-diagonal cell may
be an artifact of the setup rather than a fact about the generator.

`MeLoMIA(base_generator=...)` already implements the fix: point the base shadows
at the target's family so the internal synthetic data resembles what the target
actually released, while the probe stays whichever family can measure a loss.
Re-run the off-diagonal MeLoMIA cells that way and compare.  MVN and CVAE base
shadows are cheap (seconds and under a minute); DP-PGM base shadows are not
(~12 min each), so that column may need a smaller K.

Until this is run, read the off-diagonal MeLoMIA numbers as a lower bound.

## Smaller follow-ups

* **DP-PGM attack is weak and we know it.**  MAMA-MIA reaches roughly
  AUC 0.52 here, consistent with the abstract's 0.528.  Whether that is DP
  working as intended or the attack being mis-targeted is unresolved — worth
  trying marginals matched to the generator's *selected* set rather than all
  978 one-way and gene x subtype marginals.
* **MAMA-MIA is accidentally a good attack on NoisyDiffusion.**  Bin count was
  assumed to want matching to DP-PGM's own discretisation, and for DP-PGM it
  roughly does.  Against ND it does not: AUC climbs monotonically with
  resolution — 0.54 at 4 bins, 0.61 at 16, 0.68 at 32, 0.76 at 64 — because a
  diffusion model reproduces fine-grained per-gene marginal structure that
  coarse bins wash out.  The one-way per-gene marginals carry it on their own
  (0.79 at 64 bins on split 1, versus 0.73 for the two-way gene x subtype
  marginals and 0.76 combined), which says the attack has become a per-gene
  nearest-value detector rather than a test on marginal *shape*.  Negative
  controls pass.  Find where it tops out, check whether it holds on COMBINED,
  and decide whether to present it as MAMA-MIA at all or as a separate
  quantised-nearest-value baseline.  Either way, a marginal-ratio attack beating
  a tailored loss-trajectory attack on a diffusion model needs explaining.
* **Sanity-check every grid before reporting it.**  `scripts/sanity_check.py`
  runs the negative controls (permuted labels must fall to chance; scores from
  one split's target must not predict another split's labels).  The ridge result
  above passes both, which is the reason to believe AUC 1.00 rather than hunt
  for the leak.
* **Cross-cohort shadows.**  Can a shadow stack trained on BRCA attack COMBINED
  targets?  Cheap to test and speaks to how transferable the attack is.
* **Calibration.**  Scores are currently rank-calibrated only.  If the
  leaderboard rewards calibrated probabilities, fit an isotonic map on held-out
  shadows.
