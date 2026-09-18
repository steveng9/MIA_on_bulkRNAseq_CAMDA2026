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

### 4. PCA on genes before the attack

Project the 978 landmark genes onto a lower-dimensional PCA basis first, and
measure the effect on each attack.

Strongest motivation is MahalaMIA: it inverts a 978x978 covariance estimated
from ~870 synthetic samples, which is rank-deficient and currently handled with
a pseudo-inverse.  A k << n basis gives a better-conditioned, better-estimated
covariance.  For MeLoMIA it shrinks the shadow generators and may denoise the
loss features.

Sweep k over {10, 25, 50, 100, 250, 500, 978}.  Fit the basis only on data the
adversary legitimately holds — the released synthetic set, or the auxiliary
reference where one exists — never on `D_real`, or the comparison is meaningless.

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

## Smaller follow-ups

* **DP-PGM attack is weak and we know it.**  MAMA-MIA reaches roughly
  AUC 0.52 here, consistent with the abstract's 0.528.  Whether that is DP
  working as intended or the attack being mis-targeted is unresolved — worth
  trying marginals matched to the generator's *selected* set rather than all
  978 one-way and gene x subtype marginals.
* **Cross-cohort shadows.**  Can a shadow stack trained on BRCA attack COMBINED
  targets?  Cheap to test and speaks to how transferable the attack is.
* **Calibration.**  Scores are currently rank-calibrated only.  If the
  leaderboard rewards calibrated probabilities, fit an isotonic map on held-out
  shadows.
