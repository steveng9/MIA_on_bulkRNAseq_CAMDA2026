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

## 3. Where we disagree with the abstract

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
