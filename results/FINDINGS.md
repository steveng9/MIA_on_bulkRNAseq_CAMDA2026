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

### The MVN result is a p ≈ n effect; the CVAE result is not

COMBINED separates the two.  It has the same 978 genes but 3,458 training
samples instead of 871, so the sample covariance is comfortably full rank:

| MahalaMIA | cohort | n/p | MVN | CVAE | ND | DP-PGM |
|---|---|---|---|---|---|---|
| pseudo-inverse | BRCA | 0.89 | 0.928 | 0.891 | 0.806 | 0.503 |
| ridge, α = 1e-6 | BRCA | 0.89 | **1.000** | **0.996** | 0.826 | 0.503 |
| ridge, α = 1e-2 | BRCA | 0.89 | 0.959 | 0.844 | 0.752 | 0.504 |
| pseudo-inverse | COMBINED | 3.54 | 0.900 | 0.619 | 0.770 | 0.500 |
| ridge, α = 1e-6 | COMBINED | 3.54 | 0.888 | 0.780 | 0.765 | 0.500 |
| ridge, α = 1e-2 | COMBINED | 3.54 | **0.924** | **0.799** | 0.799 | 0.500 |

Ridge helps on both cohorts, but *the alpha that helps inverts*, and so does the
ceiling.  On BRCA the attack improves monotonically as α falls, saturating at
1.000 by α = 1e-6.  On COMBINED that ordering reverses -- 1e-6 is the worst
setting tried and does worse than the pseudo-inverse -- and the best result so
far, 0.924 at α = 1e-2, is nowhere near saturation.

That inversion is the signature of a conditioning effect rather than a property
of the generator.  At n < p the sample covariance is singular; the generator's
over-fit to its training set lives in the near-null directions, so the useful
move is to add as little as possible to the diagonal and keep them.  At n > p
there are no null directions to recover, and the ridge is doing ordinary
variance reduction on a well-conditioned estimate, which wants a much larger α.

So the honest claim is not "ridge breaks the MVN generator" but *a per-class
Gaussian generator becomes drastically more exposed as its training set shrinks
toward the gene count* -- AUC 0.92 at n/p = 3.5, and 1.00 at n/p = 0.89.  That
is the regime most single-cohort RNA-seq studies are actually in.

The CVAE column behaves differently again: conditioning is worth +0.10 on BRCA
and **+0.18** on COMBINED, so it strengthens with n and cannot be a rank
artefact.  The VAE's reconstruction error concentrates in low-variance gene
directions regardless of sample count, and the pseudo-inverse discards exactly
those.

### The sweep: attack strength is set by n/p, and the submitted estimator hid it

`scripts/cohort_size_sweep.py` holds everything fixed but n.  One cohort
(COMBINED), one generator, one attack; the training set is resampled to each
size at p = 978 genes, three trials each, so cohort composition, class count and
tissue heterogeneity cannot explain any of the trend.

| n | n/p | pseudo-inverse | ridge α=1e-6 | ridge α=1e-2 |
|---|---|---|---|---|
| 500 | 0.51 | 0.837 | **1.000** | 1.000 |
| 700 | 0.72 | 0.904 | **1.000** | 1.000 |
| 871 | 0.89 | 0.963 | **1.000** | 0.999 |
| 1100 | 1.12 | 1.000 | 1.000 | 0.999 |
| 1500 | 1.53 | 0.999 | 0.998 | 0.995 |
| 2000 | 2.04 | 0.986 | 0.980 | 0.982 |
| 2800 | 2.86 | 0.944 | 0.930 | 0.954 |
| 3458 | 3.54 | 0.899 | 0.888 | 0.924 |

(Figure: `results/figures/cohort_size_COMBINED_auc.pdf`.  All 152 COMBINED
MahalaMIA runs, these included, pass the shuffled-label and cross-split
controls.)

Two things fall out, and the second is the one that matters.

**A properly conditioned attack decays monotonically in n/p.**  With ridge at
1e-6 the attack is perfect — AUC 1.000 *and* TPR 1.000 at 10% FPR, meaning every
member is recovered without spending any false-positive budget — everywhere at
n ≤ p, and falls steadily to 0.888 by n/p = 3.5.  The MVN generator's exposure
is a smooth function of how many samples it was fitted to, not a property of a
particular cohort.  BRCA is not unusual; it is simply at n/p = 0.89.

**The pseudo-inverse is non-monotone, and it fails worst exactly where the risk
is greatest.**  It peaks at n/p ≈ 1.1 and falls off on *both* sides: 1.000 at
1.12 but 0.837 at 0.51.  That drop is a property of the estimator, not of the
generator — below n = p the pseudo-inverse discards the near-null directions,
and those are precisely where a Gaussian fitted to too few samples imprints its
training set.

So the version of MahalaMIA that was submitted **understates the risk most in
the regime where the risk is highest**.  At n/p = 0.51 it reports 0.837 where
the true exposure is 1.000.  A blue team that benchmarked against it and
concluded a small cohort was acceptably safe would have been reading an artefact
of the attack's linear algebra.  For a privacy evaluation that is the worst
possible direction for an error to run, and it is the strongest argument in this
work for reporting attacks with a conditioned covariance.

The CVAE column behaves differently again: conditioning is worth +0.10 on BRCA
and **+0.18** on COMBINED, so it strengthens with n and cannot be a rank
artefact.  The VAE's reconstruction error concentrates in low-variance gene
directions regardless of sample count, and the pseudo-inverse discards exactly
those.

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
