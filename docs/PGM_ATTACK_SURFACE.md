# What DP-PGM actually exposes, and the choices MAMA-MIA has to make

Working notes for tailoring MAMA-MIA to StratHiM-PGM.  **Nothing here is
implemented.**  This is the surface map plus the open questions, written so the
design decisions stay with the person who built MAMA-MIA.

Read `mia/attacks/mamamia.py` for the current implementation and
`~/private-pgm-rnaseq-camda2026/src/` for the generator.

---

## 0. A correction that precedes everything

Until 2026-09-20 our adapter returned DP-PGM's synthetic matrix with its
**columns permuted**.  The upstream generator selects genes by variance and
returns them in selection order, labelling them through `selected_gene_names`;
our `sample()` dropped the labels and treated column *j* as gene *j*.

Global moments are invariant to a column permutation, so nothing that looked at
the matrix as a whole could see it.  Per gene it was severe: mean Wasserstein
distance 2.9 training SDs, downstream macro-F1 0.09 against a real ceiling of
0.81, and a real-vs-synthetic discriminator at AUC 1.000.

**Every DP-PGM result in the repo predates the fix**, including the finding that
all four attacks sit at chance against it.  That finding is not yet safe to
quote: a column-scrambled dataset has had its per-gene membership signal
destroyed by construction, so "DP-PGM resists our attacks" and "we attacked
noise" are indistinguishable on the current numbers.  The column has to be
rebuilt before any of it means anything.

---

## 1. The measured marginals are the whole surface

DP-PGM touches the private data exactly three times:

1. **Marginal selection** — top-`n_1way` genes by variance, top-`n_2way` gene
   pairs by |Spearman|.  *Currently not privatised upstream.*  This is itself a
   leak: which genes were selected is a function of the training split's
   variances.  It is also an attack surface nobody has used.
2. **Discretisation** — per-gene quantile edges fitted on the training split.
   Also not privatised, and also a leak: the edges are the training split's
   quantiles.
3. **The noisy measurements** — Gaussian mechanism on the selected marginals,
   budget split by `budget_weights` across orders.

Everything downstream (PGM inference, sampling, decoding) is post-processing and
adds no leak, but it does *attenuate* what is there.

Our grid's target configuration is `epsilon=10, n_bins=4, n_1way=978, n_2way=0,
joint_mode=True`, which measures 978 one-way gene marginals plus 978 gene×label
two-way marginals, and no gene–gene structure at all.  That is deliberately last
year's CAMDA winner's structure.  It is *not* the upstream default
(`joint_mode=False, n_bins=8, n_2way=150`), and it is close to the least
informative setting the generator supports.

---

## 2. Four mismatches between MAMA-MIA as written and this generator

### 2.1 The attack bins on the wrong edges

`mia/attacks/mamamia.py` estimates quantile edges from the auxiliary pool.  The
generator fits its edges on the training split.  Close, but not the same cells,
and every misaligned candidate contributes noise instead of signal.

The edges are *recoverable*, though.  Decoding dithers uniformly inside each
bin, so the released data is piecewise-uniform with density discontinuities
exactly at the generator's edges.  A change-point or kernel-density scan of each
synthetic gene should recover them to within sampling error, with no auxiliary
data needed.  This looks like the single highest-value change, and it is
generator-specific in the way you have been arguing attacks should be.

### 2.2 The attack weights all marginals equally

`budget_weights=(0.33, 0.67)` puts two-thirds of the budget on the gene×label
marginals, so under the Gaussian mechanism they carry substantially less noise
per cell than the one-way marginals.  The attack currently sums both families
unweighted.  Under Kerckhoffs the adversary knows ε, δ and the split, so it can
weight each marginal by its inverse noise variance — the standard
inverse-variance combination, which is optimal for independent estimates.

### 2.3 The sampling noise floor may dominate the signal

With 871 synthetic rows and 4 bins, a one-way cell holds ~218 expected counts.
One member moves that cell by about 1/871 ≈ 0.0011; the multinomial sampling SD
on the same cell is about 0.015 — roughly 14× larger.  Averaging 978 genes buys
back a factor of ~31, leaving an SNR around 2.4 before DP noise is even
considered.  That is consistent with the 0.50 AUC we measure.

If that analysis is right, the binding constraint is `n_syn`, which is the
*generator's* choice, not the adversary's.  A release of 10× more synthetic rows
would be a strictly more useful release and a strictly more attackable one.
Worth measuring as a curve.

### 2.4 Stratified mode changes the surface entirely

In `joint_mode=False` there is no label node and no gene×label marginal, so half
of what MAMA-MIA targets does not exist.  Instead there are *per-class* PGMs,
which means per-class marginals.

**Correction (2026-09-20): this section originally called per-class marginals
"arguably a better target, since they are estimated from far fewer rows." That
is backwards.** The member's contribution to any cell is +1 *in absolute count*,
and the DP noise is σ *in absolute count*, so the per-cell SNR is 1/σ regardless
of how many rows the marginal was estimated from — the "fewer rows" intuition is
a relative-error intuition and does not transfer to additive noise. What
actually changes is how many cells the member touches: in joint mode they move
978 one-way cells *and* 978 gene×label cells; in stratified mode only the 978
one-way cells of their own class's submodel. Half the measurements, so roughly
**√2 less** total signal. Stratified mode is a *worse* target, not a better one.

(Each class does get the full ρ under parallel composition, so σ is unchanged
per class. Small classes also suffer heavy PGM clipping distortion — BRCA's
rarest class has 32 training rows, so an expected cell count of 8 carries σ≈29
of noise and is dominated by the non-negativity projection.)

---

## 3. Questions for Steven

1. **Recovering the generator's bin edges from the released data** — is that the
   right first move, and does the original MAMA-MIA do anything analogous
   against MST/PrivBayes, or did those expose their discretisation directly?

2. **Inverse-variance weighting across marginal orders** — is that what you did,
   or does MAMA-MIA weight by something else (marginal cardinality, observed
   ratio dispersion, a learned weight)?

3. **The ratio form.** Currently `p_syn / p_aux`, averaged over marginals.
   Options we could use instead: a likelihood ratio under the known Gaussian
   noise model, a per-cell z-score against the sampling null, or the
   leave-one-out form. Which is closest to the paper, and which do you think
   suits a generator whose marginals are post-processed through PGM inference?

4. **Should we attack the non-private steps?** Gene selection and the
   discretisation edges are both fitted on the training split with no budget
   spent. They are a genuine vulnerability in this implementation and probably
   in last year's winner too. That is a different paper section from MAMA-MIA
   proper — do you want it as its own attack, or folded in?

5. **Which configuration is "the" DP-PGM column?** The as-submitted one
   (`joint_mode=True, n_2way=0`) is what the challenge released. The tuned one
   will be a better generator and a different attack surface. My instinct is to
   report both and make the privacy/utility trade-off the point, rather than
   pick one.

6. **Stratified vs joint** — is per-class marginal attack something MAMA-MIA
   already handles, or a new variant?

---

## 4. The budget is being spent under the wrong composition theorem

Measured on BRCA split 1 at the grid's configuration (ε=10, δ=1e-5, N=871,
`n_1way=978, n_2way=0, joint_mode=True`, so 978 one-way plus 978 gene×label
marginals).  From the fitter's own log:

```
[pgm_fitter] 1-way: 978 marginals, eps_per=0.0034, sigma=1435.8
[pgm_fitter] 2-way: 978 marginals, eps_per=0.0069, sigma= 707.2
```

`PrivatePGMFitter._build_measurements` splits the budget **linearly** across
marginals -- `eps_per = frac * epsilon / len(cliques)` -- and then calibrates each
one independently with the classical Gaussian-mechanism bound
`sigma = sqrt(2 ln(1.25/delta)) / eps_per`.  That is basic sequential
composition.

For 1956 independent Gaussian measurements it is the wrong theorem.  Each
Gaussian release at L2-sensitivity 1 satisfies rho = 1/(2 sigma^2)-zCDP, zCDP
composes **additively**, and the total converts back through
`eps = rho + 2 sqrt(rho ln(1/delta))`.  Solving for eps=10, delta=1e-5 gives
rho = 1.550, so 1956 marginals can each carry

    sigma = sqrt(k / (2 rho)) = sqrt(1956 / 3.10) = 25.1

against the 1436 and 707 the code uses.  **The same formal (ε=10, δ=1e-5)
guarantee permits 28-57x less noise.**  The scaling is the whole story: sigma
grows as sqrt(k) under zCDP and as k under basic composition, so the penalty
compounds precisely because this generator measures so many marginals.

How bad is 1436 in context?  A one-way marginal over 4 bins on 871 rows has
cell counts near 218.  The injected noise is **7x the signal**, and its standard
deviation is 1.6x the size of the entire training set.  Under zCDP it would be
0.12x the signal.

Three consequences:

1. **The DP-PGM column is uninformative as it stands.**  Both the fidelity
   numbers (macro-F1 0.185 against a real ceiling of 0.811 even after the
   gene-order fix, discriminator AUC 1.000) and the attack numbers (all four
   attacks at 0.50) are measurements of noise, not of privacy.  MAMA-MIA reading
   chance against marginals that are 7x noise is the expected result and says
   nothing about MAMA-MIA.

2. **This is very likely last year's CAMDA winner's accounting too**, since
   `joint_mode=True` exists to reproduce that structure.  If so it is a finding
   about the challenge baseline, not just about our fork.

3. **It is not our call to change.**  Privacy accounting is the one thing in
   this repo where a plausible-looking edit can silently invalidate the
   guarantee being claimed.  The change should be an explicit
   `composition="basic" | "zcdp"` option with both arms reported, reviewed by
   Steven before anything built on it goes in the paper -- not a silent default
   flip.

### Question 7 -- ANSWERED 2026-09-20: re-accounting done

Steven approved the switch to zCDP and asked me to own the accounting.
Implemented as `composition="zcdp"`, now the default, with `composition="basic"`
retained so every DP-PGM result recorded before 2026-09-20 still reproduces.

**What the implementation does.**  A Gaussian release with L2 sensitivity 1 and
noise scale sigma is rho-zCDP for rho = 1/(2 sigma^2); k such releases cost the
sum of their rho.  Inverting that at a fixed budget gives

    sigma_order = sqrt(k_order / (2 * weight_order * rho_total))

so sigma grows as sqrt(k) rather than k.  `budget_weights` now splits rho
instead of epsilon.  `rho_total` comes from OpenDP's numerically-inverted
zCDP-to-approxDP conversion -- the same `cdp_rho` snsynth uses for MST and AIM
-- falling back to the closed form eps = rho + 2 sqrt(rho ln(1/delta)) (Bun &
Steinke 2016, Prop. 1.3) if OpenDP is missing.  The fallback returns a *smaller*
rho, hence more noise, so it can only ever be conservative.

**Why we should trust it.**  It is not a derivation of ours; it is McKenna's,
transcribed.  MST writes the identical rule as a weight vector normalised by its
L2 norm (`weights / np.linalg.norm(weights)`, then `sigma / weight` per clique),
which for k equal weights is exactly sigma * sqrt(k).
`tests/test_composition.py` in the generator repo checks our sigma against that
formulation directly, at k = 1, 10, 978 and 1956, and asserts that the rho
actually accumulated over the measurement loop equals rho_total to 1e-9.  The
fitter raises if it ever exceeds it.  16 tests.

**The effect at our configuration** (BRCA, 978 genes, joint mode, 1956
marginals, eps=10, delta=1e-5):

| | 1-way sigma | 2-way sigma |
|---|---|---|
| basic (as shipped) | 1435.8 | 707.2 |
| zCDP | 28.8 | 20.2 |
| reduction | 49.8x | 35.0x |

Steven's own observation is the right way to frame this in the paper: basic
composition is tolerable below roughly 50 measurements, which is the regime
tabular DP synthesisers are usually demonstrated in.  It is genomic feature
counts -- hundreds to thousands of marginals -- that turn a constant-factor
looseness into a 50x one.  That is a transferable finding about applying tabular
DP-SDG methods to omics, not a bug report about one repo.

---

## 5. What the re-accounting does NOT fix

Switching composition makes the *measurement* budget correct.  It does not make
the reported epsilon an end-to-end guarantee, and the paper must not imply that
it does.  Two steps read the private training data and spend nothing:

1. **Marginal selection** (`marginal_selection.select`) ranks genes by the
   variance of the private data and gene pairs by its Spearman correlations.
   Which genes appear in the released model is therefore a deterministic
   function of the private data.  MST spends a full **rho/3** -- a third of its
   entire budget -- on exactly this step, via the exponential mechanism.
2. **Discretisation** (`discretization.fit`) sets bin edges with
   `np.percentile(col, ...)` on the private data, and `inverse_transform`
   releases values interpolated between those edges.  This is structurally the
   same leak as finding 1 in `results/FINDINGS.md`, where NoisyDiffusion's
   `QuantileTransformer` reproduced its training set's per-gene empirical
   support and MAMA-MIA read it off at AUC 0.9996.

Both are fixable -- selection via an exponential mechanism against a rho slice,
discretisation via edges from the auxiliary/public data rather than the training
split -- and neither is fixed today.  Until then the honest statement is that
epsilon covers the noisy marginals only.

Note also that stratified mode builds one fitter per class, each spending the
full epsilon.  That is legitimate parallel composition, since the per-class
subsets are disjoint and rho-zCDP composes in parallel over them -- but it is
only legitimate because the label partitions the data, and the class sizes
themselves are released unprivatised.

---

## 6. Design consultation: focal points, simulation, aggregation

Opened 2026-09-20 at Steven's request. Below is the analysis I can do without
him, a recommendation for each of the three, and the questions where the answer
depends on what MAMA-MIA actually does rather than on what the statistics say.

**The most important number first.** For a candidate *x* and a marginal cell
*c*, the member's own contribution to that cell is exactly **+1 count**. The
noise standing between the adversary and that +1 has two layers: the Gaussian
DP noise (σ, in absolute counts) and the multinomial sampling noise from drawing
`n_syn` synthetic rows. So the per-cell SNR is

    SNR_cell  =  1 / sqrt( σ² + n_syn·p_c·(1 − p_c) )

and, over *k* roughly independent cells, `SNR_total ≈ √k · SNR_cell`.
Pretending for a moment that PGM inference is lossless and the adversary reads
cell counts optimally, this gives a **ceiling** on any marginals-based attack:

| arm | σ (1-way) | total noise/cell | SNR/cell | ×√978 | **AUC ceiling** |
|---|---|---|---|---|---|
| basic | 1435.8 | 1435.9 | 0.0007 | 0.022 | **0.514** |
| zCDP | 28.8 | 31.5 | 0.0317 | 0.992 | **0.877** |

This retrospectively explains the flat DP-PGM row completely. Under the old
accounting **no attack could have exceeded AUC 0.514** — MAMA-MIA measuring 0.50
was not a weak attack, it was a saturated ceiling. Under zCDP there is now real
headroom. The 0.877 is an upper bound, not a prediction: PGM inference and
dithered decoding only destroy information.

### 6.1 Focal-point selection — the obvious criterion does not work here

The natural criterion is rarity: a target sitting in a rare cell contributes a
larger *fraction* of that cell, so the cell should be more diagnostic. That is
the standard "outliers are more vulnerable" result, and I assume it is close to
what focal-point selection is for.

**It is largely neutralised here, for two separate reasons.**

*First, equal-frequency binning erases rarity at order 1 by construction.* Both
the generator and the attack use quantile bin edges, so every one-way cell has
p = 1/n_bins exactly. There are no rare one-way cells. Whatever focal-point
selection does, it cannot key on one-way rarity.

*Second, DP noise is absolute, so it flattens what rarity is left.* Rarity
reduces only the sampling term, and under zCDP the DP term already dominates it.
For BRCA gene×label cells (σ = 20.2):

| cell | p | expected count | sampling SD | total noise |
|---|---|---|---|---|
| rarest class (n=32) | 0.0092 | 8.0 | 2.82 | **20.42** |
| median class | 0.0431 | 37.5 | 5.99 | **21.10** |
| largest class (n=449) | 0.1289 | 112.2 | 9.89 | **22.52** |

Between the rarest and most common cell in the grid, total noise moves by **10%**.
Selecting the best decile of cells by rarity therefore buys ~10% in per-cell SNR
while discarding 90% of the cells, which costs √10 ≈ 3.2× in aggregate. **Rarity-
based focal-point selection is strictly worse than using everything**, under DP,
by roughly a factor of three.

That is a negative result I am fairly confident in, and it is specific to the DP
setting — against MST or PrivBayes without this much noise, or against a
generator that does not equal-frequency-bin, the calculus is completely
different.

**What I think focal points should key on instead**, in descending order of my
confidence:

1. **The non-negativity projection.** PGM projects noisy counts onto the
   marginal polytope: counts are clipped at 0 and forced consistent across
   overlapping cliques. Clipping is *nonlinear* and asymmetric — for a cell
   whose true count is near zero, negative noise is truncated and positive noise
   survives, so occupancy leaks through a channel the Gaussian noise does not
   cover. This is structurally the same mechanism as the NoisyDiffusion quantile
   leak in FINDINGS §1, which reached AUC 0.9996. Focal points would then be
   *near-empty cells the target occupies* — the opposite selection rule from
   rarity-for-its-own-sake, and one that exploits post-processing rather than
   fighting it.
2. **Clique multiplicity.** A gene appearing in both a one-way and a gene×label
   clique is measured twice and its inference-time estimate averages two noisy
   views; genes in more cliques are less noisy. In the shipped config every gene
   is in exactly 2, so this is flat — but it becomes a real axis as soon as
   `n_2way > 0`, where the selected gene×gene pairs give some genes much higher
   multiplicity than others.
3. **Budget weighting.** `budget_weights=(0.33, 0.67)` makes gene×label cells
   quieter (σ = 20.2) than one-way cells (σ = 28.8). Under Kerckhoffs the
   adversary knows this. It is not focal-point selection so much as weighting —
   see 6.3.

> **Q7. What does focal-point selection actually select on in MAMA-MIA proper?**
> I have been assuming rare cells / distinctive value combinations. If it is
> something else — a learned criterion, a leave-one-out influence estimate, cells
> where shadow models disagree — the analysis above may be aiming at the wrong
> target entirely. This is the question I most need answered before building.

> **Q8.** Given the rarity channel is worth ~10% here, is focal-point selection
> something you would drop for the DP setting and keep only for the
> non-DP generators, or does it have a second purpose (variance reduction,
> tractability) that I am missing?

### 6.2 Simulation — we may not need shadow generators at all

Against a black-box generator you simulate because you cannot write down the
distribution of the statistic. **Here we can**, at least for two of the three
layers. Under Kerckhoffs the adversary knows ε, δ, `n_bins`, the marginal set
and the budget weights, and therefore knows σ *exactly, in closed form*. The
null and alternative for a single cell are

    H₀ (non-member):  count_c  ~  Binomial(n_syn, p_c)  +  N(0, σ²)
    H₁ (member):      the same, shifted by +1

The only layer not available analytically is PGM inference — the projection onto
the marginal polytope, which couples cells and clips at zero.

Three options, and I would like a steer:

- **(a) Fully analytic.** Score with the closed-form LLR and ignore PGM
  smoothing. Free, instant, and exactly calibrated for the parts it models.
  Risks being badly wrong precisely where the signal is (the clipping channel
  in 6.1.1 is invisible to it).
- **(b) Fully simulated.** Shadow DP-PGMs with known membership, as MeLoMIA
  does. Costs a PGM fit per shadow: ~16 min each, so 100 shadows is ~2 h on 16
  cores. Captures everything, including clipping.
- **(c) Hybrid — my recommendation.** Simulate the *generator* but not the
  *data*: take the fitted model's own marginals as ground truth, re-noise them
  under the known σ, re-run PGM inference, and resample. This needs no real
  training data at all, so it costs nothing in privacy and can be repeated
  cheaply once the expensive fit is cached. It calibrates the PGM + sampling +
  clipping stack while keeping the analytic handle on σ.

> **Q9.** Does MAMA-MIA's simulation step build shadow *generators*, or does it
> simulate the *marginal noise* around a single observed release? My reading of
> the DP setting says the second is both cheaper and better-calibrated here, but
> I do not want to reinvent a step you already designed differently.

> **Q10.** If shadows: should they be trained on disjoint halves of the auxiliary
> pool (matching our base-shadow convention), or on resamples of the *released
> synthetic* data (the synth-shadow convention we are already reporting)? This
> connects directly to the synth-shadow ablation you asked be in the paper.

### 6.3 Aggregation — three specific defects in what we ship

The current implementation (`mia/attacks/mamamia.py`) computes the plain
arithmetic mean of `p_syn/p_aux` over all 1957 marginals. Three problems, all
fixable, in descending order of how much I think they cost:

**(i) The arithmetic mean of ratios is the wrong functional.** A ratio is
bounded below by 0 and unbounded above, so the mean is dominated by the handful
of cells where `p_aux` happens to be small — precisely the cells whose ratio is
*least* reliable. The Neyman–Pearson statistic for "was this record in the set
that produced `p_syn`" is the **sum of log-ratios**, not the mean of ratios:

    score(x)  =  Σ_m  w_m · log( p_syn(x_m) / p_aux(x_m) )

This is a one-line change with a real justification behind it, and I expect it
to matter more than anything else on this list.

**(ii) No inverse-variance weighting.** One-way cells carry σ = 28.8, gene×label
cells σ = 20.2, and they are currently summed with equal weight. For independent
estimates the optimal combination weights by inverse variance:

    w_m  =  1 / ( σ_m²  +  n_syn · p_m · (1 − p_m) )

which the adversary can compute exactly, from public parameters. Using the
numbers above this up-weights the gene×label family by about 1.2× relative to
one-way — a modest gain, but free and principled.

**(iii) `p_aux` is estimated from a pool that contains the members.**
`X_aux = X_real` is the entire cohort, training half included. So the baseline
we divide by is contaminated with exactly the signal we are trying to detect,
which shrinks every ratio toward 1. The abstract did this deliberately (COMBINED's
provided reference set has no subtype labels, and the two-way marginals need
them), so changing it changes the threat model.

> **Q11.** Is the mean-of-ratios in our implementation a faithful simplification
> of MAMA-MIA, or did the original already aggregate in log space? If the latter,
> (i) is a straight bug on our side rather than a design choice.

> **Q12.** For `p_aux`: keep the contaminated-but-labelled full pool, or move to
> the clean reference set and lose the two-way family on COMBINED? A third option
> is to estimate `p_aux` from the full pool but *leave the candidate out*, which
> removes the candidate's own contribution to its own baseline — cheap, and it
> changes the null in the right direction.

### 6.4 What I will do while waiting

None of the above blocks the baseline. The immediate steps, which need no design
decisions:

1. Rebuild the DP-PGM targets with the corrected generator (both fixes) — the
   existing ones predate all three defects and are not worth attacking.
2. Run **the current, unmodified MAMA-MIA** against them, for both cohorts and
   all five splits. That is the honest before/after for the accounting fix, and
   it tells us how much of the 0.877 ceiling the attack as written already
   reaches.
3. Report that number before changing anything else, so every later change has a
   baseline to be measured against.
