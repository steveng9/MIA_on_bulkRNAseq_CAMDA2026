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
which means per-class marginals — arguably a better target, since a member
influences only its own class's marginals and those are estimated from far fewer
rows.

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
