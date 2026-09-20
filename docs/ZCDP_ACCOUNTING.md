# The zCDP accounting, from scratch

*Written for someone comfortable with ε-DP, (ε, δ)-DP, the Gaussian mechanism
and basic composition, who has bounced off Rényi DP before.*

The good news: **you do not need Rényi DP to follow any of this.** The whole
argument needs exactly one fact beyond high-school algebra — *the sum of
independent normal random variables is normal* — and one standard tail bound.
Everything below is derived, not cited.

---

## 1. What we are claiming

> **Claim.** Fix ε, δ. Let ρ = ρ(ε, δ) be the budget from §6. Split it over
> marginal orders by weights wₒ summing to 1. Within order *o*, measure each of
> its kₒ marginals with independent Gaussian noise of scale
>
>     σₒ = Δ₂ · sqrt( kₒ / (2 · wₒ · ρ) )
>
> where Δ₂ is the L2 sensitivity of one marginal (§7). Then the entire
> collection of released noisy marginals is (ε, δ)-differentially private.

The thing to notice, and the whole point of the exercise, is that **σ grows like
√k, not like k**. At 1956 marginals that is the difference between σ = 1436 and
σ = 29 on counts that sum to 871.

---

## 2. Privacy loss as a random variable

Start with the definition you already know. A mechanism M is (ε, δ)-DP if for
neighbouring D, D′ and every output set S,

    P[M(D) ∈ S]  ≤  e^ε · P[M(D′) ∈ S]  +  δ

There is a second, equivalent way to look at this that makes composition easy.
Run the mechanism on D and get output *o*. Ask: **how much more likely was this
output under D than under D′?** Take the log:

    Z  =  ln ( P[M(D) = o] / P[M(D′) = o] )        with  o ~ M(D)

`Z` is the **privacy loss random variable**. It is random because *o* is random.

- `Z = 0` → this output was exactly as likely either way; it tells an attacker
  nothing.
- `Z = 3` → this output was e³ ≈ 20× more likely under D. Damning.
- `Z < 0` → the output actually pointed *away* from D. That happens too.

Pure ε-DP is the statement **Z ≤ ε always, with probability 1.** It is a
worst-case bound on this random variable. Approximate (ε, δ)-DP is the softer
statement that `Z ≤ ε` except on an event of probability about δ.

This reframing is the key. Once privacy loss is a *random variable*, composing
mechanisms becomes a question about *sums of random variables* — and we know a
great deal about those.

---

## 3. The privacy loss of the Gaussian mechanism is exactly normal

Here is the two-line computation that makes everything else fall out. It is
ordinary algebra; nothing is hidden.

Setup: a query *q* with `q(D) − q(D′) = Δ` (one dimension for now). The
mechanism releases `o = q(D) + σ·g` where `g ~ N(0, 1)`.

The two densities at the observed *o* are proportional to

    P[M(D)  = o]  ∝  exp( −(o − q(D) )² / (2σ²) )
    P[M(D′) = o]  ∝  exp( −(o − q(D′))² / (2σ²) )

The normalising constants are identical, so they cancel in the ratio:

    Z  =  [ −(o − q(D))²  +  (o − q(D′))² ] / (2σ²)

Now substitute `o − q(D) = σg` and therefore `o − q(D′) = σg + Δ`:

    Z  =  [ −(σg)² + (σg + Δ)² ] / (2σ²)
       =  [ σ²g² − σ²g² + 2σgΔ + Δ² ] / (2σ²)      ← the g² terms cancel
       =  Δ²/(2σ²)  +  g · Δ/σ

**Define ρ = Δ² / (2σ²).** Then `Δ/σ = √(2ρ)`, and since `g ~ N(0,1)`:

    Z  =  ρ  +  g·√(2ρ)       i.e.       Z  ~  N( ρ,  2ρ )

> **The punchline.** The privacy loss of a Gaussian mechanism is itself a
> Gaussian, with **mean ρ and variance 2ρ** — one single number controls both.
> That number, ρ, is the zCDP budget. "ρ-zCDP" is, for our purposes, exactly the
> statement *"the privacy loss is N(ρ, 2ρ) or better."*

(For a vector-valued query the same result holds with Δ the **L2 norm** of the
difference: only the noise component along the difference direction matters, and
the directions orthogonal to it contribute nothing to the likelihood ratio. This
is why L2 sensitivity is the right notion for Gaussian noise, and L1 for
Laplace.)

Two sanity checks on `Z ~ N(ρ, 2ρ)`:
- More noise (bigger σ) → smaller ρ → loss concentrated near 0. ✓
- The mean is positive. On average the output *does* point at the true dataset —
  it must, or the mechanism would be useless. Privacy comes from the spread
  being large relative to that mean: sd/mean = √(2ρ)/ρ = √(2/ρ), which blows up
  as ρ → 0. ✓

---

## 4. Composition is now a one-liner

Run k mechanisms on the same data, with independent noise. The attacker sees all
k outputs. Because the noise draws are independent, the joint likelihood ratio
factorises, and **logs turn products into sums**:

    Z_total  =  Z₁ + Z₂ + ... + Z_k

Each `Zᵢ ~ N(ρᵢ, 2ρᵢ)`, independent. And the sum of independent normals is
normal, with means and variances each adding:

    Z_total  ~  N( Σρᵢ ,  2·Σρᵢ )  =  N( ρ_total, 2·ρ_total )

> **ρ simply adds, and the answer is again of the exact same form.** There is no
> inequality here, no approximation, no slack. The family of "privacy loss is
> N(ρ, 2ρ)" statements is *closed under composition*.

This is the entire reason zCDP exists. (ε, δ)-DP is *not* closed under
composition in this way — composing two (ε, δ) guarantees gives you something
that is not naturally described by a single (ε′, δ′) pair, so you are forced to
bound it, and every bound loses something.

### Why basic composition is so lossy — the intuition you already had

Basic composition says: `Zᵢ ≤ εᵢ` for each *i*, therefore `Z_total ≤ Σεᵢ`. It
adds up the **worst cases**.

But the worst case of a sum of independent random variables is a wild
over-estimate of the sum's actual behaviour. If you flip 1956 fair coins, the
worst case is 1956 heads; the truth is 978 ± 22. Sums of independent things
**concentrate**: the mean grows like k, but the fluctuation only like √k.

That is exactly the gap. Writing it out, with each of k releases given an equal
share:

| | per-release budget | σ needed | growth |
|---|---|---|---|
| basic | ε/k | `Δ·√(2 ln(1.25/δ)) · k / ε` | **k** |
| zCDP | ρ/k | `Δ·√(k / (2ρ))` | **√k** |

Your instinct about this was right: at k = 5 the ratio is about 2×, which nobody
notices. At k = 1956 it is 40×. Below ~50 features, basic composition is merely
wasteful; at genomic scale it destroys the signal. The penalty is not a constant
factor — it is asymptotic, and it is `√k`.

```
σ_basic / σ_zcdp      k=5    k=20    k=50   k=100   k=978   k=1956
                      2.0×    4.1×    6.5×    9.1×   28.6×    40.5×
```

---

## 5. What ρ-zCDP means in general

For the Gaussian mechanism we derived `Z ~ N(ρ, 2ρ)` *exactly*. The general
definition has to cover mechanisms whose loss is not exactly normal, so it says
"no worse than N(ρ, 2ρ)" in a way that survives composition. The formal version
bounds the moment generating function,

    E[ e^{(α−1)Z} ]  ≤  e^{(α−1)·αρ}    for all α > 1

and *that* is where Rényi divergence enters — the left side is exactly
`exp((α−1)·D_α(M(D) ‖ M(D′)))`, so the condition reads `D_α ≤ αρ` for all α.

**You can skip that.** For this codebase every mechanism is Gaussian, the loss
is exactly `N(ρ, 2ρ)`, and MGF-of-a-normal is a standard computation that
reproduces the bound with equality. The α is a free parameter you optimise over
in the general theory; for Gaussians the optimisation has a closed-form answer
and is already baked into §6. Rényi DP and zCDP are two bookkeeping systems for
the same underlying object — the distribution of Z — and zCDP is the one that
collapses to a single number when your mechanisms are Gaussian.

---

## 6. Converting ρ back to (ε, δ)

We must report (ε, δ). So: given `Z_total ~ N(ρ, 2ρ)`, what (ε, δ) does that
imply? Recall (ε, δ)-DP asks for `Z ≤ ε` except with probability ~δ. So we need
the **upper tail of a normal**:

    δ  =  P[ Z > ε ]  =  P[ N(ρ, 2ρ) > ε ]  =  P[ g > (ε − ρ)/√(2ρ) ]

Apply the standard Gaussian tail bound `P[g > t] ≤ e^{−t²/2}` with
`t = (ε − ρ)/√(2ρ)`, so that `t²/2 = (ε − ρ)²/(4ρ)`:

    δ  ≤  exp( −(ε − ρ)² / (4ρ) )

Set that equal to δ and solve for ε. Taking logs, `(ε − ρ)² = 4ρ·ln(1/δ)`, so

    ε  =  ρ  +  2·sqrt( ρ · ln(1/δ) )          (Bun & Steinke 2016, Prop. 1.3)

We need the other direction — given (ε, δ), find the largest affordable ρ.
Substitute `t = √ρ`, giving the quadratic `t² + 2t√L − ε = 0` with `L = ln(1/δ)`.
The positive root is

    √ρ  =  −√L  +  √(L + ε)

which is exactly the fallback in `rho_from_eps_delta`. At ε = 10, δ = 1e-5 it
gives **ρ = 1.550**.

**We do not actually use that number.** The bound `P[g>t] ≤ e^{−t²/2}` is loose,
and a tighter conversion (accounting for the fact that losses far below ε are
"refunded") lets you buy more ρ for the same ε. OpenDP's `cdp_rho` inverts that
tighter conversion numerically, and gives **ρ = 1.783** at the same (ε, δ) —
about 15% more budget, worth ~7% in σ. `snsynth`, and therefore MST and AIM, use
that function, and so do we.

The closed form is kept as a fallback. Note the direction of the inequality:
1.550 < 1.783, so the fallback buys *less* budget and produces *more* noise. A
fallback that errs toward more noise can never weaken the guarantee, and
`tests/test_composition.py` pins that ordering so it cannot silently flip.

---

## 7. Sensitivity: what "one person's data" means

Everything above is in terms of Δ₂, the **L2 sensitivity** — how much the
released vector can move when one person's data changes. This depends on a
modelling choice that is easy to leave implicit and expensive to get wrong.

A marginal query returns a **count vector**: for genes A, B discretised to 4 bins
each, the 2-way marginal is 16 numbers counting how many patients fall in each
(binA, binB) cell.

**Add/remove neighbours (unbounded DP).** D′ is D with one patient inserted or
deleted. That patient sat in exactly one cell, so exactly one entry changes by 1:

    Δ₂ = ‖(0,…,1,…,0)‖₂ = 1

**Replace neighbours (bounded DP).** D′ is D with one patient's record changed.
They leave one cell and arrive in another, so one entry drops by 1 and another
rises by 1:

    Δ₂ = ‖(0,…,−1,…,1,…,0)‖₂ = √2

Since ρ = Δ₂²/(2σ²), **replace costs exactly 2× the ρ of add/remove**, or
equivalently √2 times the noise for the same budget. MST and AIM assume
add/remove, and so do we by default.

### The defect this uncovered

The two conventions differ in a second place, and mixing them is not
conservative — it is simply invalid:

- Under **add/remove**, the row count *n* is itself sensitive: D and D′ have
  different sizes. Releasing the exact *n* distinguishes them with probability 1.
  That is an infinite privacy loss, and no σ repairs it.
- Under **replace**, *n* is identical for all neighbours, so it is public and
  free to release — but then Δ₂ = √2 and every σ must grow accordingly.

Before 2026-09-20 this implementation used **Δ₂ = 1 noise** (an add/remove claim)
while passing **the exact row count** to `FactoredInference` (a bounded-DP
assumption). Neither guarantee held. Stratified mode had it twice over, since it
also sized each class's synthetic sample from the exact per-class counts, leaking
the class histogram exactly.

The fix costs nothing. `mbi` will estimate the total itself from measurements
already paid for — it takes the minimum-variance unbiased combination of the
noisy 1-way marginals, each of which sums to *n*. With 978 marginals at σ = 28.8,
the estimate of n = 871 has a standard deviation of **1.84, or 0.21%**. We pass
`total=None` and let it do that; stratified mode allocates from the submodels'
noisy totals. This is what MST does, and it is why MST never passes a total.

### Parallel composition, and why it is also convention-dependent

Stratified mode fits one PGM per class and gives each the **full** ρ. That is
parallel composition: the classes partition the data, so one added-or-removed
patient touches exactly one submodel, and the budgets do not add.

That argument **requires add/remove**. Under replace, a changed patient can move
*between* classes, touching two submodels, and their budgets would have to
compose sequentially instead — costing 2ρ, not ρ. A third reason the default is
add/remove.

---

## 8. What none of this covers

The claim in §1 is about **the noisy marginals**. Three other things read the
private data, and the reported ε does not cover any of them:

1. **Gene selection** (`marginal_selection.select`) ranks genes by the variance
   of the private data and pairs them by its Spearman correlations. Which 978 of
   ~20 000 genes appear in the output is itself a data-dependent release. MST
   spends a full ρ/3 here, via the exponential mechanism. We spend 0.
2. **Discretisation** (`discretization.fit`) takes bin edges from `np.percentile`
   of the private data, and `inverse_transform` dithers between them — so the
   released values interpolate empirical quantiles of the training set. This is
   structurally the same leak we documented for NoisyDiffusion in FINDINGS §1.
3. **`n_samples`**, the number of synthetic rows requested, is chosen by the
   caller and in our experiments is set to the true training-set size.

So: **ε is a guarantee for the measurement step, not for the pipeline.** That
should be stated plainly in the paper. It is also, conveniently, the most
promising surface for the attack — selection and binning carry membership signal
that the noise never touches.

### And a third defect, in the legacy arm

`composition="basic"` uses `δ = 1e-5` for **each** of the 1956 marginals without
dividing it by k. Basic composition adds δ as well as ε, so the σ it produces is
really a (10, **0.0196**)-DP guarantee, not (10, 1e-5) — δ under-reported by
1956×. A correct basic accounting would use δ/k and be a further **1.28×**
noisier.

We have deliberately *not* changed this: `basic` exists only to reproduce results
recorded before 2026-09-20, and changing it would defeat that purpose. But it
means any comparison against `basic` slightly **flatters** it — the honest zCDP
advantage is a little larger than the measured one. The paper should report
`basic` as "the original implementation," never as "basic composition correctly
applied."

---

## 9. Pseudocode

```
INPUT   epsilon, delta, cliques_by_order[1..4], weights w[1..4] (sum to 1),
        neighboring in {add_remove, replace}

# ---- sensitivity ----------------------------------------------------
Delta = 1            if neighboring == add_remove       # one cell moves by 1
        sqrt(2)      if neighboring == replace          # one out, one in

# ---- total budget ---------------------------------------------------
rho_total = cdp_rho(epsilon, delta)                     # OpenDP, tighter
            # fallback: (-sqrt(L) + sqrt(L + epsilon))^2,  L = ln(1/delta)

# ---- renormalise over orders that actually have cliques -------------
active = { o : cliques_by_order[o] nonempty AND w[o] > 0 }
W      = sum of w[o] for o in active                    # so no budget is wasted

# ---- measure --------------------------------------------------------
measurements = []
rho_spent    = 0
for o in active:
    k     = count(cliques_by_order[o])
    frac  = w[o] / W                                    # this order's share
    sigma = Delta * sqrt( k / (2 * frac * rho_total) )  # <-- the sqrt(k) rule

    for clique in cliques_by_order[o]:
        x = true_marginal_counts(data, clique)          # sums to n
        y = x + Normal(0, sigma^2, size=len(x))
        measurements.append( (identity, y, sigma, clique) )
        rho_spent += Delta^2 / (2 * sigma^2)            # each release's cost

assert rho_spent <= rho_total * (1 + 1e-9)              # accounting must close

# ---- fit ------------------------------------------------------------
total = n            if neighboring == replace          # n is public
        None         if neighboring == add_remove       # n is SENSITIVE:
                                                        # let mbi estimate it
model = FactoredInference(domain).estimate(measurements, total=total)
```

The `assert` is not decoration. It is the one line that makes a wrong edit to the
σ formula fail loudly instead of silently shipping a weaker guarantee, and it is
what `test_both_neighboring_relations_spend_exactly_rho_total` exercises.

Note that the budget is spent *entirely* on the marginals we actually measure.
The renormalisation over `active` matters in stratified mode, where an order can
be requested but end up with no cliques; without it that order's share would be
silently thrown away.

---

## 10. Worked example — the shipped configuration

BRCA, n = 871 patients, 978 genes, joint mode, ε = 10, δ = 1e-5,
weights 0.33 / 0.67 over 1-way / 2-way, add/remove neighbours (Δ₂ = 1).

```
rho_total = cdp_rho(10, 1e-5)                     = 1.7827

1-way:  k = 978,  frac = 0.33,  rho_order = 0.33 * 1.7827 = 0.5883
        sigma = 1 * sqrt(978 / (2 * 0.5883))           = 28.83
2-way:  k = 978,  frac = 0.67,  rho_order = 0.67 * 1.7827 = 1.1944
        sigma = 1 * sqrt(978 / (2 * 1.1944))           = 20.23

check:  978 * 1/(2*28.83^2) + 978 * 1/(2*20.23^2)
     =  0.5883 + 1.1944  =  1.7827  =  rho_total          ✓ exactly spent
```

Against the old `basic` arm at the same nominal (ε, δ): σ = 1435.8 and 707.2 —
**49.8× and 35.0×** more noise, on counts that sum to 871. Noise of size 1436 on
a count that cannot exceed 871 means the measurement carried essentially no
information; the "private" model was fitting almost pure noise. That, and not
any property of differential privacy, is why the DP-PGM row of the attack grid
was flat.

And if we had chosen `replace` instead, every σ above would be √2 larger
(40.77 and 28.62) and stratified mode would lose parallel composition as well.

---

## 11. Auditing it empirically

Everything above is a *prediction about the code*, so it is worth measuring
rather than trusting. §3 says the privacy loss of the entire release is exactly
`Z ~ N(ρ, 2ρ)`. That is directly testable:

1. Build the real measurements on D, and on a **worst-case neighbour** D′.
2. Draw actual releases from the mechanism run on D.
3. Score each release under both D and D′ and take the log ratio — that *is* Z.
4. Check its mean against ρ_spent and its variance against 2·ρ_spent.

This is a much stronger check than re-deriving the σ formula, because it uses
the *actual* marginal difference between D and D′ rather than an assumed
sensitivity. If Δ₂ were wrong, the measured mean would miss ρ_spent, and no
amount of internally-consistent arithmetic elsewhere would catch it.

Measured over 20 000 releases, 6 genes × 4 bins, 400 rows, ε = 3, δ = 1e-5,
6 one-way + 5 two-way marginals:

| neighbour | σ (1-way) | ρ budgeted | Z mean (measured) | Z variance | vs 2ρ |
|---|---|---|---|---|---|
| add/remove | 5.783 | 0.22425 | **0.22407 ± 0.00473** | 0.44814 | 0.9992 |
| replace | 8.179 | 0.22425 | **0.21996 ± 0.00475** | 0.45122 | 1.0061 |

Both land within one standard error of the budget, and both variances match
2ρ to well under 1%. The mechanism spends exactly what it claims, under either
convention, and the √2 in the `replace` σ is doing precisely the work it should.

**One trap worth recording.** The first version of this audit changed a single
gene's value to build the `replace` neighbour, and measured a loss of 0.040
against a budget of 0.224 — an apparent 5× over-charge. The implementation was
fine; the *test* was. Changing one gene only perturbs the cliques containing
that gene (2 of 11 here), so it is nowhere near the worst case. A replace
neighbour must move **every** attribute for every clique to see the full √2.
An audit that is not worst-case will happily report that your mechanism is
conservative when it is not.

`test_measured_privacy_loss_matches_the_predicted_normal` runs a 6000-trial
version of this on each convention.

---

## 12. Where to check this

| Claim | Checked by |
|---|---|
| ρ ↔ (ε, δ) round-trips | `test_closed_form_roundtrip` |
| OpenDP ρ ≥ closed-form ρ (fallback is conservative) | `test_opendp_rho_at_least_closed_form` |
| σ matches MST's `weights/‖weights‖` formulation | `test_sigma_matches_mst_formulation` (k = 1, 10, 978, 1956, rel 1e-12) |
| σ grows as √k, not k | `test_sigma_grows_as_sqrt_k` |
| replace costs exactly √2 the noise | `test_replace_costs_sqrt2_more_noise_than_add_remove` |
| ρ spent = ρ budgeted, both conventions | `test_both_neighboring_relations_spend_exactly_rho_total` |
| add/remove never releases exact n | `test_add_remove_never_releases_the_exact_row_count` |
| **measured** privacy loss = N(ρ, 2ρ), both conventions | `test_measured_privacy_loss_matches_the_predicted_normal` |

MST's own code, for comparison
(`snsynth/mst/mst.py`): `rho = cdp_rho(eps, delta)`, `sigma = sqrt(3/(2*rho))`,
then `weights / norm(weights)` and `sigma/wgt` per clique. For k equal weights
`wgt = 1/√k`, so `sigma/wgt = sigma·√k` — identical to our
`sqrt(k / (2·rho_order))`. It passes no `total`.
