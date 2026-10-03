# Note: per-record calibration across shadow models (MeLoMIA improvement)

Origin: TimeDiff / MIDST audit (ansons_capstone/audit/compare.py), October 2026.

## 1. In one sentence

Before the meta-classifier sees a record's loss features, express each feature as
"how far is this model's value for this record from the record's own typical value
across the attacker's shadow models", in units of that record's spread:

    z[i, j] = (x_model[i, j] - mean_k x_k[i, j]) / std_k x_k[i, j]

Here i = record, j = feature (e.g. log mean loss at diffusion step t), and k runs over
the shadow models (white-box) or synth-shadow models (black-box).

## 2. Why it works

A record's raw loss mixes two things:

1. How hard the record is for any model trained on this kind of data (outliers, rare
   phenotypes, noisy samples). This is a property of the record, not of membership.
2. How much this particular model memorised it. This is the membership signal.

Across records, (1) varies far more than (2). So a classifier on raw losses mostly
learns "unusual records look like non-members", and hard members are missed.
Comparing each record with itself across many models cancels (1) and leaves (2).

The question changes from "is this loss low?" to "is this loss low *for this record*?"

## 3. Precedent and names

This is not a new idea. It is the core of the strongest standard membership attacks.

- **Difficulty calibration.** Watson, Guo, Cormode, Sablayrolles, "On the Importance of
  Difficulty Calibration in Membership Inference Attacks", ICLR 2022. They subtract each
  record's expected loss under reference models from the target's loss. This is the
  closest official name. Our version also divides by the record's spread.
- **Per-example thresholds.** Sablayrolles et al., "White-box vs Black-box: Bayes Optimal
  Strategies for Membership Inference", ICML 2019, derive the optimal attack as a
  per-sample threshold.
- **LiRA.** Carlini et al., "Membership Inference Attacks From First Principles", IEEE S&P
  2022. It fits a Gaussian to each record's loss across shadow models. The *offline*
  variant scores a record as a one-sided z-score against the models it was not in, which
  is essentially the formula above. The *online* variant also fits the "in" Gaussian and
  takes a likelihood ratio.
- **Reference-model attacks.** Ye et al., "Enhanced Membership Inference Attacks against
  Machine Learning Models", CCS 2022 (Attack R). Zarifzadeh et al., "Low-Cost
  High-Power Membership Inference Attacks" (RMIA), ICML 2024. Both are refinements of
  the same per-record comparison.
- **Synthetic-data MIAs.** DOMIAS (van Breugel et al., AISTATS 2023) calibrates the
  synthetic-data density at a record by a reference density, the same idea for density
  attacks. Stadler et al., "Synthetic Data – Anonymisation Groundhog Day", USENIX Security
  2022, train shadows per target record.

What we did differently is small: instead of turning the z-score into a single attack
score (as LiRA does), we z-score *every* feature of the MeLoMIA signature and give the
whole calibrated vector to the meta-classifier. I am not aware of a standard name for
that combination. "Per-record (difficulty) calibration of the meta-classifier input" is
an accurate description, and citing Watson et al. 2022 and Carlini et al. 2022 covers
the precedent.

## 4. Is it inside the threat model?

Yes, for both variants. It uses only what the attacker already has:

- the candidate records (which the attacker must hold to query anything), and
- shadow and synth-shadow models the attacker trained themselves, on splits it chose.

It never uses the target's true membership labels. Each record only needs to be passed
through every shadow model for inference; a shadow does not need to have trained on it.

## 5. Recipe

Inputs: F[k] of shape (n_records, n_features) for shadow k = 1..K, with labels L[k],
and the attack-time features F_t (the target in white-box, the proxy in black-box).
Work in log space (log of loss summaries) before z-scoring.

1. **Training rows (shadow k):** calibrate F[k] against the *other* K-1 shadows only.

       mu = mean over j != k of F[j];  sd = std over j != k of F[j]
       Z[k] = (F[k] - mu) / (sd + eps)

   Leave-one-out matters. If shadow k is included in its own reference, part of its
   membership signal is subtracted out, and its training rows no longer look like the
   attack-time rows.
2. **Attack rows:** calibrate F_t against all K shadows (the target or proxy is never one
   of them).
3. Train the meta-classifier on the pooled (Z[k], L[k]) and apply it to Z_t.
4. Choose the classifier by validation on held-out shadows, never by the target's AUC.

Code (numpy):

```python
def calibrate(F, others, eps=1e-6):
    """F: (n, d); others: (m, n, d) the same records in other models."""
    return (F - others.mean(0)) / (others.std(0) + eps)

Fs = np.stack(shadow_features)                       # (K, n, d)
Z_train = np.stack([calibrate(Fs[k], np.delete(Fs, k, 0)) for k in range(len(Fs))])
Z_attack = calibrate(F_attack, Fs)
```

## 6. Design choices and pitfalls

- **Which shadows go into the reference.** We use all of the other shadows. Each record
  was a member of about half of them, so the reference is a 50/50 mix. Alternatives:
  - OUT-only reference (LiRA offline): cleaner, but uses half the models per record.
  - Both IN and OUT references (LiRA online): strongest, but needs more shadows.

  We have only tried the all-shadows version so far.
- **Number of shadows.** The per-record std needs enough models to be stable. With 32
  shadows it was fine. Below about 16, consider a pooled std per feature (one spread
  shared by all records, LiRA's "global variance" trick) and keep only the per-record mean.
- **Same noise draws everywhere.** Use the same frozen noise draws (and diffusion steps)
  for every model, so the differences between models are not drowned by sampling noise.
- **Every candidate record goes through every model.** That is members, non-members,
  and the records the attack will score.
- **Raw features are dropped.** In our runs the calibrated features replace the raw ones.
  Feeding both to the classifier has not been tried yet.

## 7. How this differs from MeLoMIA's current `calibrate_against_reference`

`mia/attacks/melomia/features.py: calibrate_against_reference` standardises each sweep
point by the mean and std of a *known non-member reference population* (the
TCGA-COMBINED auxiliary set), averaged over samples and draws. That removes each model's
overall loss *scale*. It does not remove *record difficulty*: every record in a model
gets the same shift and scale, so a hard member still looks like a non-member.

Per-record calibration is a different axis: it normalises each record across models,
not each model across records. The two can be combined: model-level first (if a
reference set exists), then per-record across shadows. Per-record calibration also
needs no auxiliary data set, so it works on every cohort.

In MeLoMIA's cache, each `features/shadow_{k}.npz` already holds `losses` for all real
records under synth-shadow k. That is exactly the stack `Fs` above, so the change is
local to `_pooled` / feature loading (apply leave-one-out per shadow) and to the proxy's
features at attack time (calibrate against all shadows).

## 8. Results that motivated this (TimeDiff on PhysioNet 2012)

Setup: 11,816 records, 5,908 members (50/50), TimeDiff recipe target, 32 shadows with the
target's settings, 32 synth-shadows and a proxy (hidden 256, 1.4M steps). Features: 16
diffusion steps × 600 frozen noise draws, summaries per step. AUC on the target's true
members; the classifier was chosen by AUC on 4 held-out shadows.

| Classifier | White-box raw | White-box calibrated | Black-box raw | Black-box calibrated |
| --- | --- | --- | --- | --- |
| LightGBM | 0.549 | 0.628 | 0.501 | 0.521 |
| LightGBM, larger | 0.556 | 0.626 | 0.496 | 0.521 |
| Logistic regression | 0.526 | 0.617 | 0.507 | 0.518 |
| MLP | 0.534 | 0.623 | 0.503 | 0.515 |
| Rank average of all four | 0.547 | 0.626 | 0.502 | 0.521 |

- With calibration, white-box TPR at 1% FPR rose from 3.2% to 5.7%.
- Held-out-shadow AUCs tracked the target AUCs within 0.015, so the gain is not
  tuning to the target.
- Classifier choice barely matters; calibration is the whole gain.
- The black-box gain is smaller. Synth-shadows and the proxy see each record only
  through synthetic data, so less of the record's membership signal survives to be
  calibrated.

## 9. Status in this repo (2026-10-03)

Implemented and on by default: `MeLoMIA.per_record_calibration` (`mia/attacks/melomia/
attack.py`), helpers `per_record_train / per_record_reference / per_record_apply` in
`mia/attacks/melomia/features.py`, tests in `tests/test_core.py`.

Differences from the recipe above:

- **Model standardisation first.** Each model's (log-loss) features are standardised over
  the candidate records before the per-record z-score. Without it the proxy's overall
  loss scale, which differs from the synth-shadows' when the target is another generator
  family, shifts every calibrated row at once (BRCA MeLoMIA-ND on MVN: 0.825 without,
  0.929 with). It needs no reference set, so it replaces `calibrate_against_reference`.
- **Members are 80% of every shadow here** (the challenge's split), not 50%, so the
  all-shadows reference is an 80/20 mix. OUT-only would leave about K/5 models per record.
- **Few shadows are fine.** K = 5 already gives most of the gain (BRCA ND 0.824 -> 0.939;
  K = 30: 0.858 -> 0.969). A pooled spread was no better than the per-record one at any K.
- The reference (per-record mean and spread at each classifier's chosen feature slice) is
  saved next to the classifier as `record_reference.npz`.

Trial on cached features: `scripts/trial_per_record_calibration.py`,
`results/per_record_calibration/`. Results: `results/FINDINGS.md` section 10q.
