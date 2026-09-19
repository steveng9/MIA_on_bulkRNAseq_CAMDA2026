# The five model roles, and which ones the code actually builds

A MeLoMIA run instantiates models in five distinct *roles*.  The roles are not
properties of the artifacts — the same trained generator can serve different
roles in different experiments, which is what makes reuse possible (see
`MODEL_ZOO.md`).  This file says what each role is for, audits the current
pipeline against it, and resolves the "is the internal proxy the same as the
final proxy?" question.

## The roles

| | role | trained on | membership known? | code |
|---|---|---|---|---|
| a | **target** | a real 80% split | yes (we made the split) | `mia/targets.py: load_target` |
| b | **base shadow** | a real shadow split | yes (we made the split) | `attack.py: _ensure_internal_synth` |
| c | **synth-shadow** | b's *synthetic output* | inherited from b | `attack.py: _ensure_synth_shadows` |
| d | **internal proxy** | a *held-out* base shadow's synthetic output | inherited | — see below |
| e | **final proxy** | the target's released synthetic data | no (that is the question) | `attack.py: _ensure_proxy_features` |

Under the black-box threat model the adversary never touches (a).  It sees one
released synthetic dataset and trains (e) on it; every loss it reads comes from
(e).  (b) and (c) exist only to manufacture training data for the
meta-classifier under conditions that match inference.

**Why (c) rather than shadows on real data.**  If the meta-classifier learned
from models trained on real splits, it would be fitted to one loss distribution
and applied to another — (e) has only ever seen synthetic data.  Inserting (b)
so that (c) trains on synthetic data puts the training-time and inference-time
feature extractors in the same domain.  Membership labels survive the detour
because whatever (b) memorised about its members leaves a trace in the synthetic
data it emitted, and (c) re-learns that trace.  `configs/experiments/
ablation_synth_shadow.yaml` measures what this is worth.

**Under a white-box threat model both proxy layers collapse.**  With access to
the target model there is no (e), so there is no domain to match, so there is no
reason for (b) or (c) either — shadows train directly on real splits.  That is
the `synth_shadow: false` path, which is therefore not only an ablation but also
the white-box configuration (TODO item 7).

## Audit: the pipeline builds four of the five

(a), (b), (c) and (e) are all built and cached.  **(d) is not built at all.**

Its job is taken, today, by `MM.evaluate_grouped` — `StratifiedGroupKFold`
grouped by `sample_id` over the pooled shadow features.  That holds out
*samples*, so every one of the K shadow models appears on both sides of every
fold.  The number it reports answers "how well does this meta-classifier score
records it has not seen, using models it has seen?" — but at inference the
models are the new thing, not the records.  Every record the attack scores is a
record it saw during training, thirty times over, under thirty different
membership labels.

This does not contaminate the reported grid numbers.  `score()` runs the real
deployment path — features from (e), labels never consulted — and Optuna never
sees a target's labels.  What it does mean is that **model selection is
optimising the wrong generalisation axis**, and that the CV AUC printed next to
each classifier is not an estimate of deployment performance.

## (d) and (e): same artifact, different role

The useful observation is that the pipeline already trains models that sit in
exactly (e)'s structural position:

```
base shadow k   ->  internal synthetic k  ->  synth-shadow k     (b -> c)
target          ->  released synthetic    ->  final proxy        (a -> e)
```

A synth-shadow *is* an internal proxy.  It is a probe fitted to a synthetic
dataset emitted by a generator fitted to a real split — the same construction as
the final proxy, differing only in that we know the split.  The distinction
between (c) and (d) is therefore **not a distinction between artifacts.  It is a
role assignment**: shadows whose features go into the meta-classifier's training
pool are playing (c); shadows held out of that pool and used to score it are
playing (d).

So the answer to "can we treat (d) and (e) as the same now that we are not
submitting to a competition?":

- **For reporting, yes.** We hold the real membership labels for our own
  targets, so the honest headline number is (e) scored against the truth — which
  is what the grid already reports.
- **For model selection, no.** Choosing timesteps, noise budget, hyperparameters
  and ensemble weights against (e)'s labels is tuning on the evaluation set.
  Selection needs (d).

(d)'s role therefore changes from *performance estimate* (its competition job,
where (e)'s labels were genuinely hidden) to *model-selection set*.

## What this implies for the code

Rotating the (d) role over the shadow pool is exactly model-disjoint
cross-validation: for each fold, hold out a set of shadow models entirely, train
the meta-classifier on the rest, and score the held-out models' features.  Each
fold's held-out shadows are internal proxies for that fold.

`meta.py: block_folds` does this, splitting on samples *and* on shadows at once
so that a fold's validation rows share neither a record nor a model with its
training rows.  `tests/test_validation.py` pins the difference: a signal that
lives in a per-model offset scores ~1.0 under sample-grouped CV and at chance
under block CV.

Enable it with `internal_proxy_selection: true` on a MeLoMIA attack config; the
meta-cache tag changes so nothing cached under the old selection is reused.  See
TODO item 11 for the re-run that compares the two.
