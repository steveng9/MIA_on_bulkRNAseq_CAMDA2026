# The model zoo: five roles, one store

Every experiment in this repo is built out of the same two kinds of object -- a
generator fitted to some data, and a synthetic dataset drawn from one -- arranged
into a provenance DAG whose roots are real samples.  The zoo
(`mia/zoo/`) stores them content-addressed, so asking for the same thing twice
returns the same object instead of building a second copy.

The important design decision is that **a role is not a property of an
artifact**.  A fitted CVAE is just a fitted CVAE.  Whether it is a target, a
shadow, or a probe is a statement about the experiment using it, and lives in
`mia/zoo/roles.py`.  That separation is what makes reuse possible: one trial's
shadow can be another trial's target without copying, renaming, or rebuilding
anything.

## The five roles

| role | trained on | membership labels | exists under |
|---|---|---|---|
| `target` | real training split | unknown to the adversary | both threat models |
| `base_shadow` | a real split we control | known | black-box only |
| `synth_shadow` | a `base_shadow`'s synthetic output | inherited from its base shadow | black-box only |
| `internal_proxy` | a **held-out** `base_shadow`'s synthetic output | known | black-box only |
| `final_proxy` | the target's released synthetic data | unknown | black-box only |

**`target`** is the model we want to learn about. We only see what it released.

**`base_shadow`** exists for one reason: to emit a synthetic dataset whose
membership labels we know. It is never queried for losses itself.

**`synth_shadow`** is the point of synth-shadow modelling. The meta-classifier
will be applied to a model trained on *synthetic* data (the proxy), so the
models it learns from must be trained on synthetic data too. Training shadows on
real data instead leaves a domain gap between the loss distributions it learns
and the ones it is asked to score. Its membership labels come from its base
shadow's real split, recomputed from the stored closure rather than remembered.

**`internal_proxy`** is a proxy trained on a *held-out* base shadow's synthetic
data — a stand-in target whose labels we know. In the CAMDA competition this was
the only way to estimate performance at all, because the real target's labels
were secret. Running internally we can score the final proxy directly, so this
role has a narrower job now, but not an optional one: **it is the only honest
signal for choosing hyperparameters and ensemble weights.** Selecting those
against the final proxy's labels would be tuning on the evaluation set.

**`final_proxy`** is trained on the target's released synthetic data. It is what
the reported scores actually come from.

### Under a white-box threat

`base_shadow`, `synth_shadow` and `final_proxy` all disappear. With access to
the target model there is nothing to proxy, and since the target was trained on
*real* data, shadows must be trained on real data too, to match the inference
condition. White-box is therefore a strictly smaller pipeline: target plus a
real-data shadow population. `synth_shadow=False` in the MeLoMIA config gives
the real-data-shadow arm of that comparison, but not the whole threat model —
true white-box also drops the proxy.

## Reuse across experiments

Because artifacts are content-addressed by `(generator, params, source data,
seed)` and carry their training closure, they can be reused freely across
trials. A few patterns that fall out of this:

- **A shadow from one trial as a target in another.** Mechanically a base
  shadow's synthetic output is just a synthetic dataset with known provenance —
  exactly what a target is. Using it as a target gives a *fifth* generator
  column for free, and with labels we already trust.
- **One shadow stack, many attacks.** The MeLoMIA-ND and MeLoMIA-CVAE stacks are
  keyed on the probe family, so an experiment that changes only the
  meta-classifier, the timestep set, or `K` reuses the whole stack.
- **One target set, many attacks.** All four attacks score the same 20 targets
  per cohort, so the grid costs 20 generator fits, not 80.
- **Shadow count sweeps are free after the first.** `stack_tag()` deliberately
  excludes `K`, so a K=50 run reuses the K=30 stack's first 30 shadows.

### What is *not* allowed

Reuse across experiments is the point; reuse across **roles inside one
experiment** is contamination. `mia/zoo/roles.py` checks six things, and
`scripts/zoo.py verify` re-runs them over every recorded run:

1. No artifact holds two roles in one experiment.
2. Nothing the adversary builds descends from the target (the `final_proxy` is
   the deliberate exception — it is *supposed* to).
3. No `base_shadow` has the target's exact training closure, which would hand
   the meta-classifier the target's own membership labels.
4. The `final_proxy` really is descended from the target.
5. No `internal_proxy` shares ancestry with a `synth_shadow` — otherwise the
   meta-classifier has already seen the data it is being validated against.
6. Every artifact comes from the cohort the experiment says it does.

Check 5 is the subtle one, and it is the reason the internal-proxy role is
tracked at all rather than being approximated by cross-validation over the
shadow pool. Sample-disjoint CV holds out *samples*; it leaves every shadow
model in both the training and validation side of every fold, so it measures
"generalise to a new sample under a model I have seen" when deployment asks
"generalise to a new model on samples I have all seen".

## Layout

```
artifacts/zoo/
  index.jsonl              one JSON record per artifact, append-only
  fit/<ab>/<id>/           model.pt, spec.json, closure.txt
  sample/<ab>/<id>/        data.npz, spec.json, closure.txt
```

`closure.txt` — the real sample ids underlying the artifact — is the field
everything else rests on. Membership labels are recomputed from it on every use,
so a reused shadow can never carry stale labels, and contamination checks reduce
to comparing one hash.

`spec.json` is written last, so its presence marks a completed artifact and a
worker interrupted mid-build leaves nothing that looks finished.

## Commands

```bash
python scripts/zoo.py list --dataset BRCA --generator cvae
python scripts/zoo.py lineage <id>       # provenance back to real samples
python scripts/zoo.py reusable --dataset BRCA --role base_shadow
python scripts/zoo.py verify             # contamination check over all runs
python scripts/zoo.py du                 # disk by kind and generator
```
