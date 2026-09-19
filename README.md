# Membership Inference Attacks on Synthetic Bulk RNA-seq

Red-team attacks for the **CAMDA 2026 Health Privacy Challenge, Track I**
(University of Washington Tacoma, PPML-Huskies).

The question: does releasing synthetic bulk RNA-seq protect the donors whose
data trained the generator?  Four attacks, four generators, measured on two
TCGA cohorts.

> **Does Synthetic Bulk RNA-seq Data Protect Donors? Privacy Auditing through
> Membership Inference Attacks** — Jarrell, Filienko, Kim, Szebenyi, Pentyala,
> Golob, De Cock.  Abstract in `main_bulkRNA_CAMDA26_abstract.tex`.

---

## The attacks

| Attack | Target generator | Idea |
|---|---|---|
| **MahalaMIA** | MVN | Mahalanobis distance from the candidate to the synthetic distribution. The MVN generator's released data *is* a draw from a Gaussian fitted to the training split, so its covariance is a direct imprint of the members. |
| **MeLoMIA-ND** | NoisyDiffusion | Per-sample denoising-error trajectory across diffusion timesteps, read from synth-shadow models, classified by a tuned ensemble. |
| **MeLoMIA-CVAE** | CVAE | Same machinery, with the diffusion timestep sweep replaced by a sweep over posterior temperature, plus the per-dimension KL of the encoded posterior. |
| **MAMA-MIA** | DP-PGM | Domain ratios `p_syn(x_m) / p_aux(x_m)` over the low-order marginals DP-PGM actually releases. |

Every attack runs against every generator.  The diagonal is each attack against
what it was designed for; the off-diagonal measures how much of its power came
from knowing the generative mechanism.

### Synth-shadow modelling

The load-bearing idea behind both MeLoMIA variants.  The adversary never sees
the target model — only its released synthetic data — so at inference it can
only train a *proxy* on that synthetic data.  Shadow models trained on real data
therefore produce loss distributions from a different domain than the proxy
does, and a meta-classifier fitted to one transfers badly to the other.  Earlier
measurements in this repo put the gap at TPR@10%FPR 0.58 → 0.14.

Synth-shadow modelling inserts a layer:

```
real split k ──► base shadow (target-faithful) ──► internal synthetic data k
                                                            │
                                                            ▼
                                     synth-shadow k (sharp) ──► loss features
```

Real data enters only through the base shadows.  Everything features are read
from — synth-shadows in training, the proxy at inference — has seen only
synthetic data.  A sample's membership label for shadow *k* is inherited from
the base shadow: `x` is a member of shadow *k* iff `x` was in split *k*'s
training half, even though synth-shadow *k* never saw `x` at all.

### The five roles

Five distinct jobs, not five kinds of model — the same fitted generator can hold
a different role in a different experiment, which is what makes the artifact
store reusable.

| role | trained on | labels known? | exists under |
|---|---|---|---|
| target | real training split | no | both threat models |
| base shadow | a real split we control | yes | black-box only |
| synth shadow | a base shadow's synthetic output | yes, inherited | black-box only |
| internal proxy | a *held-out* base shadow's synthetic output | yes | black-box only |
| final proxy | the target's released synthetic data | no | black-box only |

Under a white-box threat the middle three collapse: with access to the target
there is nothing to proxy, and since the target saw *real* data, shadows must be
trained on real data to match the inference condition.

The internal-proxy role deserves a note, because its job changed when we stopped
competing.  In the challenge it was the only way to estimate performance at all,
since the target's labels were secret.  Running internally we know our own
targets' labels and score the final proxy directly — so the internal proxy is no
longer a performance estimate, but it is still the only honest signal for
choosing hyperparameters and ensemble weights.  Selecting those against the
final proxy's labels would be tuning on the evaluation set.

The internal proxy is also not a separate kind of model.  A synth-shadow is a
probe fitted to synthetic data emitted by a generator fitted to a real split,
which is the final proxy's construction exactly, minus the secrecy of the split.
Holding a set of shadows out of the meta-classifier's training pool and scoring
them *is* the internal-proxy role, and rotating that hold-out over the pool is
model-disjoint cross-validation.  `MeLoMIA(internal_proxy_selection=True)` does
this; it is off by default while the current grid is in flight.

See `docs/FIVE_ROLES.md` for the audit of what the pipeline builds against what
the five roles require, and `docs/MODEL_ZOO.md` for the artifact store, the
reuse patterns, and the six contamination rules that are checked rather than
asserted.

---

## Layout

```
mia/
  paths.py           where everything lives; the only file to edit on a new machine
  datasets.py        TCGA cohort loaders, class labels, the 5 canonical splits
  zoo/               content-addressed model + synthetic-dataset store
    ids.py             canonical hashing; closure hashes over real sample ids
    registry.py        get-or-create for fits and samples, provenance, locking
    roles.py           the five roles and the contamination rules
  targets.py         building and loading the target synthetic datasets
  metrics.py         AUC / AUPR / TPR@FPR, shared by every attack
  runs.py            experiment record keeping (results/index.csv)
  experiment.py      YAML-driven runner
  preprocessing.py   scalers, persisted alongside model weights
  generators/        the four SDG methods: mvn, cvae, nd, pgm
  attacks/
    mahalamia.py     Mahalanobis distance attack
    mamamia.py       marginal domain-ratio attack
    melomia/
      attack.py      the synth-shadow pipeline
      backends.py    ND and CVAE halves: shadow construction + loss extraction
      features.py    slicing and summarising the loss grid
      meta.py        classifier zoo, Optuna search, ensembling

configs/experiments/ one YAML per experiment
scripts/             build_targets, run_experiment, make_tables
results/             tracked: per-run config, row-level scores, metrics
artifacts/           git-ignored: targets, shadow models, features, classifiers
legacy/mia_v1/       the pre-refactor pipeline, kept for reference
notes/               design notes and historical run logs
```

---

## Quick start

```bash
# 1. Build the 20 target synthetic datasets (4 generators x 5 splits)
python scripts/build_targets.py --dataset BRCA

# 2. Run the full grid
python scripts/run_experiment.py configs/experiments/grid_brca.yaml

# ...or just the cheap attacks
python scripts/run_experiment.py configs/experiments/grid_brca.yaml \
    --only mahalamia mamamia

# 3. Render the grid
python scripts/make_tables.py --dataset BRCA
python scripts/make_tables.py --dataset BRCA --format latex --out results/tables
```

Building a MeLoMIA shadow stack is the expensive step (hours on one GPU for
ND at K=30).  It is cached under `artifacts/attacks/`, keyed by the attack's
hyperparameters, and every stage is idempotent — an interrupted run resumes.

To prepare the stack without scoring anything:

```bash
python scripts/run_experiment.py configs/experiments/grid_brca.yaml \
    --only melomia_nd --prepare-only
```

---

## Experimental design

**Five canonical splits.**  All four generators are trained on the same five
80/20 partitions of each cohort, taken from the blue team's published splits
YAML.  A sample therefore has the same membership label in every cell of the
grid, so differences between cells are differences between attacks and
generators, not between partitions.  It also lets the ND column reuse the blue
team's already-published synthetic data rather than retraining it.

**Shadow splits are independent.**  MeLoMIA draws its own K partitions from a
separate seed offset.  The adversary does not know the target's partition, and
reusing it would leak.

**Grouping by sample.**  Each real sample contributes one row per shadow, with
that shadow's membership label.  Every split inside the meta-classifier —
Optuna folds, early-stopping holdouts, reported CV — is grouped by sample id, so
the classifier cannot win by recognising samples instead of membership.

**Results are append-only.**  One directory per run under `results/runs/`, with
the resolved config, row-level scores and metrics; one row per run in
`results/index.csv`.  Run ids are deterministic hashes of the configuration, so
re-running updates in place instead of accumulating near-duplicates.

---

## Environment

```bash
conda env create -f environment.yml && conda activate camda-mia
python -m pytest tests/ -q
```

Python 3.9 with CUDA.  `numpy` is pinned below 2 for the Private-PGM stack; the
rest of the versions are recorded for reproducibility rather than because they
are delicate.

Three external repos are read but not vendored, resolved through `mia/paths.py`
or the matching environment variable:

| Path | Env var | Used for |
|---|---|---|
| `~/data/CAMDA26` | `CAMDA_DATA` | challenge expression data and labels |
| `~/CAMDA25_NoisyDiffusion` | `CAMDA_ND_REPO` | canonical splits, published ND synthetic data |
| `~/private-pgm-rnaseq-camda2026` | `CAMDA_PGM_REPO` | the DP-PGM generator |

The NoisyDiffusion architecture itself is vendored in
`mia/generators/nd_model.py` so model code and weights cannot drift apart.

---

## Open work

See [TODO.md](TODO.md) — shadow-count scaling, sweep-axis ablations, PCA
preprocessing, vine copulas, white-box comparison, and the with/without
synth-shadow ablation that justifies the method.
