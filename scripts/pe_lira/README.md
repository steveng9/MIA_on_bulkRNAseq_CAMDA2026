# Joint membership attack on Private Evolution (PE)

Membership inference against **Private Evolution** synthetic data (tabular PE-Tabular,
Aug-PE text, DPSDA images).  In every PE round, each private record votes for its nearest
synthetic candidate; the curator adds Gaussian noise to the vote counts and releases them
(the "counts" view of federated PE).  Both attacks below read **only these released, noised
counts** (plus the candidates, the noise level sigma and a reference set Ref).  Neither
reads the clean histogram, a noise-free run, or the true class sizes.

## The original attack vs. the joint attack, simply

**Original (EVO-MIA, e.g. variant A4).** One target at a time.  For each round, find the
cell the target would vote for and ask: *"is this cell's released count higher than I would
expect if the target were not in the data?"*  The expected count comes from Ref.  Every
other vote in that cell is treated as anonymous background noise.  Sum the evidence over
rounds.

**Joint.** All targets at once.  Each released count is a sum of votes, and many of those
voters are other targets the attacker holds.  So write one equation per (round, cell):

    released count  ~  (number of member targets voting for this cell)
                     + (votes of private records outside the target pool)
                     + Gaussian noise

and solve for everyone's membership together.  A neighbour's vote is no longer anonymous
noise: it is explained by that neighbour's own estimated membership, from every round the
neighbour appears in.  Each target's score is its estimated membership probability.

Example: targets A, B, C, D share a cell whose count is about 3 (1.5 expected).  The original
attack gives all four the same "busy cell" evidence.  The joint attack sees that B and C sit
in near-empty cells in other rounds (probably not members) while D sits in busy cells again,
so A and D explain this count -- A gets evidence of its own.

The model of the counts is the same; the difference is how the *other* votes are handled:
marginalised into a Ref-based background (original) or estimated jointly (joint).

### When to use which

The joint attack needs the target pool to cover a large part of the private data set, and
moderate privacy:

| setting | use |
|---|---|
| eps ~ 10, pool covers most of the private set (e.g. full-population audits) | **joint** |
| eps ~ 1 | original (A4); the joint attack loses 0.02-0.10 AUC |
| pool covers a small part of the private set (e.g. 500 audit targets) | original (A4) |

## Algorithm (joint)

Per class, from the released counts only (`joint_sparse.py`):

1. **Who votes where.**  For every round, each target's nearest synthetic candidate (its
   cell), exactly as PE's records vote.  This gives a 0/1 matrix `A`: rows = (round, cell),
   columns = targets.
2. **Model.**  `y = A s + background + N(0, sigma^2)`, with `y` the released counts and
   `s_i = 1` if target i is a member.
3. **Unknown sizes, from the release.**  Class size `n` = mean of each round's released
   total.  Member share of the pool `s0`: try a grid of values, fit on half the rounds,
   keep the one that best predicts the other half's counts.  Background = the remaining
   `n - s0 * pool size` votes, spread over cells like Ref.
4. **Solve.**  Prior `s_i ~ N(s0, s0 (1 - s0))`; the posterior mean of `s` given all counts
   is a regularised least-squares problem, solved with conjugate gradients.
5. **Score.**  Each target's posterior mean.

## How to run

Prerequisites (paths as on the development machine):

* A read-only export of evo-mia's committed code, used for its transcript loaders, embedders
  and the original scorers:
  `mkdir -p artifacts/evo_mia_head && (cd ~/privacy/evo-mia && git archive <commit>) | tar -x -C artifacts/evo_mia_head`
* PE runs **with a histogram log** (every cell's noised count each round).  The published
  DPSDA_tab tabular runs keep only the winners' counts; generate logged runs with
  evo-mia's `generation/tabular/run_world.py` (DPSDA_tab `.venv`,
  `PYTHONPATH=~/privacy/DPSDA_tab`).
* Python: the `dp` conda env for tabular and text, DPSDA's `.venv` for images.

**Original attack (A4 and the other EVO-MIA variants), tabular testbed worlds**

```bash
python scripts/pe_lira/score_worlds.py --worlds '~/privacy/nfree_worlds/breast-cancer_eps10_w*' \
    --out results/pe_lira/breast-cancer_eps10
```

(evo-mia's own scorer, `run_worlds_eval.score_world`, without writing caches into the worlds.)

**Joint attack, tabular** (prints the joint attack and, with `--a4`, A4 on the same targets)

```bash
python scripts/pe_lira/joint_sparse.py --worlds '~/privacy/nfree_worlds/breast-cancer_eps10_w*' --a4
```

Pools of chosen coverage on one logged run:

```bash
python scripts/pe_lira/make_coverage_worlds.py --run artifacts/pe_worlds/adult_eps10_full \
    --train ~/privacy/nfree_data/adult_train.csv --test ~/privacy/nfree_data/adult_test.csv \
    --out artifacts/pe_worlds/adult_cov --sizes 500:3,8000:2,all:1
python scripts/pe_lira/joint_sparse.py --worlds 'artifacts/pe_worlds/adult_cov_m8000_r*' --a4
```

**Joint vs. original, text and images**: `joint_modality.py` runs evo-mia's own
`run_nfree_mia` driver (same audit set, embedder, Ref and transcript) and replaces only its
final scoring step, printing the joint attack next to A4 / A6 / B5 / C1.  Build a deploy copy
first (`artifacts/evo_deploy/<modality>/attacks` = evo-mia's `attacks/<modality>` +
`attacks/common`), and point `--out_json` / `--emb_cache_dir` inside this repository:

```bash
python scripts/pe_lira/joint_modality.py --deploy artifacts/evo_deploy/text --repo ~/privacy/aug-pe-baseline -- \
    --result_folder <run> --dataset openreview --member_file data/openreview/iclr23_reviews_train.csv \
    --nonmember_file data/openreview/iclr23_reviews_test.csv --reference_file data/openreview/iclr23_reviews_val.csv \
    --noise_multiplier 1.57 --count_threshold 0 --combine_divide_L 7 \
    --emb_cache_dir <cache in this repo> --out_json <json in this repo>
```

## Results (AUC, TPR at 1% FPR in brackets; same release and targets for both attacks)

Tabular testbed worlds (16 PE runs per setting; full coverage for breast-cancer, 55% for
artificial-characters):

| setting | joint | original (A4) |
|---|---|---|
| breast-cancer, eps 10 | **0.840** (0.178) | 0.655 (0.034) |
| artificial-characters, eps 10 | **0.822** (0.159) | 0.698 (0.036) |
| breast-cancer, eps 1 | 0.557 | 0.548 |
| artificial-characters, eps 1 | 0.527 | **0.574** |

Coverage of the private set, eps 10 (one logged run each):

| coverage | adult: joint / A4 | person-activity: joint / A4 |
|---|---|---|
| ~1-6% | 0.54-0.55 / **0.57-0.58** | 0.54-0.55 / **0.56** |
| ~25% | 0.565 / 0.567 | 0.558 / 0.553 |
| ~50% | **0.603** / 0.568 | **0.592** / 0.553 |
| ~75-90% | **0.69-0.77** / 0.57 | **0.67-0.74** / 0.55 |
| 100% | **0.826** (0.199) / 0.568 (0.020) | **0.789** (0.146) / 0.554 (0.013) |

Text and images (evo-mia's audits use all members, i.e. full coverage):

| run | eps | joint | best original variant |
|---|---|---|---|
| OpenReview (Aug-PE) | ~10 | **0.720** (0.113) | 0.687 A4 (0.047) |
| MIMIC (Aug-PE) | ~10 | **0.648** (0.055) | 0.619 A4 (0.022) |
| Camelyon17 (DPSDA) | 10 | **0.643** (0.034) | 0.575 A4 (0.019) |
| Cat (DPSDA, 34 non-members) | 10 | 0.719 | 0.709 B5 |
| CIFAR-10 (DPSDA), 10 rounds | 10 | 0.521 | **0.556** B5 |
| all of the above | ~1 | 0.50-0.52 | **0.52-0.61** (A4) |

Checks: shuffling each round's counts across cells drops the joint attack to 0.50; at eps 1
its TPR at 1% FPR stays inside the DP bound; the attacker's nearest cell matches PE's vote
cell for 100% of records.  Exact binary inference (Gibbs sampling) gives the same result as
the Gaussian solve, so the remaining gap is information, not inference.

## Files

| file | what |
|---|---|
| `joint_sparse.py` | **the joint attack** (sparse, cross-validated member share); tabular worlds / pools |
| `joint_modality.py` | joint vs. original on text and image runs, through evo-mia's drivers |
| `make_coverage_worlds.py` | target pools of chosen coverage on one logged PE run |
| `score_worlds.py` | original EVO-MIA variants per tabular world (evo-mia's scorer) |
| `joint_attack.py` | first, dense version (empirical-Bayes member share); small pools only |
| `blackbox_worlds.py` | final-synthetic-data-only baselines (DCR, k-NN density ratio) |
| `world_lira.py` | per-record LiRA calibration across worlds (does not help PE) |
| `joint_binary.py`, `joint_bg.py`, `joint_plus_a4.py` | variants tried (binary inference, learned background, ensemble); no reliable gain |
