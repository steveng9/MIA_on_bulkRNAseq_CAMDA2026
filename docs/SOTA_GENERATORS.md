# State-of-the-art generators: TabSyn, TabPFN, DPSynth, DP-CVAE

Added 2026-10-03 to answer the expected reviewer objection that the CAMDA
generators (MVN, CVAE, NoisyDiffusion, DP-PGM) are not the state of the art for
tabular synthesis.  All four are ordinary targets: built by
`scripts/build_targets.py`, written in the same format to
`artifacts/targets/<dataset>/<generator>[/<variant>]/split_<s>/`, and attacked
by the existing attacks through `configs/experiments/grid_sota_{brca,combined}.yaml`.

```bash
python scripts/build_targets.py --dataset BRCA --generators tabsyn dpcvae \
    "dpcvae@epsilon=1000" "dpsynth@mechanism=star" \
    "tabpfn@model_version=v2,preprocess=standard+pca:64"
python scripts/run_experiment.py configs/experiments/grid_sota_brca.yaml
```

## Where each one comes from

| name | method | source used | how it is included | runs in |
|---|---|---|---|---|
| `tabsyn` | latent diffusion over a transformer VAE (Zhang et al., ICLR 2024) | amazon-science/tabsyn @ cb5ac0f | model code vendored verbatim (`tabsyn_vae.py`, `tabsyn_diffusion.py`); training/sampling recipe re-expressed in `tabsyn.py` with upstream's constants as defaults | main env |
| `tabpfn` | in-context generation from a tabular foundation model (Hollmann et al., Nature 2025) | `tabpfn` 9.1.0 + `tabpfn-extensions` 0.6.3 (`TabPFNUnsupervisedModel`) | imported | `camda_sota` |
| `dpsynth` | DP marginals + Private-PGM: MST, AIM, SWIFT, independent, star | google/dpsynth @ 15acc97 (`TabularConfig`, in-memory path), mbi @ 1821fa0 | imported | `camda_sota` |
| `dpcvae` | the challenge's DP-SGD CVAE baseline | Health-Privacy-Challenge starter, `cvae.py` with `generator_name == "dpcvae"` and its `dpcvae_config` | training loop mirrored on our vendored CVAE; Opacus 1.6 | `camda_sota` |

### The second environment

TabPFN needs torch >= 2.5, Opacus 1.x needs torch >= 2.6 and DPSynth needs
Python >= 3.12, while the main environment is pinned to Python 3.9 / torch 1.13
/ numpy 1.23 by Private-PGM and the published NoisyDiffusion weights.
`environment-sota.yml` describes the sibling environment (`camda_sota`,
6.4 GB).  You never activate it: `mia.generators.build` notices that a
generator's `requires` are not met and returns a proxy that runs the same class
in the other interpreter (`mia/generators/remote.py`).  Arrays cross as `.npy`
files.  A generator's `report()` (noise multiplier, epsilon spent, what the
guarantee covers) and its fully resolved parameters come back and are written
to the target's `meta.json`.

## Recipes and deviations

Anything not listed is upstream's default.

**TabSyn.**  VAE: 2 layers, token width 4, 4,000 epochs, beta 1e-2 -> 1e-5;
diffusion: width-1024 MLP, up to 10,001 epochs, 50 sampling steps; scaling:
quantile-normal with n // 30 knots.  The label is the table's one categorical
column and is generated jointly.  Two deviations, both forced:

- `batch_size` 256, not 4,096.  Each gene is a token, so attention is over 980
  tokens; the whole cohort in one batch needs ~40 GB.
- `val_frac` 0.  Upstream schedules the VAE on the dataset's test split, which
  here is the non-member half.  The schedule runs on the training rows instead,
  so every member is trained on.  `val_frac=0.1` holds out members instead.

**TabPFN.**  Temperature 1.0, 3 conditioning permutations, label first and
flagged categorical.  Nothing is trained; the training table is the context.

- The current weights (v2.5 and later, including the default, TabPFN-3.5) are
  gated under Prior Labs' non-commercial research licences.  They were accepted
  on 2026-10-03 for Steven's account; the API key is read from `TABPFN_TOKEN`
  or from `~/.config/camda/tabpfn_token` (outside the repository, mode 600).
  `model_version=v2` (the Nature-paper weights) needs neither.
- Cost decides the preprocessing.  One in-context fit per (column,
  permutation); a late column costs ~10 s on our shared GPUs for v2 and 3.5
  alike, so gene-by-gene generation of 978 genes took 3.8 h for one BRCA target.  `preprocess=standard+pca:64` generates 64
  component scores instead (9-13 min), at the price of a rank-64 release.
- The extension fails writing GPU predictions into its CPU table; the wrapper
  moves each prediction to the CPU.  Telemetry is switched off.

**DPSynth.**  32 bins per gene, 10% of the budget on per-column
initialisation.  One deviation: DPSynth calibrates its noise with an RDP and a
PLD accountant and keeps the tighter; past 64 columns we run RDP only
(`accountant=auto`), because the PLD pass takes over an hour at 978 genes
(4 s for RDP).  At 20 genes, where both ran, RDP was the tighter of the two, so
the released noise is what the library would have chosen.  `mechanism=star` is DPSynth's `Direct`
mechanism on the (gene, label) cliques of our own DP-PGM, so table selection
can be compared inside one library.  DPSynth privatises its own binning and the
row count, so with public bounds (`bounds=public`, default 0-24 for VST) the
release is DP end to end -- unlike `pgg` and `dpcvae`.  `bounds=data` uses each
gene's training range and says so in the report.

**DP-CVAE.**  z = 64, one-hot condition, clip 0.1, epsilon 10, delta 1e-5,
10,000 iterations.  Deviations:

- `accountant=auto`: PRV (Opacus's default, so the baseline's) up to
  epsilon = 100, RDP beyond, because PRV calibration does not finish at
  epsilon = 1000.
- As in the baseline, the StandardScaler and the label counts are not
  privatised, so the default is DP-SGD-trained but not DP end to end;
  `report()` records this.  `preprocess=fixed:0:24` and `private_labels=true`
  close both gaps.

## Preprocessing

Model-level preprocessing is one generator parameter, a chain such as
`clip:0.001:0.999+quantile` or `standard+pca:64` (`mia/preprocessing.py`).  It
is part of the target's name, recorded in `meta.json` with three flags
(lossless, data-independent, uses labels), and swept with
`scripts/preprocess_ablation.py`.  Cohort-level preparation for new datasets
(CPM, log, gene selection) is separate and lives in the dataset manifest:
`configs/datasets/README.md`.

## Results so far (BRCA, split 1 only)

One split: differences of a few points are noise.  Full tables need splits 1-5.

### Fidelity and utility (`scripts/eval_fidelity.py`)

TSTR = macro-F1 of a subtype classifier trained on synthetic, tested on real
(real-on-real: 0.811).  Discriminator AUC: 0.5 is indistinguishable from real.

| target | TSTR F1 | ratio to real | discriminator AUC | per-gene W1 | correlation MAE |
|---|---|---|---|---|---|
| CVAE | 0.780 | 0.961 | 0.898 | 0.173 | 0.069 |
| MVN | 0.708 | 0.872 | 0.963 | 0.241 | 0.048 |
| NoisyDiffusion | 0.740 | 0.912 | 0.962 | 0.197 | 0.099 |
| DP-PGM (`pgg`, eps 10) | 0.498 | 0.614 | 1.000 | 0.292 | 0.129 |
| TabSyn (upstream recipe) | 0.629 | 0.775 | 0.998 | 0.424 | 0.100 |
| TabSyn, tuned (final recipe) | 0.776 | 0.956 | 0.931 | 0.089 | 0.053 |
| TabSyn, tuned diffusion stage only | 0.628 | 0.774 | 0.935 | 0.129 | 0.057 |
| TabPFN-3.5, gene by gene | 0.795 | 0.979 | 0.651 | 0.058 | 0.037 |
| TabPFN-3.5, 64 components | 0.732 | 0.902 | 0.965 | 0.123 | 0.064 |
| TabPFN v2, 64 components | 0.714 | 0.880 | 0.978 | 0.147 | 0.066 |
| DP-CVAE, eps 10 | 0.261 | 0.322 | 1.000 | 0.960 | 0.181 |
| DP-CVAE, eps 1000 | 0.517 | 0.637 | 0.999 | 0.481 | 0.115 |

### Attack AUC

| target | MahalaMIA | GAN-leaks | MC | conf-LR | conf-RF |
|---|---|---|---|---|---|
| CVAE | 0.854 | 0.731 | 0.563 | 0.534 | 0.537 |
| MVN | 0.942 | 0.541 | 0.502 | 0.561 | 0.515 |
| NoisyDiffusion | 0.822 | 0.561 | 0.516 | 0.544 | 0.503 |
| DP-PGM (`pgg`) | 0.512 | 0.514 | 0.517 | 0.516 | 0.510 |
| TabSyn | 0.577 | 0.631 | 0.525 | 0.528 | 0.512 |
| TabSyn, tuned (final recipe) | 0.697 | 0.782 | 0.563 | 0.554 | 0.540 |
| TabSyn, tuned diffusion stage only | 0.661 | 0.757 | 0.563 | 0.521 | 0.527 |
| TabPFN-3.5, gene by gene | 0.589 | 0.536 | 0.516 | 0.530 | 0.522 |
| TabPFN-3.5, 64 components | 0.635 | 0.554 | 0.508 | 0.507 | 0.511 |
| TabPFN v2, 64 components | 0.627 | 0.546 | 0.516 | 0.520 | 0.516 |
| DP-CVAE, eps 10 | 0.518 | 0.508 | 0.520 | 0.518 | 0.508 |
| DP-CVAE, eps 1000 | 0.534 | 0.514 | 0.508 | 0.479 | 0.503 |

MeLoMIA and MAMA-MIA have not been run on the new targets.

Gene-by-gene TabPFN-3.5 over BRCA splits 1, 2, 4, 5 (split 3 is building):
discriminator AUC 0.62-0.65, per-gene W1 0.057-0.062, correlation MAE
0.036-0.039, TSTR F1 at 0.90-1.02 of real; MahalaMIA 0.572 +- 0.021, GAN-leaks
0.528 +- 0.014, MC 0.518, conf-LR 0.536, conf-RF 0.516.

Tuned TabSyn over BRCA splits 1-5 (3.9 h per target with four built at once):

| split | TSTR F1 | real F1 | ratio | discriminator AUC | per-gene W1 | correlation MAE | MahalaMIA | GAN-leaks | MAMA-MIA |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.776 | 0.811 | 0.956 | 0.931 | 0.089 | 0.053 | 0.697 | 0.782 | 0.559 |
| 2 | 0.664 | 0.732 | 0.907 | 0.972 | 0.095 | 0.064 | 0.692 | 0.789 | 0.543 |
| 3 | 0.642 | 0.836 | 0.768 | 0.979 | 0.125 | 0.060 | 0.629 | 0.760 | 0.544 |
| 4 | 0.628 | 0.767 | 0.819 | 0.984 | 0.153 | 0.073 | 0.631 | 0.774 | 0.547 |
| 5 | 0.838 | 0.843 | 0.995 | 0.941 | 0.102 | 0.061 | 0.663 | 0.759 | 0.583 |
| mean | 0.710 | 0.798 | 0.889 | 0.961 | 0.113 | 0.062 | 0.662 | 0.773 | 0.555 |

Split 1, the split the recipe was tuned on, is among the best two: the
five-split mean (utility ratio 0.89, discriminator AUC 0.96) is the honest
figure, not the split-1 row in the tables above.  GAN-leaks is stable across
splits (0.76-0.79) and its TPR at 1% FPR is 0.50-0.53: about half the members
are recovered with almost no false positives, i.e. the tuned model copies
training rows closely.  MC reads 0.563 on every split because it is saturated:
it flags the 10% of challenge records nearest to a synthetic row, all of them
are members, and 0.5 + 0.125/2 is the most that rule can score.

### Preprocessing ablation, CVAE (`results/preprocess_ablation.csv`)

| preprocess | ratio to real | TSTR F1 | per-gene W1 | correlation MAE | discriminator AUC |
|---|---|---|---|---|---|
| standard (challenge default) | 0.961 | 0.780 | 0.173 | 0.069 | 0.898 |
| quantile | 0.984 | 0.798 | 0.195 | 0.068 | 0.913 |
| minmax | 0.731 | 0.593 | 0.449 | 0.278 | 0.999 |
| robust | 0.913 | 0.740 | 0.189 | 0.058 | 0.904 |
| clip:0.001:0.999+standard | 0.980 | 0.795 | 0.187 | 0.066 | 0.898 |
| classcenter+standard | 0.940 | 0.763 | 0.232 | 0.074 | 0.972 |
| standard+pca:64 | 0.959 | 0.778 | 0.366 | 0.128 | 0.999 |
| standard+pca:256 | 1.019 | 0.827 | 0.259 | 0.080 | 0.972 |

### Preprocessing ablation, TabSyn and DP-CVAE

| generator | preprocess | ratio to real | TSTR F1 | per-gene W1 | correlation MAE | discriminator AUC | build |
|---|---|---|---|---|---|---|---|
| TabSyn | quantile_n30 (upstream) | 0.775 | 0.629 | 0.424 | 0.100 | 0.998 | 1.9 h |
| TabSyn | standard+pca:64+standard | 0.722 | 0.585 | 0.193 | 0.059 | 0.985 | 13 min |
| TabSyn | standard+pca:256+standard | 0.596 | 0.484 | 0.111 | 0.059 | 0.961 | 11 min |
| TabSyn | standard+pca:64+quantile_n30 | 0.739 | 0.599 | 0.149 | 0.062 | 0.976 | 7 min |
| DP-CVAE eps 10 | standard (baseline) | 0.322 | 0.261 | 0.960 | 0.181 | 1.000 | 5 min |
| DP-CVAE eps 10 | fixed:0:24 (DP end to end) | 0.183 | 0.148 | 1.324 | 0.214 | 1.000 | 2 min |
| DP-CVAE eps 10 | standard+pca:64+standard | 0.282 | 0.229 | 1.126 | 0.103 | 0.999 | 2 min |

PCA scores must be re-standardised before TabSyn (`+standard`); on raw scores
its VAE diverges.  For TabSyn, PCA more than halves the marginal and
correlation error and cuts the build from hours to minutes, but subtype
utility falls (0.629 -> 0.484 at 256 components); the cause is not diagnosed.


### Cost per BRCA target (shared GPU, 871 training rows, 978 genes)

| target | wall-clock |
|---|---|
| TabSyn, upstream recipe | 1.9 h (VAE 4,000 epochs; diffusion stopped early at 1,625) |
| TabPFN, 64 components | 9-13 min |
| TabPFN-3.5, gene by gene | 3.8 h |
| DP-CVAE | ~5 min |
| DPSynth star, 100 genes | 4.5 min (3 min private binning, 1.5 min fit) |
| DPSynth star, 250 genes | 25 min (7 min binning, 17 min fit) |
| DPSynth star, 978 genes | not completed: killed after 3 h at 24 GB |


### DPSynth does not reach 978 genes as shipped

Three separate costs, found in this order:

1. **Calibration.**  The PLD accountant composes ~7 events per gene one at a
   time: over an hour at 978 genes.  Fixed with the library's own RDP option
   (`accountant=auto`), 4 s.
2. **Progress logging.**  mbi's default callback checks consistency between
   every pair of cliques sharing an attribute -- all ~478,000 pairs in a star --
   and its compilation does not finish.  Switched off in our wrapper; the
   fitted model is unchanged.
3. **The fit itself.**  JAX compiles one program for the whole graphical model
   and the time grows roughly with the cube of the gene count on CPU (1.5 min
   at 100 genes, 17 min at 250, with either the Shafer-Shenoy or the implicit
   oracle).  Extrapolated to 978 genes that is of the order of a day, and the
   one attempt was killed at 24 GB after 3 h.  Not fixed.

So `dpsynth` targets build today up to a few hundred genes.  On a 50-gene probe
with public 0-24 bounds the synthetic per-gene SD was 2.9x the real one, so
bounds and binning will dominate its quality (`bounds=data` is the comparison).
Ways to reach 978 genes, none done: a GPU JAX build; fitting DPSynth's noisy
marginals with our existing Private-PGM code; or, for the star only, the
closed-form model p(label) x prod p(gene | label).

## Tuning TabSyn for a small, wide cohort

TabDDPM was considered instead and dropped: with all-numerical genes and one
label it is Gaussian diffusion on scaled expression with an MLP denoiser, which
is what NoisyDiffusion already is.  TabSyn (diffusion in a learned latent
space) is the distinct method.

**Diagnosis** (`logs/sota/tabsyn_diag.py`, BRCA split 1, upstream recipe).
Decoding the encoded training rows reproduces them exactly (per-gene SD ratio
1.000, correlation error 0.000, labels 100%), so the VAE is not the problem.
Decoding *diffusion samples* gives SD ratio 1.55 and less than half the real
correlation strength.  In latent space the samples have 4x the training
variance and 9% of it in their top 10 principal directions, against 46% for
the training latents: close to isotropic noise.  Two causes:

1. *Latent scale.*  Upstream feeds the denoiser (z - mean) / 2.  EDM's
   preconditioning assumes data of standard deviation 0.5; here the result has
   0.15 (median per dimension), because the KL weight is annealed to 1e-5.
2. *Schedule.*  Upstream multiplies the LR by 0.9 after 20 epochs without
   improvement and stops after 500.  With 871 rows in batches of 4,096 an epoch
   is one gradient step on a loss that is noisy by construction, so training
   ended after 1,896 steps with the LR at 2e-7.

**Fixes** (`logs/sota/tabsyn_diff_exp.py`: same trained VAE, diffusion stage
only).

| diffusion stage | TSTR F1 | per-gene W1 | correlation MAE | discriminator AUC |
|---|---|---|---|---|
| upstream | 0.513 | 0.418 | 0.104 | 0.998 |
| 20,000 steps, upstream scale | 0.623 | 0.505 | 0.098 | 0.999 |
| latents scaled to SD 0.5, upstream schedule | 0.672 | 0.110 | 0.045 | 0.888 |
| scaled + 5,000 steps | 0.652 | 0.124 | 0.057 | 0.907 |
| scaled + 20,000 steps (seeds 0 / 1 / 2) | 0.771 / 0.699 / 0.749 | 0.123-0.135 | 0.056-0.060 | 0.923-0.937 |
| scaled + 40,000 / 60,000 steps | 0.620 / 0.672 | 0.118 / 0.132 | 0.062 | 0.921 / 0.913 |
| scaled + 20,000, denoiser width 512 / 2,048 | 0.739 / 0.636 | 0.133 / 0.121 | 0.052 / 0.043 | 0.889 / 0.907 |
| scaled + 20,000, LR 3e-4 | 0.661 | 0.119 | 0.053 | 0.907 |
| scaled + 20,000, no weight averaging | 0.682 | 0.131 | 0.059 | 0.924 |
| scaled + 20,000, diffusing in the latents' top 256 / 64 PCs | 0.685 / 0.548 | 0.354 / 0.170 | 0.140 / 0.143 | 0.968 / 0.983 |

The scale is the decisive fix.  Beyond it, width, LR, step count and weight
averaging move marginal and correlation error very little, and their effect on
utility is inside the seed spread (0.70-0.77), so upstream's width and LR are
kept with 20,000 steps.  Reducing the latent space by PCA hurts.

**Label.**  TabSyn generates the subtype jointly and under-produces the rare
ones (the 4% subtype comes out at 2-3%).  Keeping generated rows class by class
until the release has the training proportions (`class_freq=train`; no row is
altered) raises TSTR F1 from 0.656 / 0.677 / 0.690 to 0.773 / 0.706 / 0.730
over three sampling seeds, with the other measures unchanged.

**Input scaling.**  Upstream's quantile-normal scaling returns a few genes at
about twice their real spread.  With `clip:0.001:0.999+standard` instead, on
all 978 genes: TSTR F1 0.776, per-gene W1 0.089 (0.129 with quantile),
correlation MAE 0.053, discriminator AUC 0.931.

**Tuned recipe** (final):
`tabsyn@class_freq=train,diff_schedule=steps,latent_scale=std,preprocess=clip:0.001:0.999+standard`.
`tabsyn` alone remains the upstream recipe.  Upstream -> tuned on BRCA split 1:
TSTR F1 0.629 -> 0.776, W1 0.424 -> 0.089, correlation MAE 0.100 -> 0.053,
discriminator AUC 0.998 -> 0.931.

**Privacy.**  Fixing the generator made it much easier to attack: GAN-leaks
goes from AUC 0.631 (upstream) to 0.782 (final recipe), above its 0.731 on
CVAE, and MahalaMIA from 0.577 to 0.697.

## Gene subset (for DPSynth)

DPSynth cannot fit 978 genes (below), so it is run on a subset, and every other
generator is rebuilt on the same subset for a like-for-like comparison.  The
subset is a derived cohort, e.g. `configs/datasets/BRCA_HVG200.yaml`
(`parent: BRCA`, `prepare: hvg:200`): same samples, labels and splits.

**Rule: the 200 genes with the largest variance over the whole cohort.**
`scripts/gene_subset_ablation.py` compared three rules at five sizes without
any generator (`results/gene_subset_ablation.csv`; means over splits 1-5).
"F1 kept" is real-on-real macro-F1 on the subset relative to all 978 genes
(0.798 BRCA, 0.968 COMBINED); "variance" is the share of held-out variance of
all 978 genes that a ridge regression on the subset explains; "overlap" is the
Jaccard index between the set chosen on the whole cohort and on members only.

| cohort | rule | F1 kept at k = 25 / 50 / 100 / 200 / 400 | variance at 200 | overlap at 200 |
|---|---|---|---|---|
| BRCA | most variable (`hvg`) | 0.911 / 0.958 / 0.970 / 0.986 / 0.981 | 0.572 | 0.967 |
| BRCA | most class-separating (`anova`, F statistic) | 0.880 / 0.963 / 0.966 / 0.959 / 0.989 | 0.595 | 0.942 |
| BRCA | random | 0.725 / 0.900 / 0.919 / 0.919 / 0.973 | 0.630 | - |
| COMBINED | most variable | 0.953 / 0.967 / 0.978 / 0.990 / 0.999 | 0.705 | 0.988 |
| COMBINED | most class-separating | 0.959 / 0.970 / 0.980 / 0.994 / 1.000 | 0.721 | 0.970 |
| COMBINED | random | 0.877 / 0.960 / 0.975 / 0.989 / 0.995 | 0.758 | - |

- Variance and the F statistic keep the same utility from 50 genes up; both
  beat random.  Variance is preferred because it never sees the labels (a
  label-chosen subset would favour utility by construction), is the more stable
  under a change of who is a member, and is the standard rule in
  transcriptomics.  The data are variance-stabilised (DESeq2 VST), so raw
  variance is not confounded with expression level.
- k = 200: utility is flat from 100 genes up, and 200 is the largest size at
  which every DPSynth mechanism builds in minutes.
- Limitation: a random subset predicts slightly more of the rest of the
  transcriptome than the most variable genes do (variable genes are mutually
  redundant).
- Privacy: selection is cohort-level preparation, like the challenge's VST and
  landmark-gene filter; it sees members and non-members alike.  It is not
  covered by any generator's DP guarantee.

### BRCA_HVG200, split 1

Real-on-real F1 0.812.  DPSynth and DP-PGM at eps = 10, DP-CVAE at eps = 10.

| target | TSTR F1 | discriminator AUC | per-gene W1 | correlation MAE | MahalaMIA | GAN-leaks |
|---|---|---|---|---|---|---|
| MVN | 0.720 | 0.960 | 0.219 | 0.050 | 0.671 | 0.548 |
| CVAE | 0.757 | 0.896 | 0.194 | 0.069 | 0.667 | 0.719 |
| NoisyDiffusion | 0.748 | 0.910 | 0.220 | 0.090 | 0.644 | 0.671 |
| TabPFN-3.5, gene by gene | 0.797 | 0.623 | 0.056 | 0.038 | 0.572 | 0.547 |
| TabSyn, tuned (final recipe) | 0.776 | 0.985 | 0.133 | 0.057 | 0.574 | 0.782 |
| TabSyn, tuned diffusion (labels as generated) | 0.755 | 0.999 | 0.235 | 0.066 | 0.558 | 0.755 |
| DP-PGM (`pgg`, our star) | 0.607 | 1.000 | 0.284 | 0.108 | 0.516 | 0.529 |
| DPSynth star, public bounds 0-24 | 0.564 | 1.000 | 0.680 | 0.131 | 0.513 | 0.517 |
| DPSynth star, per-gene data bounds | 0.521 | 0.998 | 0.167 | 0.135 | 0.517 | 0.521 |
| DPSynth MST | 0.360 | 1.000 | 1.083 | 0.177 | 0.515 | 0.522 |
| DP-CVAE | 0.477 | 1.000 | 1.054 | 0.178 | 0.508 | 0.515 |

TabSyn's discriminator AUC of 0.999 here (0.935 on all 978 genes) comes from a
few genes generated with about twice the real spread.  Input scaling is part of
it (tuned recipe with class matching, same cohort and split):

| TabSyn input scaling | TSTR F1 | per-gene W1 | correlation MAE | discriminator AUC |
|---|---|---|---|---|
| quantile_n30 (upstream) | 0.788 | 0.232 | 0.068 | 0.999 |
| quantile (1,000 knots) | 0.692 | 0.214 | 0.068 | 0.998 |
| standard | 0.768 | 0.156 | 0.058 | 0.993 |
| clip:0.001:0.999+standard | 0.776 | 0.133 | 0.057 | 0.985 |

Standard scaling nearly halves the marginal error; it was then confirmed on all
978 genes and adopted.  Why the discriminator still separates the sets on this
cohort (0.985) and much less on the full one (0.931) is not resolved.

Within DPSynth the star beats MST by a wide margin, and our own DP-PGM star is
ahead of DPSynth's.  Public 0-24 bounds cost DPSynth most of its marginal
fidelity (W1 0.680 against 0.167 with data bounds).  MC, conf-LR and conf-RF are
at 0.49-0.56 on every target here.  DPSynth builds take 7-12 min.

## MeLoMIA-TabSyn (black box and white box)

Added 2026-10-05 because the tuned TabSyn is the most attackable new target.
`melomia_tabsyn` is a third MeLoMIA backend beside ND and CVAE
(`mia/attacks/melomia/backends.py`, `TabSynBackend`); everything above the
backend (shadow splits, per-record calibration, classifier search) is shared.

**Model roles and how each is tuned** (the rule of the TimeDiff audit,
`~/ansons_capstone` commits ccb82c0 and 1c01066):

| role | recipe | tuned for |
|---|---|---|
| target | `tabsyn.TUNED` | fidelity, utility |
| base shadow | identical to the target; releases as many rows as it was trained on | being a copy of the target, so its synthetic data is distributed like the target's release |
| synth-shadow | probe: `TUNED` with denoiser width 1024 -> 2048 (4x parameters), 20,000 -> 40,000 steps, label-free | memorising the synthetic data it is fitted to |
| final proxy | the same probe recipe, fitted to the target's release | the same, so its losses are distributed like the synth-shadows' |

The probe's VAE keeps the target's settings: it already reconstructs its
training rows to an MSE of ~2e-5.  Every model scales its inputs with its own
training data's statistics and scores real records through that same scaler.

**Loss signature**, from both halves of the model.

- *Diffusion half* (the searchable grid, as in MeLoMIA-ND).  A record is
  scaled, encoded to its latent and normalised as in training; at 13 EDM noise
  levels (0.02 ... 40, around the training distribution's 0.03-3) and 200
  frozen noise draws it is corrupted and the denoiser's squared error in
  recovering it is recorded.
- *VAE half* (as in MeLoMIA-CVAE).  The encoder gives each record's posterior
  (mu, sigma); at temperatures 0, 0.5, 1, 1.5, 2, 3 the latent is perturbed as
  mu + alpha * sigma * eps for 16 frozen draws, decoded and scored against the
  input.  It enters as 27 extra features: per temperature the mean, SD, minimum
  and median of the log error; the posterior's KL mean / SD / maximum; the
  norms of mu and sigma; the norm of the normalised latent.  Unlike
  MeLoMIA-CVAE the per-dimension KL is summarised (there are 3,916 latent
  coordinates, not 64), and the temperature sweep is not part of the Optuna
  feature search, because the grid has one sweep axis and the diffusion levels
  hold it.

First look, white box, the real tuned-TabSyn target of BRCA split 1, each
feature alone with no classifier: the diffusion error separates members from
non-members at AUC 0.997-1.000 for every noise level from 0.02 to 5 (0.59 at
10, 0.51 at 40).  The VAE features are weak alone: reconstruction error 0.54-0.55
at temperatures 0-2, KL and norms 0.49-0.51.

**White box** (`white_box: true`).  The adversary holds the target model.
Shadows are the target recipe on real splits, i.e. the black-box stack's base
shadows, which are kept and reused; the signature is read from them and, at
inference, from the target model itself.  No synthetic layer, no proxy.  The
adversary is assumed to know each candidate's class, which a label-conditional
model needs in order to be read.

**Queued** (`scripts/queue_melomia_tabsyn.sh`, progress in
`logs/melomia_tabsyn/`): BRCA K=10 then K=30, black box and white box on the
tuned-TabSyn targets, then MeLoMIA-TabSyn against MVN / CVAE / ND / DP-PGM /
TabPFN; the existing attacks (MeLoMIA-ND, MeLoMIA-CVAE, MAMA-MIA) on the
tuned-TabSyn and TabPFN targets; then COMBINED (K=20).  A TabSyn fit is about
one GPU-hour on BRCA, so the BRCA stack is ~60 GPU-hours plus ~30 for proxies;
COMBINED is about four times that.

Smoke test only so far (200-gene cohort, K=3, models trained for 20 VAE epochs,
no search): black box 0.635, white box 0.992 AUC.  Not a result.
