# MIA on Bulk RNA-seq (CAMDA 2026)

Membership inference attack (MIA) against synthetic bulk RNA-seq data for the
**CAMDA 2026 Health Privacy Challenge (Red Team track)**.

Two attack families are implemented in parallel — NoisyDiffusion (ND) and CVAE —
each with a real-data-shadow variant and a synth-shadow ablation.

---

## Repository Layout

```
mia/
├── config.py          # Shared config + ND profiles + CVAE profiles
├── data_utils.py      # Data loading, custom splits, scalers, label inference
├── classifier.py      # MembershipMLP; train_classifier; load_classifier_from_dir
│
├── nd/                # NoisyDiffusion attack
│   ├── shadow_model.py    # Train/load EmbeddedDiffusion shadows (imports from ~/CAMDA25_NoisyDiffusion)
│   ├── loss_features.py   # Extract N_NOISE × T_LIST MSE loss trajectories
│   ├── attack.py          # Main pipeline (real-data shadows)
│   └── attack_synth_shadow.py  # Synth-shadow pipeline (ND synthetic data as training set)
│
└── cvae/              # CVAE attack (parallel structure)
    ├── model.py           # CVAE nn.Module (adapted from Health-Privacy-Challenge repo)
    ├── shadow_model.py    # Train/load CVAE shadows + generate synthetic data
    ├── loss_features.py   # Extract stochastic rec-losses + per-dim KL features
    ├── attack.py          # Main pipeline (real-data shadows)
    └── attack_synth_shadow.py  # Synth-shadow pipeline (CVAE-generated synthetic data)

mia_output/            # All artifacts (auto-created)
├── splits/BRCA/splits.json          # Custom train/test splits (shared by ND and CVAE)
├── shadow_models/BRCA/              # ND shadow checkpoints + target_proxy.pt
├── features/BRCA/                   # ND feature files (.npz)
├── classifiers/mlp_best.pt          # ND MLP
├── synth_val_models/BRCA/           # ND domain-gap validation models
├── synth_val_features/BRCA/         # ND domain-gap features
├── synth_shadow/                    # ND synth-shadow outputs
│   ├── shadow_models/BRCA/
│   ├── features/BRCA/
│   └── classifiers/mlp_best.pt
└── cvae/
    ├── shadow_models/BRCA/          # CVAE shadow checkpoints
    ├── features/BRCA/               # CVAE feature files (rec_losses + per_dim_kl)
    ├── classifiers/mlp_best.pt      # CVAE MLP
    ├── synthetic_data/BRCA/         # CVAE-generated synthetic data (for synth-shadow)
    └── synth_shadow/
        ├── shadow_models/BRCA/
        ├── features/BRCA/
        └── classifiers/mlp_best.pt
```

---

## Prerequisites

### External repos (must be cloned as siblings)

```
~/
├── MIA_on_bulkRNAseq_CAMDA2026/   ← this repo
├── CAMDA25_NoisyDiffusion/         ← Blue Team repo (provides ND model + labeled synthetic data)
└── Health-Privacy-Challenge/       ← Challenge repo (provides challenge data paths)
```

The ND shadow model code (`mia/nd/shadow_model.py`) imports `EmbeddedDiffusion` and
`DiffusionTrainer` directly from `~/CAMDA25_NoisyDiffusion/TCGA-BRCA/model.py` at runtime.

### Challenge data

Expected at `/home/golobs/data/CAMDA26/` (configured in `mia/config.py:CHALLENGE_DIR`):

```
/home/golobs/data/CAMDA26/
├── RED_TCGA-BRCA/
│   ├── TCGA-BRCA_primary_tumor_star_deseq_VST_lmgenes.tsv   # real data (978 genes × samples)
│   └── synthetic_data_1.csv                                  # challenge synthetic data
└── RED_TCGA-COMBINED/
    ├── TCGA-COMBINED_primary_tumor_star_deseq_VST_lmgenes.tsv
    └── synthetic_data_1.csv
```

### Python environment

```bash
conda env create -f environment.yaml   # or use the existing conda env
conda activate <env_name>
```

Required packages (non-exhaustive): `torch`, `numpy`, `pandas`, `scikit-learn`, `pyyaml`.

---

## Running the Attacks

All pipelines are run as Python modules from the **project root**:

```bash
cd ~/MIA_on_bulkRNAseq_CAMDA2026
```

### NoisyDiffusion attack

#### Main pipeline (real-data shadows)

```bash
python -m mia.nd.attack [OPTIONS]

Options:
  --dataset   BRCA | COMBINED          (default: BRCA)
  --device    cuda:0 | cpu             (default: cuda:0)
  --profile   baseline | tuned         (default: current config values)
  --force     STAGES                   comma-separated or 'all'
                                       valid: shadows, features, classifier,
                                              synth_val, challenge

Examples:
  python -m mia.nd.attack --dataset BRCA
  python -m mia.nd.attack --dataset BRCA --profile tuned
  python -m mia.nd.attack --dataset BRCA --force shadows,features
  python -m mia.nd.attack --dataset BRCA --force all
```

**Pipeline steps:**
```
STEP 0: Generate N custom 80/20 splits of real data (skipped if splits.json exists)
STEP 1: Train N shadow EmbeddedDiffusion models on real data subsets
STEP 2: Extract loss features for all real samples via each shadow
STEP 3: Train MLP classifier (70% of splits for train, 30% for val)
STEP 4: Per-split evaluation (ACC + TPR@10%FPR)
STEP 5: Synthetic validation — domain gap measurement
STEP 6: Challenge predictions → synthetic_data_1_predictions.csv
```

#### Synth-shadow pipeline (ablation: shadows trained on ND synthetic data)

```bash
python -m mia.nd.attack_synth_shadow [OPTIONS]

Options: same as above (--dataset, --device, --profile, --force)

Examples:
  python -m mia.nd.attack_synth_shadow --dataset BRCA
  python -m mia.nd.attack_synth_shadow --dataset BRCA --profile tuned --force all
```

---

### CVAE attack

#### Main pipeline (real-data shadows)

```bash
python -m mia.cvae.attack [OPTIONS]

Options:
  --dataset            BRCA | COMBINED          (default: BRCA)
  --device             cuda:0 | cpu             (default: cuda:0)
  --profile            baseline | tuned         (default: current config values)
  --label-mode         knn | none               (default: knn)
  --generate-synthetic                          also save CVAE synthetic data
                                               (required before running synth-shadow)
  --force              STAGES                   comma-separated or 'all'
                                               valid: shadows, features, classifier,
                                                      synth_gen, challenge

Examples:
  python -m mia.cvae.attack --dataset BRCA
  python -m mia.cvae.attack --dataset BRCA --label-mode none
  python -m mia.cvae.attack --dataset BRCA --generate-synthetic
  python -m mia.cvae.attack --dataset BRCA --profile tuned --force all
```

**Pipeline steps:**
```
STEP 0: (Re-use) custom splits from mia_output/splits/ — same as ND attack
STEP 1: Train N shadow CVAE models on real data subsets
STEP 2: Extract CVAE features for all real samples via each shadow
STEP 3: Train MLP classifier (70/30 split)
STEP 4: Per-split evaluation (ACC + TPR@10%FPR)
STEP 5: (if --generate-synthetic) Generate and save CVAE synthetic data per split
STEP 6: Challenge predictions → synthetic_data_1_predictions_cvae.csv
```

**Label mode options:**
- `knn` (default): Predict cancer subtype labels via KNN on ND synthetic data; pass as
  class condition to CVAE encoder and decoder. Exploits class structure.
- `none`: Pass a true zero vector as condition, bypassing class conditioning entirely.
  Trains and evaluates an unconditional-equivalent CVAE. Useful as an ablation.

#### Synth-shadow pipeline (ablation: shadows trained on CVAE-generated synthetic data)

Must run main pipeline with `--generate-synthetic` first.

```bash
python -m mia.cvae.attack_synth_shadow [OPTIONS]

Options:
  --dataset     BRCA | COMBINED          (default: BRCA)
  --device      cuda:0 | cpu             (default: cuda:0)
  --profile     baseline | tuned
  --label-mode  knn | none
  --force       STAGES

Examples:
  # Step 1: generate synthetic data (if not done yet)
  python -m mia.cvae.attack --dataset BRCA --generate-synthetic --force synth_gen

  # Step 2: run synth-shadow
  python -m mia.cvae.attack_synth_shadow --dataset BRCA
```

---

## Configuration Reference

All configuration lives in `mia/config.py`.

### Shared settings

| Variable | Default | Description |
|---|---|---|
| `DEVICE` | `cuda:0` | PyTorch device |
| `SEED` | `42` | Global random seed |
| `SPLIT_MODE` | `custom` | `custom` (random 80/20) or `noisydiffusion` (YAML splits) |
| `NUM_CUSTOM_SPLITS` | `30` | Number of random splits to generate |
| `CUSTOM_SPLIT_RATIO` | `0.8` | Member fraction per split |

### NoisyDiffusion profiles (`--profile`)

| Parameter | `baseline` | `tuned` |
|---|---|---|
| `T_LIST` | `[5,10,20,30,40,50,100]` | `[1,2,5,10,20,50,100,200]` |
| `N_NOISE` | `300` | `100` |
| `FEATURE_MODE` | `raw` (2100 dims) | `summary` (32 dims) |
| `MLP_HIDDEN_DIM` | `200` | `64` |
| `MLP_DROPOUT` | `0.0` | `0.3` |
| `MLP_WEIGHT_DECAY` | `0.0` | `1e-3` |
| `MLP_EPOCHS` | `750` | `2000` |

Raw features are always saved as-is (2100 dims). `prepare_features()` applies summarization
at MLP time, so switching `--profile` **never requires re-running shadow training or feature
extraction**.

### CVAE profiles (`--profile`)

| Parameter | `baseline` | `tuned` |
|---|---|---|
| `CVAE_TEMP_LIST` | `[0.0, 0.5, 1.0, 1.5, 2.0]` | `[0.0, 0.5, 1.0, 1.5, 2.0, 3.0]` |
| `CVAE_N_SAMPLES` | `300` | `100` |
| `CVAE_FEATURE_MODE` | `raw` (1628 dims) | `summary` (148 dims) |
| `CVAE_MLP_HIDDEN_DIM` | `200` | `64` |
| `CVAE_MLP_DROPOUT` | `0.0` | `0.3` |
| `CVAE_MLP_WEIGHT_DECAY` | `0.0` | `1e-3` |
| `CVAE_MLP_EPOCHS` | `750` | `2000` |

### Skip / force logic

Steps 0–2 (split generation, shadow training, feature extraction) are **skipped if outputs
already exist**. To force re-execution:

```bash
--force all                          # re-run everything
--force shadows                      # re-train shadow models only
--force shadows,features             # re-train shadows and re-extract features
--force features,classifier          # re-extract features and re-train MLP
```

Valid stage names:

| Stage name | What it controls |
|---|---|
| `shadows` | Shadow model training (Step 1) |
| `features` | Feature extraction (Step 2) |
| `classifier` | MLP training (Step 3) |
| `synth_val` | ND domain-gap validation (Step 5, ND only) |
| `synth_gen` | CVAE synthetic data generation (Step 5, CVAE only) |
| `challenge` | Challenge proxy + predictions (Step 6) |
| `all` | All of the above |

---

## Feature Design

### NoisyDiffusion features

For each real sample x₀:
- Draw `N_NOISE` fixed noise vectors ε ~ N(0,I)
- For each timestep t in `T_LIST`: compute x_t, predict ε̂, measure MSE(ε̂, ε)
- Result: `(N_NOISE × len(T_LIST))` matrix → 2100 features (baseline raw)

Members have lower MSE because the shadow model has "seen" them during training.

### CVAE features

For each real sample x (with class condition y):

1. **Encode** x → μ, logvar ∈ R^z_dim (deterministic, one encoder pass per sample)
2. **Per-dimension KL**: -0.5 × (1 + logvar_d - μ_d² - exp(logvar_d)) → z_dim = 128 features
   Captures how much each latent dimension is used for this sample.
3. **Temperature sweep**: for each α ∈ `CVAE_TEMP_LIST`:
   - For each of `N_SAMPLES` fixed noise vectors ε ~ N(0,I):
     - z = μ + α·σ·ε  (σ = exp(0.5·logvar))
     - rec_loss = MSE(decode(z, y), x)
   - α=0: deterministic z=μ (all N_SAMPLES give identical loss)
   - Higher α: more stochastic z, reconstruction degrades
   - Members degrade more gracefully (decoder has been trained on them)

**Raw mode**: `N_SAMPLES × len(TEMP_LIST) + z_dim` = 1628 features  
**Summary mode**: `4 stats × len(TEMP_LIST) + z_dim` = 148 features  
(4 stats = mean, std, min, max across noise samples per temperature)

---

## Results

### NoisyDiffusion (BRCA, 30 custom splits)

| Variant | Val TPR@10%FPR | Per-split TPR@10%FPR |
|---|---|---|
| Real-data shadows (baseline) | 0.58 | 0.67 / 0.71 / 0.50 / 0.54 / 0.56 |
| Synth validation (domain gap) | ~0.14 | — |

### CVAE (BRCA)

Not yet evaluated — run `python -m mia.cvae.attack --dataset BRCA`.

---

## Key Design Decisions

- **Shared custom splits**: ND and CVAE use the **same** `splits.json`, enabling
  direct side-by-side comparison on identical membership partitions.
- **Scaler per split**: Each shadow's `QuantileTransformer` is fitted on the same
  real (or synthetic) data used for that shadow's training, then applied to all real
  samples. This prevents information leakage across splits.
- **Fixed noise bank**: The same seed-fixed noise vectors are used across all samples
  and splits, making feature extraction fully reproducible.
- **TPR@10%FPR as primary metric**: Matches the CAMDA challenge evaluation criterion.
- **Domain gap measurement**: ND synth validation step quantifies how much performance
  degrades when the proxy is trained on synthetic (vs real) data — a key diagnostic
  for predicting challenge-time performance.
- **CVAE `condition_mode="none"`**: Passes a true zero tensor (not a dummy label),
  completely bypassing all embedding/linear conditioning layers. This is a genuine
  unconditional ablation, not just fixing the condition to class 0.
