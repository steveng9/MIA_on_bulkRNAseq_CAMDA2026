# MIA on Bulk RNA-seq (CAMDA 2026)

Membership inference attack on diffusion-generated synthetic bulk RNA-seq data
for the CAMDA 2026 Health Privacy Challenge (Red Team).

## Modules

| Module | Purpose |
|--------|---------|
| `config.py` | Paths, hyperparameters, dataset configs (BRCA + COMBINED), split mode settings |
| `data_utils.py` | Data loading, splits YAML, custom split generation, label prediction |
| `shadow_model.py` | Train/load EmbeddedDiffusion shadow models (QuantileTransformer + OneCycleLR + DP noise) |
| `loss_features.py` | Extract 2100-dim loss trajectories (300 noises x 7 timesteps) |
| `classifier.py` | MembershipMLP (2100->200->200->1, tanh+sigmoid, BCE, best by TPR@10%FPR) |
| `attack.py` | Main pipeline: real-data shadows + synthetic validation |
| `attack_synth_shadow.py` | Alternative pipeline: synthetic-data shadows (no domain gap) |

## Two Attack Pipelines

### 1. Main pipeline (`attack.py`)

Shadows trained on **real data** subsets. Includes synthetic validation to measure
the real->synthetic domain gap at inference.

```
STEP 0 (if custom splits): Generate N random 80/20 splits of real data
STEP 1: Train N shadow models on real data subsets
STEP 2: Extract loss features for all real samples (one per shadow)
STEP 3: Train MLP classifier (70/30 train/val split of N shadows)
STEP 4: Per-split evaluation
STEP 5: Synthetic validation (train on ND synth, evaluate MLP — domain gap)
STEP 6: Challenge prediction (target proxy on challenge synthetic)
```

Split modes (configured via `SPLIT_MODE` in `config.py`):
- `"custom"`: generates random 80/20 splits (configurable N, ratio, seed)
- `"noisydiffusion"`: uses NoisyDiffusion repo's 5-fold splits YAML

```bash
python -m mia.attack --dataset BRCA --device cuda:0
```

### 2. Synth-shadow pipeline (`attack_synth_shadow.py`)

Shadows trained on **ND synthetic data** — eliminates the domain gap between
shadow training and challenge inference (both use synthetic-trained models).

```
STEP 1: Train 5 shadow models on ND synthetic datasets
STEP 2: Extract loss features (scaler fitted on synth data)
STEP 3: Train MLP classifier (splits 1-3 train, 4-5 val)
STEP 4: Per-split evaluation
STEP 5: Challenge prediction
```

All artifacts saved under `mia_output/synth_shadow/` (separate from main pipeline).
Challenge predictions: `synthetic_data_1_predictions_synth_shadow.csv`.

```bash
python -m mia.attack_synth_shadow --dataset BRCA --device cuda:0
```

## Key Design Decisions

- **Real data** is a genes x samples TSV (transposed on load), not pre-split CSVs
- **Labels** inferred via KNN on NoisyDiffusion's labeled synthetic data (no separate label files)
- **Ground truth membership** from NoisyDiffusion splits YAML (train_index/test_index)
- **Shadow models** train unconditionally (dummy label=0)
- Both **BRCA** (5 subtypes, 1089 samples) and **COMBINED** (12 cancer types, ~4300 samples) supported

## Hyperparameters

- Shadow: 200 epochs, batch 32, LR 0.001, OneCycleLR, DP noise 1e-5
- Features: T_LIST=[5,10,20,30,40,50,100], N_NOISE=300 -> 2100 dims
- MLP: 750 epochs, LR 1e-4, batch 64, tanh activations, BCE loss
- Model: EmbeddedDiffusion, hidden [2048,2048], time_emb 128, label_emb 64

## Results (BRCA, real-data shadows)

- Best val TPR@10%FPR: 0.58
- Per-split: Split1=0.67, Split2=0.71, Split3=0.50, Split4=0.54, Split5=0.56
- Synthetic validation TPR@10%FPR: ~0.14 (domain gap from real->synth)

## Output Structure

```
mia_output/
├── shadow_models/BRCA/       # real-data shadow_split_{1-5}.pt, target_proxy.pt
├── features/BRCA/            # features_split_{1-5}.npz
├── classifiers/              # mlp_best.pt (main pipeline)
├── splits/BRCA/              # splits.json (custom splits)
├── synth_val_models/BRCA/    # synth_val_split_{1-5}.pt
├── synth_val_features/BRCA/  # features_synth_val_split_{1-5}.npz
└── synth_shadow/             # alternative pipeline artifacts
    ├── shadow_models/BRCA/   # synth-trained shadow_split_{1-5}.pt
    ├── features/BRCA/        # features from synth shadows
    └── classifiers/          # mlp_best.pt (synth-shadow pipeline)
```
