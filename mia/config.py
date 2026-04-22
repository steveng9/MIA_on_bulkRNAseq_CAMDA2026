import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Challenge data paths ─────────────────────────────────────────────────────
CHALLENGE_DIR = "/home/golobs/data/CAMDA26"
BRCA_CHALLENGE_DIR = os.path.join(CHALLENGE_DIR, "RED_TCGA-BRCA")
COMBINED_CHALLENGE_DIR = os.path.join(CHALLENGE_DIR, "RED_TCGA-COMBINED")

# ── NoisyDiffusion repo (has labeled synthetic data + splits YAML) ───────────
NOISY_DIFFUSION_ROOT = os.path.join(PROJECT_ROOT, "..", "CAMDA25_NoisyDiffusion")
BRCA_ND_DIR = os.path.join(NOISY_DIFFUSION_ROOT, "TCGA-BRCA")
COMBINED_ND_DIR = os.path.join(NOISY_DIFFUSION_ROOT, "TCGA-COMBINED")

# ── MIA output ───────────────────────────────────────────────────────────────
MIA_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "mia_output")
SHADOW_MODEL_DIR = os.path.join(MIA_OUTPUT_DIR, "shadow_models")
FEATURES_DIR = os.path.join(MIA_OUTPUT_DIR, "features")
CLASSIFIER_DIR = os.path.join(MIA_OUTPUT_DIR, "classifiers")

# ── Dataset configs ──────────────────────────────────────────────────────────
DATASETS = {
    "BRCA": {
        "challenge_dir": BRCA_CHALLENGE_DIR,
        "real_tsv": os.path.join(BRCA_CHALLENGE_DIR, "TCGA-BRCA_primary_tumor_star_deseq_VST_lmgenes.tsv"),
        "synthetic_csv": os.path.join(BRCA_CHALLENGE_DIR, "synthetic_data_1.csv"),
        "nd_dir": BRCA_ND_DIR,
        "splits_yaml": os.path.join(BRCA_ND_DIR, "TCGA-BRCA_splits.yaml"),
        "nd_synthetic_dir": os.path.join(BRCA_ND_DIR, "synthetic_data"),
        "num_classes": 5,
        "label_list": ["BRCA.Basal", "BRCA.Normal", "BRCA.Her2", "BRCA.LumA", "BRCA.LumB"],
        # Ground-truth subtype labels for real data (BLUE team zip)
        "blue_zip": os.path.join(CHALLENGE_DIR, "BLUE_TCGA-BRCA.zip"),
        "blue_zip_path": "BLUE_TCGA-BRCA/TCGA-BRCA_primary_tumor_subtypes.csv",
        "blue_label_col": "Subtype",
    },
    "COMBINED": {
        "challenge_dir": COMBINED_CHALLENGE_DIR,
        "real_tsv": os.path.join(COMBINED_CHALLENGE_DIR, "TCGA-COMBINED_primary_tumor_star_deseq_VST_lmgenes.tsv"),
        "synthetic_csv": os.path.join(COMBINED_CHALLENGE_DIR, "synthetic_data_1.csv"),
        "reference_tsv": os.path.join(COMBINED_CHALLENGE_DIR, "TCGA-COMBINED_primary_tumor_star_deseq_VST_lmgenes_reference.tsv"),
        "nd_dir": COMBINED_ND_DIR,
        "splits_yaml": os.path.join(COMBINED_ND_DIR, "TCGA-COMBINED_splits.yaml"),
        "nd_synthetic_dir": os.path.join(COMBINED_ND_DIR, "synthetic_data"),
        "num_classes": 12,
        "label_list": [
            "TCGA-KIRC", "TCGA-PRAD", "TCGA-LIHC", "TCGA-ESCA", "TCGA-BRCA", "TCGA-OV",
            "TCGA-LUSC", "TCGA-PAAD", "TCGA-KIRP", "TCGA-LUAD", "TCGA-COAD", "TCGA-SKCM",
        ],
        # Ground-truth subtype labels for real data (BLUE team zip)
        "blue_zip": os.path.join(CHALLENGE_DIR, "BLUE_TCGA-COMBINED.zip"),
        "blue_zip_path": "BLUE_TCGA-COMBINED/TCGA-COMBINED_primary_tumor_subtypes.csv",
        "blue_label_col": "project",
    },
}

# ── Model hyper-parameters (same for both datasets except num_classes) ───────
INPUT_DIM = 978
NUM_TIMESTEPS = 1000
HIDDEN_DIMS = [2048, 2048]
DROPOUT = 0.2
TIME_EMBEDDING_DIM = 128
LABEL_EMBEDDING_DIM = 64
ATTN_NUM_HEADS = 0
ATTN_NUM_TOKENS = 64
NUM_GROUPS = 8
BETA_SCHEDULE = "linear"
LINEAR_BETA_START = 0.001
LINEAR_BETA_END = 0.02
NORM_METHOD = "quantile"

# Shadow-model training (match target)
SHADOW_EPOCHS = 200
SHADOW_BATCH_SIZE = 32
SHADOW_LR = 0.001
SHADOW_LR_WEIGHT_DECAY = 0.001
SHADOW_LR_PCT_START = 0.2
SHADOW_LR_DIV_FACTOR = 25
SHADOW_LR_FINAL_DIV_FACTOR = 25
SHADOW_LR_ANNEAL_STRATEGY = "cos"
SHADOW_EARLY_STOPPING = True
SHADOW_EARLY_STOPPING_PATIENCE = 30
SHADOW_EARLY_STOPPING_MIN_DELTA = 0.0001
#SHADOW_DP_NOISE_MULTIPLIER = 0.00001
SHADOW_DP_NOISE_MULTIPLIER = 0
SHADOW_MAX_GRAD_NORM = 1.0
DUMMY_LABEL = 0
UNCONDITIONAL_NUM_CLASSES = 1

# ── Classifier type ─────────────────────────────────────────────────────────
# "mlp" — 3-layer MLP (default)  |  "rf" — RandomForest (see classifier.RF_PARAMS)
CLASSIFIER_TYPE = "mlp"

# ── Skip / force logic ──────────────────────────────────────────────────────
# Stages that should be forced to re-run even if outputs exist.
# Populated by --force CLI arg.  Valid stage names:
#   "shadows", "features", "classifier", "synth_val", "challenge"
# Use "all" to force everything.
FORCE_STAGES = set()

# ── Loss-feature extraction ──────────────────────────────────────────────────
T_LIST = [5, 10, 20, 30, 40, 50, 100]
N_NOISE = 300

# ── MLP classifier ───────────────────────────────────────────────────────────
MLP_INPUT_DIM = N_NOISE * len(T_LIST)   # 2100
MLP_HIDDEN_DIM = 200
MLP_EPOCHS = 2000
MLP_LR = 1e-4
MLP_BATCH_SIZE = 64
MLP_DROPOUT = 0.0
MLP_WEIGHT_DECAY = 0.0

# ── Feature mode ─────────────────────────────────────────────────────────────
FEATURE_MODE = "raw"   # "raw" or "summary"

# ── Named profiles ───────────────────────────────────────────────────────────
ACTIVE_PROFILE = "baseline"

PROFILES = {
    "baseline": {
        "T_LIST": [5, 10, 20, 30, 40, 50, 100],
        "N_NOISE": 300,
        "FEATURE_MODE": "raw",
        "MLP_HIDDEN_DIM": 200,
        "MLP_DROPOUT": 0.0,
        "MLP_WEIGHT_DECAY": 0.0,
        "MLP_EPOCHS": 750,
        "MLP_LR": 1e-4,
    },
    "tuned": {
        "T_LIST": [1, 2, 5, 10, 20, 50, 100, 200],
        "N_NOISE": 100,
        "FEATURE_MODE": "summary",
        "MLP_HIDDEN_DIM": 64,
        "MLP_DROPOUT": 0.3,
        "MLP_WEIGHT_DECAY": 1e-3,
        "MLP_EPOCHS": 2000,
        "MLP_LR": 1e-4,
    },
}


def apply_profile(name):
    """Apply a named profile, setting module-level variables."""
    import sys
    mod = sys.modules[__name__]
    if name not in PROFILES:
        raise ValueError(f"Unknown profile '{name}'. Choose from: {list(PROFILES.keys())}")
    for key, val in PROFILES[name].items():
        setattr(mod, key, val)
    mod.ACTIVE_PROFILE = name
    # Recompute derived values
    if mod.FEATURE_MODE == "summary":
        mod.MLP_INPUT_DIM = len(mod.T_LIST) * 4  # mean, std, min, max per timestep
    else:
        mod.MLP_INPUT_DIM = mod.N_NOISE * len(mod.T_LIST)

# ── Split configuration ─────────────────────────────────────────────────────
SPLIT_MODE = "custom"  # "custom" or "noisydiffusion"
NUM_CUSTOM_SPLITS = 30
CUSTOM_SPLIT_RATIO = 0.8  # fraction of real data used as "members" per split
CUSTOM_SPLITS_DIR = os.path.join(MIA_OUTPUT_DIR, "splits")

# ── Synthetic validation ────────────────────────────────────────────────────
SYNTH_VAL_MODEL_DIR = os.path.join(MIA_OUTPUT_DIR, "synth_val_models")
SYNTH_VAL_FEATURES_DIR = os.path.join(MIA_OUTPUT_DIR, "synth_val_features")

# ── Synth-shadow variant (shadows trained on synthetic data) ────────────────
SYNTH_SHADOW_OUTPUT_DIR = os.path.join(MIA_OUTPUT_DIR, "synth_shadow")
SYNTH_SHADOW_MODEL_DIR = os.path.join(SYNTH_SHADOW_OUTPUT_DIR, "shadow_models")
SYNTH_SHADOW_FEATURES_DIR = os.path.join(SYNTH_SHADOW_OUTPUT_DIR, "features")
SYNTH_SHADOW_CLASSIFIER_DIR = os.path.join(SYNTH_SHADOW_OUTPUT_DIR, "classifiers")

NUM_SPLITS = 5  # NoisyDiffusion splits (always 5)
DEVICE = "cuda:0"
SEED = 42

# ═══════════════════════════════════════════════════════════════════════════════
# CVAE attack configuration
# ═══════════════════════════════════════════════════════════════════════════════

# ── CVAE model architecture ───────────────────────────────────────────────────
CVAE_INPUT_DIM = 978         # same gene space as ND
CVAE_Z_DIM = 128             # latent space dimensionality
CVAE_BETA = 1.0              # KL weight in ELBO
CVAE_TRANSFORM = "none"      # output activation: "none"|"exp"|"sigmoid"|"relu"
CVAE_CONDITION_TYPE = "onehot"   # "onehot" | "embedding"
CVAE_DISEASE_EMBED_DIM = 20      # only used when CVAE_CONDITION_TYPE == "embedding"

# ── CVAE label / conditioning mode ───────────────────────────────────────────
# "real" — ground-truth subtypes from BLUE team zip (only valid for real data;
#           challenge synthetic data falls back to "knn" automatically)
# "knn"  — predict class labels via KNN on ND synthetic data; use as condition
# "none" — pass zero condition vector, disabling class conditioning entirely
CVAE_LABEL_MODE = "real"

# ── CVAE shadow training ──────────────────────────────────────────────────────
CVAE_EPOCHS = 200
CVAE_BATCH_SIZE = 32
CVAE_LR = 1e-3
CVAE_LR_WEIGHT_DECAY = 1e-3
CVAE_LR_PCT_START = 0.2
CVAE_LR_DIV_FACTOR = 25
CVAE_LR_FINAL_DIV_FACTOR = 25
CVAE_LR_ANNEAL_STRATEGY = "cos"
CVAE_EARLY_STOPPING = True
CVAE_EARLY_STOPPING_PATIENCE = 30
CVAE_EARLY_STOPPING_MIN_DELTA = 1e-4

# ── CVAE loss-feature extraction ──────────────────────────────────────────────
# TEMP_LIST: analogous to T_LIST in ND.  α=0 gives deterministic z=μ.
CVAE_TEMP_LIST = [0.0, 0.5, 1.0, 1.5, 2.0]
CVAE_N_SAMPLES = 300         # K stochastic z draws per temperature (≈ N_NOISE in ND)
CVAE_FEATURE_MODE = "raw"    # "raw" | "summary"

# Derived: total MLP input dim
# raw:     CVAE_N_SAMPLES * len(CVAE_TEMP_LIST) + CVAE_Z_DIM = 300*5+128 = 1628
# summary: 4 * len(CVAE_TEMP_LIST) + CVAE_Z_DIM              = 4*5+128  =  148
CVAE_MLP_INPUT_DIM = CVAE_N_SAMPLES * len(CVAE_TEMP_LIST) + CVAE_Z_DIM  # recomputed by profile

# ── CVAE MLP classifier ───────────────────────────────────────────────────────
CVAE_MLP_HIDDEN_DIM = 200
CVAE_MLP_EPOCHS = 750
CVAE_MLP_LR = 1e-4
CVAE_MLP_BATCH_SIZE = 64
CVAE_MLP_DROPOUT = 0.0
CVAE_MLP_WEIGHT_DECAY = 0.0

# ── CVAE output directories ───────────────────────────────────────────────────
CVAE_OUTPUT_DIR = os.path.join(MIA_OUTPUT_DIR, "cvae")
CVAE_SHADOW_MODEL_DIR = os.path.join(CVAE_OUTPUT_DIR, "shadow_models")
CVAE_FEATURES_DIR = os.path.join(CVAE_OUTPUT_DIR, "features")
CVAE_CLASSIFIER_DIR = os.path.join(CVAE_OUTPUT_DIR, "classifiers")
CVAE_SYNTH_DIR = os.path.join(CVAE_OUTPUT_DIR, "synthetic_data")

CVAE_SYNTH_SHADOW_OUTPUT_DIR = os.path.join(CVAE_OUTPUT_DIR, "synth_shadow")
CVAE_SYNTH_SHADOW_MODEL_DIR = os.path.join(CVAE_SYNTH_SHADOW_OUTPUT_DIR, "shadow_models")
CVAE_SYNTH_SHADOW_FEATURES_DIR = os.path.join(CVAE_SYNTH_SHADOW_OUTPUT_DIR, "features")
CVAE_SYNTH_SHADOW_CLASSIFIER_DIR = os.path.join(CVAE_SYNTH_SHADOW_OUTPUT_DIR, "classifiers")

# ── CVAE named profiles ───────────────────────────────────────────────────────
CVAE_ACTIVE_PROFILE = "baseline"

CVAE_PROFILES = {
    "baseline": {
        "CVAE_TEMP_LIST": [0.0, 0.5, 1.0, 1.5, 2.0],
        "CVAE_N_SAMPLES": 300,
        "CVAE_FEATURE_MODE": "raw",
        "CVAE_MLP_HIDDEN_DIM": 200,
        "CVAE_MLP_DROPOUT": 0.0,
        "CVAE_MLP_WEIGHT_DECAY": 0.0,
        "CVAE_MLP_EPOCHS": 750,
        "CVAE_MLP_LR": 1e-4,
    },
    "tuned": {
        "CVAE_TEMP_LIST": [0.0, 0.5, 1.0, 1.5, 2.0, 3.0],
        "CVAE_N_SAMPLES": 100,
        "CVAE_FEATURE_MODE": "summary",
        "CVAE_MLP_HIDDEN_DIM": 64,
        "CVAE_MLP_DROPOUT": 0.3,
        "CVAE_MLP_WEIGHT_DECAY": 1e-3,
        "CVAE_MLP_EPOCHS": 2000,
        "CVAE_MLP_LR": 1e-4,
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# Unified pipeline configuration  (used by mia.attack + mia.pipeline)
# ═══════════════════════════════════════════════════════════════════════════════

# Number of shadow models for MLP training (K) and internal holdout eval (Q).
# Total models trained per run = K + Q.
DEFAULT_K = 15
DEFAULT_Q = 5

# Unified output tree — completely separate from old per-attack paths above.
# Layout: pipeline/{real_shadows,synth_shadows,synthetic,features,classifiers}/
#                /{nd,cvae}/{dataset}/...
PIPELINE_BASE         = os.path.join(MIA_OUTPUT_DIR, "pipeline")
PIPELINE_REAL_SHADOW  = os.path.join(PIPELINE_BASE, "real_shadows")
PIPELINE_SYNTH_SHADOW = os.path.join(PIPELINE_BASE, "synth_shadows")
PIPELINE_SYNTHETIC    = os.path.join(PIPELINE_BASE, "synthetic")
PIPELINE_FEATURES     = os.path.join(PIPELINE_BASE, "features")
PIPELINE_CLASSIFIERS  = os.path.join(PIPELINE_BASE, "classifiers")
PIPELINE_TARGET_PROXY = os.path.join(PIPELINE_BASE, "target_proxy")


def apply_cvae_profile(name):
    """Apply a named CVAE profile, updating module-level variables."""
    import sys
    mod = sys.modules[__name__]
    if name not in CVAE_PROFILES:
        raise ValueError(f"Unknown CVAE profile '{name}'. Choose from: {list(CVAE_PROFILES.keys())}")
    for key, val in CVAE_PROFILES[name].items():
        setattr(mod, key, val)
    mod.CVAE_ACTIVE_PROFILE = name
    # Recompute derived MLP input dim
    n_t = len(mod.CVAE_TEMP_LIST)
    if mod.CVAE_FEATURE_MODE == "summary":
        mod.CVAE_MLP_INPUT_DIM = 4 * n_t + mod.CVAE_Z_DIM
    else:
        mod.CVAE_MLP_INPUT_DIM = mod.CVAE_N_SAMPLES * n_t + mod.CVAE_Z_DIM
