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
SHADOW_DP_NOISE_MULTIPLIER = 0.00001
SHADOW_MAX_GRAD_NORM = 1.0
DUMMY_LABEL = 0
UNCONDITIONAL_NUM_CLASSES = 1

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
MLP_EPOCHS = 750
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
NUM_CUSTOM_SPLITS = 15
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
