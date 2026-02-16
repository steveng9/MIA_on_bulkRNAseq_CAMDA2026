import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Challenge data paths ─────────────────────────────────────────────────────
CHALLENGE_DIR = "/Users/stevengolob/Documents/school/CAMDA-26"
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
SMOTE_UPSAMPLE_TO = 3000

# ── Loss-feature extraction ──────────────────────────────────────────────────
T_LIST = [5, 10, 20, 30, 40, 50, 100]
N_NOISE = 300

# ── MLP classifier ───────────────────────────────────────────────────────────
MLP_INPUT_DIM = N_NOISE * len(T_LIST)   # 2100
MLP_HIDDEN_DIM = 200
MLP_EPOCHS = 750
MLP_LR = 1e-4
MLP_BATCH_SIZE = 64

NUM_SPLITS = 5
DEVICE = "cuda:0"
SEED = 42
