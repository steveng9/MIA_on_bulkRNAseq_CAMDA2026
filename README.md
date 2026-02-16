# MIA_on_bulkRNAseq_CAMDA2016


  Implementation Summary                                                                      
                                                                                              
  6 modules in mia/:                                                                          
                                                                                              
  Module: config.py

  Purpose: All paths, hyperparameters, dataset configs (BRCA + COMBINED)       

  ────────────────────────────────────────

  Module: data_utils.py

  Purpose: Load challenge data (transposed TSV → samples×genes), NoisyDiffusion repo data,
    splits YAML for ground truth, KNN label inference

  ────────────────────────────────────────

  Module: shadow_model.py

  Purpose: Train/load shadow EmbeddedDiffusion models on labeled synthetic data (SMOTE +
    QuantileTransformer + OneCycleLR + DP noise)

  ────────────────────────────────────────

  Module: loss_features.py

  Purpose: Extract loss-trajectory features: 300 noise vectors × 7 timesteps = 2100 features
    per sample

  ────────────────────────────────────────
 
 Module: classifier.py

 Purpose: 3-layer MLP (2100→200→200→1, tanh+sigmoid), trained with BCE, saved by best
    TPR@10%FPR

  ────────────────────────────────────────

  Module: attack.py

  Purpose: End-to-end pipeline: train shadows → extract features → train MLP → evaluate
    per-split → output challenge predictions

  Key design changes from the original plan:
  - Real data is a genes×samples TSV (transposed on load), not pre-split CSVs
  - No separate label files — subtype labels are inferred via KNN on NoisyDiffusion's labeled
  synthetic data
  - Both BRCA (5 classes) and COMBINED (12 classes) datasets are supported
  - Ground truth membership comes from the NoisyDiffusion splits YAML (train_index/test_index)
  - Challenge predictions are saved as synthetic_data_1_predictions.csv

  To run: python -m mia.attack --dataset BRCA --device cuda:0