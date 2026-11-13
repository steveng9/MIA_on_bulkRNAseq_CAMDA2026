#!/bin/bash
# Two gaps in the white-box study and the normalisation ablation (2026-11-13).
#   lane W (CPU 16-17): white-box MahalaMIA and MAMA-MIA on corrected DP-PGM, TCGA-BRCA
#          splits 4-5.  They failed on the first pass because those targets were saved
#          without their model; scripts/whitebox/materialise_pgm_models.py has since
#          rebuilt and verified the models (logs/whitebox/verify_pgm_*.log).
#   lane A (GPU 0, CPU 40-43): the TabSyn "standardise, 64 PCs" arm.  At the TabSyn
#          defaults its denoiser died on split 3 (final loss 1.0), so the arm had 2 of 5
#          splits.  It is rebuilt on all five splits with the stabilised denoiser used
#          for the MeLoMIA probes (lr 3e-4, 1,000-step warm-up, clip 1; see
#          docs/SOTA_GENERATORS.md, "Probe collapse"), so the arm is one recipe, then
#          scored with the paper's shadow-free attacks.
# Restart-safe: builds skip existing targets; results append to results/index.csv.
cd ~/MIA_on_bulkRNAseq_CAMDA2026
PY=/home/golobs/miniconda3/envs/recon_/bin/python
L=logs/paper_fill
PGM="pgm@binning=dp_quantile,edge_estimator=threshold,n_bins=16"
PC64="tabsyn@class_freq=train,diff_clip=1.0,diff_lr=0.0003,diff_schedule=steps,diff_warmup=1000,latent_scale=std,preprocess=standard+pca:64+standard"
stamp() { echo "### [$(date -u +%m-%d\ %H:%M)] $*" | tee -a $L/progress.log; }
env2() { env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 "$@"; }

(   # lane W
  for arm in mahalamia_wb mamamia_v2_wb_model; do
    env2 CUDA_VISIBLE_DEVICES=0 nice -n 5 taskset -c 16-17 $PY -W ignore scripts/run_experiment.py \
      configs/experiments/whitebox_brca.yaml --only $arm --generators "$PGM" --splits 4 5 \
      > logs/whitebox/${arm}_brca_pgm_s45.log 2>&1 \
      && stamp "white box: $arm BRCA DP-PGM splits 4-5 done" || stamp "white box: $arm BRCA DP-PGM splits 4-5 FAILED"
  done
) &

(   # lane A
  env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MIA_N_JOBS=4 nice -n 5 \
    taskset -c 40-43 $PY -W ignore scripts/preprocess_ablation.py --dataset BRCA --generator tabsyn \
    --set class_freq=train diff_schedule=steps latent_scale=std diff_lr=0.0003 diff_warmup=1000 diff_clip=1.0 \
    --specs standard+pca:64+standard --splits 1 2 3 4 5 --device cuda > $L/preprocess_tabsyn_pc64_stable.log 2>&1 \
    || { stamp "preprocess ablation: TabSyn 64-PC (stabilised) build FAILED"; exit 1; }
  stamp "preprocess ablation: TabSyn 64-PC (stabilised) built, 5 splits"
  env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MIA_N_JOBS=4 nice -n 5 \
    taskset -c 40-43 $PY -W ignore scripts/run_experiment.py configs/experiments/paper_preprocess_brca.yaml \
    --generators "$PC64" > $L/preprocess_attacks_tabsyn_pc64_stable.log 2>&1 \
    && stamp "PREPROCESS_TABSYN_PC64_DONE" || stamp "preprocess ablation: TabSyn 64-PC (stabilised) attacks FAILED"
) &
wait
