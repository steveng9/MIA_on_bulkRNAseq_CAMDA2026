#!/bin/bash
# Fill the blank cells of the paper grid (2026-11-11): the default MahalaMIA,
# MAMA-MIA and RedSigma against TabSyn and TabPFN, the reference-set generic
# attacks against them on TCGA-COMBINED, and every attack on the one TabPFN
# TCGA-BRCA split that was never scored.  GPU 0 only, 8 logical cores.
cd ~/MIA_on_bulkRNAseq_CAMDA2026
PY=/home/golobs/miniconda3/envs/recon_/bin/python
L=logs/paper_fill
T="tabsyn@class_freq=train,diff_schedule=steps,latent_scale=std,preprocess=clip:0.001:0.999+standard"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MIA_N_JOBS=8
stamp() { echo "### [$(date -u +%m-%d\ %H:%M)] $*" | tee -a $L/progress.log; }
run() { nice -n 5 taskset -c 20-23,44-47 $PY -W ignore scripts/run_experiment.py "$@"; }
CHEAP="mahalamia_ridge mamamia_v2 redsigma"
run configs/experiments/paper_grid_brca.yaml --only $CHEAP --generators "$T" tabpfn > $L/brca_cheap.log 2>&1
stamp "BRCA cheap attacks on TabSyn / TabPFN"
run configs/experiments/paper_grid_combined.yaml --only $CHEAP generic_logan_d1 generic_gan_leaks_cal generic_domias_kde \
  --generators "$T" tabpfn > $L/combined_cheap.log 2>&1
stamp "COMBINED cheap attacks on TabSyn / TabPFN"
run configs/experiments/paper_grid_brca.yaml --only generic_mc generic_gan_leaks generic_conf_lr generic_conf_rf melomia_nd melomia_cvae \
  --generators tabpfn --splits 3 > $L/brca_tabpfn_s3.log 2>&1
stamp "BRCA TabPFN split 3, remaining attacks"
stamp "PAPER_FILL_DONE"
