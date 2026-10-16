#!/bin/bash
# COMBINED: existing attacks on the tuned-TabSyn targets (five splits) and the one
# TabPFN target (split 1).  MeLoMIA-ND / MeLoMIA-CVAE use the K=20 calibrated stacks
# of melomia_prc_combined.yaml.  GPU 0 only (Steven, 2026-10-10).
cd ~/MIA_on_bulkRNAseq_CAMDA2026
PY=/home/golobs/miniconda3/envs/recon_/bin/python
T="tabsyn@class_freq=train,diff_schedule=steps,latent_scale=std,preprocess=clip:0.001:0.999+standard"
L=logs/sota/combined_scoring
CHEAP="mahalamia mamamia generic_mc generic_gan_leaks generic_conf_lr generic_conf_rf"
export CUDA_VISIBLE_DEVICES=0
run() { taskset -c 8-23 $PY -W ignore scripts/run_experiment.py "$@"; }
run configs/experiments/grid_sota_combined.yaml --only $CHEAP --generators "$T" > $L/cheap_tabsyn.log 2>&1
run configs/experiments/grid_sota_combined.yaml --only $CHEAP --generators tabpfn --splits 1 > $L/cheap_tabpfn.log 2>&1
run configs/experiments/melomia_prc_combined.yaml --generators "$T" > $L/melomia_tabsyn_target.log 2>&1
run configs/experiments/melomia_prc_combined.yaml --generators tabpfn --splits 1 > $L/melomia_tabpfn_target.log 2>&1
echo "### [$(date -u +%m-%d\ %H:%M)] COMBINED_SOTA_SCORING_DONE" >> $L/progress.log
