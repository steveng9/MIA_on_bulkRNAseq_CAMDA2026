#!/bin/bash
# Restart after the disk filled on 2026-11-12 ~09:35 UTC (every queue died with ENOSPC).
#  A (12-15) + B (16-19): remaining CVAE COMBINED arms, then one cached pass so every
#      fidelity row is written by a single process, then attack scoring of both cohorts.
#  G (GPU 0, cores 40-43): TabSyn arms on BRCA, scored; then TabPFN COMBINED splits 3-5.
#  R (36-39): calibrated second sample for the CVAE, COMBINED then BRCA.
cd ~/MIA_on_bulkRNAseq_CAMDA2026
PY=/home/golobs/miniconda3/envs/recon_/bin/python
L=logs/paper_fill
stamp() { echo "### [$(date -u +%m-%d\ %H:%M)] $*" | tee -a $L/progress.log; }
CPU="env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MIA_N_JOBS=4 nice -n 5"
ALL="standard quantile minmax robust clip:0.001:0.999+standard classcenter+standard standard+pca:64"
TS="class_freq=train diff_schedule=steps latent_scale=std"
stamp "restart after disk full: preprocess ablation, TabSyn arms, CVAE resplit, then TabPFN 3-5"
(
  $CPU taskset -c 12-15 $PY -W ignore scripts/preprocess_ablation.py --dataset COMBINED --generator cvae \
    --specs minmax robust clip:0.001:0.999+standard --splits 1 2 3 4 5 --device cpu > $L/preprocess_cvae_COMBINED_a.log 2>&1 &
  $CPU taskset -c 16-19 $PY -W ignore scripts/preprocess_ablation.py --dataset COMBINED --generator cvae \
    --specs classcenter+standard standard+pca:64 --splits 1 2 3 4 5 --device cpu > $L/preprocess_cvae_COMBINED_b.log 2>&1 &
  wait
  for ds in BRCA COMBINED; do
    $CPU taskset -c 12-19 $PY -W ignore scripts/preprocess_ablation.py --dataset $ds --generator cvae \
      --specs $ALL --splits 1 2 3 4 5 --device cpu > $L/preprocess_cvae_${ds}_final.log 2>&1
  done
  stamp "preprocess ablation: CVAE arms built and fidelity scored (both cohorts)"
  for ds in brca combined; do
    $CPU taskset -c 12-19 $PY -W ignore scripts/run_experiment.py configs/experiments/paper_preprocess_$ds.yaml \
      --generators cvae "cvae@preprocess=quantile" "cvae@preprocess=minmax" "cvae@preprocess=robust" \
      "cvae@preprocess=clip:0.001:0.999+standard" "cvae@preprocess=classcenter+standard" "cvae@preprocess=standard+pca:64" \
      > $L/preprocess_attacks_$ds.log 2>&1
    stamp "preprocess ablation: CVAE attacks $ds done"
  done
  stamp "PREPROCESS_CVAE_DONE"
) &
(
  env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MIA_N_JOBS=4 nice -n 5 \
    taskset -c 40-43 $PY -W ignore scripts/preprocess_ablation.py --dataset BRCA --generator tabsyn --set $TS \
    --specs clip:0.001:0.999+standard standard quantile standard+pca:64+standard --splits 1 2 3 4 5 --device cuda \
    > $L/preprocess_tabsyn_BRCA_gpu.log 2>&1
  stamp "preprocess ablation: TabSyn BRCA built (GPU 0)"
  $CPU taskset -c 40-43 $PY -W ignore scripts/run_experiment.py configs/experiments/paper_preprocess_brca.yaml \
    --generators "tabsyn@class_freq=train,diff_schedule=steps,latent_scale=std,preprocess=standard" \
    "tabsyn@class_freq=train,diff_schedule=steps,latent_scale=std,preprocess=quantile" \
    "tabsyn@class_freq=train,diff_schedule=steps,latent_scale=std,preprocess=clip:0.001:0.999+standard" \
    "tabsyn@class_freq=train,diff_schedule=steps,latent_scale=std,preprocess=standard+pca:64+standard" \
    > $L/preprocess_attacks_tabsyn.log 2>&1
  stamp "PREPROCESS_TABSYN_DONE"
  SPLITS="3 4 5" bash scripts/queue_tabpfn_combined.sh
) &
(
  for ds in COMBINED BRCA; do
    $CPU taskset -c 36-39 $PY -W ignore scripts/perturb/donor_linked_resplit.py --dataset $ds --generator cvae \
      --reps 60 --device cpu > $L/resplit_cvae_$ds.log 2>&1 && \
      stamp "default MahalaMIA: calibrated second sample, CVAE $ds (60 reps) done" || stamp "resplit CVAE $ds FAILED"
  done
  stamp "RESPLIT_CVAE_DONE"
) &
wait
