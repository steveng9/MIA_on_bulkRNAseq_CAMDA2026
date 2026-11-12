#!/bin/bash
# TabSyn arms of the normalisation ablation (BRCA, 5 splits) on GPU 0, started only
# once the TabPFN COMBINED queue has released the GPU (TabSyn on CPU is days).
cd ~/MIA_on_bulkRNAseq_CAMDA2026
PY=/home/golobs/miniconda3/envs/recon_/bin/python
L=logs/paper_fill
stamp() { echo "### [$(date -u +%m-%d\ %H:%M)] $*" | tee -a $L/progress.log; }
until grep -qE "TABPFN_COMBINED_DONE|TabPFN COMBINED (queue stopped|split [0-9] FAILED)" $L/progress.log; do sleep 300; done
until grep -q "PREPROCESS_ABLATION_DONE" $L/progress.log; do sleep 300; done
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MIA_N_JOBS=4
nice -n 5 taskset -c 16-19 $PY -W ignore scripts/preprocess_ablation.py --dataset BRCA --generator tabsyn \
  --set class_freq=train diff_schedule=steps latent_scale=std \
  --specs clip:0.001:0.999+standard standard quantile standard+pca:64+standard --splits 1 2 3 4 5 --device cuda \
  > $L/preprocess_tabsyn_BRCA_gpu.log 2>&1
stamp "preprocess ablation: TabSyn BRCA built (GPU 0)"
nice -n 5 taskset -c 12-19 $PY -W ignore scripts/run_experiment.py configs/experiments/paper_preprocess_brca.yaml \
  --generators "tabsyn@class_freq=train,diff_schedule=steps,latent_scale=std,preprocess=standard" \
  "tabsyn@class_freq=train,diff_schedule=steps,latent_scale=std,preprocess=quantile" \
  "tabsyn@class_freq=train,diff_schedule=steps,latent_scale=std,preprocess=clip:0.001:0.999+standard" \
  "tabsyn@class_freq=train,diff_schedule=steps,latent_scale=std,preprocess=standard+pca:64+standard" \
  > $L/preprocess_attacks_tabsyn.log 2>&1
stamp "PREPROCESS_TABSYN_DONE"
