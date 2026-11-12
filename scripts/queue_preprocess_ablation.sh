#!/bin/bash
# Normalisation ablation for the paper (2026-11-12), CPU only (GPU 0 holds TabPFN).
# Lane A (cores 12-15): CVAE under six preprocessing specs, BRCA then COMBINED.
# Lane B (cores 16-19): TabSyn (tuned recipe) under four specs, BRCA.
# Then scores every arm with the shadow-free paper attacks.  Stamps in logs/paper_fill/progress.log.
cd ~/MIA_on_bulkRNAseq_CAMDA2026
PY=/home/golobs/miniconda3/envs/recon_/bin/python
L=logs/paper_fill
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MIA_N_JOBS=4
stamp() { echo "### [$(date -u +%m-%d\ %H:%M)] $*" | tee -a $L/progress.log; }
CVAE_SPECS="standard quantile minmax robust clip:0.001:0.999+standard classcenter+standard standard+pca:64"
(
  for ds in BRCA COMBINED; do
    nice -n 5 taskset -c 12-15 $PY -W ignore scripts/preprocess_ablation.py --dataset $ds --generator cvae \
      --specs $CVAE_SPECS --splits 1 2 3 4 5 --device cpu > $L/preprocess_cvae_$ds.log 2>&1
    stamp "preprocess ablation: CVAE $ds built and fidelity scored"
  done
) &
(
  nice -n 5 taskset -c 16-19 $PY -W ignore scripts/preprocess_ablation.py --dataset BRCA --generator tabsyn \
    --set class_freq=train diff_schedule=steps latent_scale=std \
    --specs clip:0.001:0.999+standard standard quantile standard+pca:64+standard --splits 1 2 3 4 5 --device cpu \
    > $L/preprocess_tabsyn_BRCA.log 2>&1
  stamp "preprocess ablation: TabSyn BRCA built and fidelity scored"
) &
wait
for ds in brca combined; do
  nice -n 5 taskset -c 12-19 $PY -W ignore scripts/run_experiment.py configs/experiments/paper_preprocess_$ds.yaml \
    > $L/preprocess_attacks_$ds.log 2>&1
  stamp "preprocess ablation: attacks $ds done"
done
stamp "PREPROCESS_ABLATION_DONE"
