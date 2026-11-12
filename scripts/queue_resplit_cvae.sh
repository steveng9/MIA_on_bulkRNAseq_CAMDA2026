#!/bin/bash
# Calibrated second-sample (donor-linked resplit) design for the CVAE, default MahalaMIA,
# 60 redrawn splits each, CPU cores 16-19 (2026-11-12; MVN was done with 100 reps).
cd ~/MIA_on_bulkRNAseq_CAMDA2026
PY=/home/golobs/miniconda3/envs/recon_/bin/python
L=logs/paper_fill
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
stamp() { echo "### [$(date -u +%m-%d\ %H:%M)] $*" | tee -a $L/progress.log; }
for ds in COMBINED BRCA; do
  nice -n 5 taskset -c 16-19 $PY -W ignore scripts/perturb/donor_linked_resplit.py --dataset $ds --generator cvae \
    --reps 60 --device cpu > $L/resplit_cvae_$ds.log 2>&1
  stamp "default MahalaMIA: calibrated second sample, CVAE $ds (60 reps) done"
done
stamp "RESPLIT_CVAE_DONE"
