#!/usr/bin/env bash
# Calibrated second-sample attack with the paper's MahalaMIA (2026-11-12).
# Three jobs, four cores each, on cores 12-23.
cd "$(dirname "$0")/../.."
PY=/home/golobs/miniconda3/envs/recon_/bin/python
L=logs/paper_fill
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
taskset -c 12-15 $PY scripts/perturb/donor_linked_resplit.py --dataset BRCA --generator mvn --reps 100 > $L/default_resplit_BRCA.log 2>&1 &
taskset -c 16-19 $PY scripts/perturb/donor_linked_resplit.py --dataset COMBINED --generator mvn --reps 100 > $L/default_resplit_COMBINED.log 2>&1 &
taskset -c 20-23 $PY scripts/perturb/gtex_donor_linked_resplit.py --reps 100 > $L/default_resplit_GTEX.log 2>&1 &
wait
echo "### [$(date -u +%m-%d\ %H:%M)] default MahalaMIA: calibrated second-sample (resplit) done" >> $L/progress.log
