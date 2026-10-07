#!/bin/sh
# Re-analyse the saved donor-linked scores with the "separate tumour sample"
# group (re-sequenced aliquots excluded), and run DOMIAS on all 978 genes.
# Nothing is retrained.   usage: sh scripts/perturb/run_regroup.sh <log dir>
PY=/home/golobs/miniconda3/envs/recon_/bin/python
export CAMDA_ARTIFACTS=/home/golobs/MIA_on_bulkRNAseq_CAMDA2026/artifacts OMP_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=""
L=${1:-.}
taskset -c 12-17 $PY -W ignore scripts/perturb/domias_full.py > $L/domias_full.log 2>&1 &
taskset -c 18-23 $PY -W ignore scripts/perturb/donor_linked_report.py > $L/rep.log 2>&1 &
taskset -c 36-41 $PY -W ignore scripts/perturb/donor_linked_report.py --suffix=_melomia > $L/rep_melo.log 2>&1 &
(taskset -c 42-47 $PY -W ignore scripts/perturb/donor_linked_resplit.py --dataset COMBINED --reps 100 --reuse
 taskset -c 42-47 $PY -W ignore scripts/perturb/donor_linked_resplit.py --dataset BRCA --reps 100 --reuse) > $L/resplit.log 2>&1 &
wait
echo REGROUP_DONE
