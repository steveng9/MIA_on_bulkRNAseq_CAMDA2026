#!/bin/sh
# Donor-linked MeLoMIA on each backend's own generator, both cohorts.
# usage: run_donor_linked_melomia.sh <backend: nd|cvae> <gpu>
PY=/home/golobs/miniconda3/envs/recon_/bin/python
export CUDA_VISIBLE_DEVICES=$2 OMP_NUM_THREADS=4 MIA_N_JOBS=4
for ds in BRCA COMBINED; do
  taskset -c 12-23,36-47 $PY -W ignore scripts/perturb/donor_linked_melomia.py \
      --dataset $ds --backend $1 --generators $1
done
echo MELOMIA_DONE_$1
