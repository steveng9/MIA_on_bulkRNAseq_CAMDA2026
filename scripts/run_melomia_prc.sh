#!/usr/bin/env bash
# MeLoMIA with per-record calibration: classifier search + scoring on the slide-table
# generators, from the cached shadow stacks and proxy features (nothing is retrained).
# Two streams (one per cohort), 12 threads each, on the cores and GPU we are allowed.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD
PY=/home/golobs/miniconda3/envs/recon_/bin/python
run() {  # cohort cores
  for a in melomia_nd melomia_cvae; do
    CUDA_VISIBLE_DEVICES=0 MIA_N_JOBS=12 OMP_NUM_THREADS=12 taskset -c $2 $PY -W ignore \
      scripts/run_experiment.py configs/experiments/melomia_prc_$1.yaml --only $a \
      > logs/melomia_prc_$1_$a.log 2>&1
    echo "=== $1 $a done $(date -Is)"
  done
}
run brca 12-17,36-41 &
run combined 18-23,42-47 &
wait
echo "MELOMIA PRC COMPLETE $(date -Is)"
