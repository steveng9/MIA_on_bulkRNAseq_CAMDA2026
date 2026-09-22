#!/usr/bin/env bash
# The 4-attack DP-PGM column on the DP-valid binnings, on GPU 0.
# Waits for the CPU sweep to finish building the eps=10 targets first.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=0
until grep -q "DP-SAFE SWEEP COMPLETE" logs/dp_safe_sweep.log; do sleep 300; done
for ds in brca combined; do
    taskset -c 12-23,36-47 conda run --no-capture-output -n recon_ \
        python run_experiment.py grid_dpsafe_$ds.yaml \
        --only mahalamia mamamia melomia_nd melomia_cvae
done
echo "DPSAFE GRID COMPLETE $(date -Is)"
