#!/usr/bin/env bash
# The 4-attack DP-PGM column on the DP-valid binnings, on GPU 0.
# Waits for the CPU sweep to build the eps=10 targets, and for the COMBINED
# synth-shadow arm to release the GPU.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=0 MIA_N_JOBS=10
PY=/home/golobs/miniconda3/envs/recon_/bin/python
until grep -q "DP-SAFE SWEEP COMPLETE" logs/dp_safe_sweep.log \
   && grep -q "SYNTH SHADOW COMBINED COMPLETE" logs/synth_shadow_combined.log; do
    sleep 300
done
for ds in brca combined; do
  echo "=== grid_dpsafe_$ds $(date -Is) ==="
  taskset -c 12-23,36-47 $PY scripts/run_experiment.py \
      configs/experiments/grid_dpsafe_$ds.yaml \
      --only mahalamia mamamia melomia_nd melomia_cvae
done
echo "DPSAFE GRID COMPLETE $(date -Is)"
