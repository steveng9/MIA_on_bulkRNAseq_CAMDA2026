#!/usr/bin/env bash
# The 4-attack grid on DP-PGM at eps=1000: the three DP-valid binnings plus the
# legacy binning as a reference.  GPU 0 only; GPU 1 stays free for other users.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=0 MIA_N_JOBS=10
PY=/home/golobs/miniconda3/envs/recon_/bin/python
for ds in brca combined; do
  echo "=== grid_dpsafe_eps1000_$ds $(date -Is) ==="
  taskset -c 12-23,36-47 $PY scripts/run_experiment.py \
      configs/experiments/grid_dpsafe_eps1000_$ds.yaml \
      --only mahalamia mamamia melomia_nd melomia_cvae
done
echo "EPS1000 GRID COMPLETE $(date -Is)"
