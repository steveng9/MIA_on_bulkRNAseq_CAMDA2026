#!/usr/bin/env bash
# DP-PGM as the CAMDA-26 challenge ran it (Steven, 2026-09-30): build the
# CAMDA-25 winner's generator (mia/generators/pgg.py) on splits 1-5 of both
# cohorts, run every slide attack on it, rebuild the slide table.
# Our cores (12-23, 36-47) and GPU 0 only.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD MIA_N_JOBS=10
PY=/home/golobs/miniconda3/envs/recon_/bin/python
PIN="taskset -c 12-23,36-47"
mkdir -p logs/slides

echo "=== builds $(date -Is) ==="
for ds in BRCA COMBINED; do
  for s in 1 2 3 4 5; do
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 $PIN $PY -W ignore scripts/build_targets.py --dataset $ds \
      --generators pgg --splits $s --device cpu > logs/slides/build_${ds}_pgg_s$s.log 2>&1 &
  done
done
wait
echo "=== builds done $(date -Is) ==="
grep -l "Error\|Traceback" logs/slides/build_*_pgg_*.log && echo "BUILD FAILURES ABOVE"

echo "=== attacks $(date -Is) ==="
for ds in brca combined; do
  gen=$(grep -o "generic_[a-z_0-9]*" configs/experiments/grid_slides_pgg_$ds.yaml | sort -u | tr '\n' ' ')
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 $PIN $PY scripts/run_experiment.py \
    configs/experiments/grid_slides_pgg_$ds.yaml --only mahalamia mamamia mamamia_v2 $gen
done
echo "=== CPU attacks done $(date -Is) ==="
for ds in brca combined; do
  CUDA_VISIBLE_DEVICES=0 $PIN $PY scripts/run_experiment.py \
    configs/experiments/grid_slides_pgg_$ds.yaml --only melomia_nd melomia_cvae
done
echo "PGG FILL COMPLETE $(date -Is)"
