#!/usr/bin/env bash
# Fill every cell of the group-meeting slide table (Steven, 2026-09-27).
#   A. build DP-PGM old (5 splits) and DP-PGM new splits 4-5, both cohorts, in
#      parallel on CPU; meanwhile run v2 + generic baselines on MVN/CVAE/ND
#   B. every attack on old and new DP-PGM: CPU attacks, then MeLoMIA on GPU 0
#   C. fidelity rows and the table (scripts/slide_table.py)
# Our cores (12-23, 36-47) and GPU 0 only.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD MIA_N_JOBS=10
PY=/home/golobs/miniconda3/envs/recon_/bin/python
PIN="taskset -c 12-23,36-47"
OLD="pgm@composition=basic,neighboring=legacy_exact_n"
NEW="pgm@binning=dp_quantile,edge_estimator=threshold,n_bins=16"
mkdir -p logs/slides

echo "=== A: builds $(date -Is) ==="
for ds in BRCA COMBINED; do
  for s in 1 2 3 4 5; do
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 $PIN $PY scripts/build_targets.py --dataset $ds \
      --generators "$OLD" --splits $s --device cpu > logs/slides/build_${ds}_old_s$s.log 2>&1 &
  done
  for s in 4 5; do
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 $PIN $PY scripts/build_targets.py --dataset $ds \
      --generators "$NEW" --splits $s --device cpu > logs/slides/build_${ds}_new_s$s.log 2>&1 &
  done
done
for ds in brca combined; do
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 $PIN $PY scripts/run_experiment.py \
    configs/experiments/grid_slides_extra_$ds.yaml
done
wait
echo "=== builds done $(date -Is) ==="
grep -l "Error\|Traceback" logs/slides/build_*.log && echo "BUILD FAILURES ABOVE"

echo "=== B: attacks $(date -Is) ==="
for ds in brca combined; do
  gen=$(grep -o "generic_[a-z_0-9]*" configs/experiments/grid_slides_$ds.yaml | sort -u | tr '\n' ' ')
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 $PIN $PY scripts/run_experiment.py \
    configs/experiments/grid_slides_$ds.yaml --only mahalamia mamamia mamamia_v2 $gen
done
for ds in brca combined; do
  CUDA_VISIBLE_DEVICES=0 $PIN $PY scripts/run_experiment.py \
    configs/experiments/grid_slides_$ds.yaml --only melomia_nd melomia_cvae
done

echo "=== C: table $(date -Is) ==="
CUDA_VISIBLE_DEVICES= $PIN $PY scripts/slide_table.py
echo "SLIDE FILL COMPLETE $(date -Is)"
