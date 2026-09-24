#!/usr/bin/env bash
# DP-PGM k/l forest sweep (configs/experiments/pgm_forest_sweep.yaml).
# CPU only, 12 workers on our half of the box (physical cores 12-23 and their
# siblings 36-47); cores 0-11 / 24-35 and GPU 1 stay free for other users.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=/home/golobs/miniconda3/envs/recon_/bin/python
taskset -c 12-23,36-47 $PY scripts/pgm_structure_sweep.py --config configs/experiments/pgm_forest_sweep.yaml --workers 12 "$@"
