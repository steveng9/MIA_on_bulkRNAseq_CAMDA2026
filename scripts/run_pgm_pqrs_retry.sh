#!/usr/bin/env bash
# PQRS k-way retry under zCDP (configs/experiments/pgm_pqrs_retry.yaml).  Waits
# for the MAMA-MIA v2 shadow evaluation to free its 8 workers, then runs on our
# half of the box only (cores 12-23 / 36-47); other users keep 0-11 / 24-35.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=/home/golobs/miniconda3/envs/recon_/bin/python
until grep -q "MAMAMIA V2 COMPLETE" logs/mamamia_v2_shadow.log; do sleep 120; done
taskset -c 12-23,36-47 $PY scripts/pgm_structure_sweep.py \
    --config configs/experiments/pgm_pqrs_retry.yaml --workers 8 "$@"
