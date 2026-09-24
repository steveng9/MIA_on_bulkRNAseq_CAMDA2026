#!/usr/bin/env bash
# PQRS part 2 (configs/experiments/pgm_pqrs_maxdeg*.yaml): the upstream
# max_degree cap.  One job at a time, since fits can take several GiB; our half
# of the box only (cores 12-23 / 36-47).
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PY=/home/golobs/miniconda3/envs/recon_/bin/python
for c in pgm_pqrs_maxdeg16 pgm_pqrs_maxdeg; do
    taskset -c 12-23,36-47 $PY scripts/pgm_structure_sweep.py \
        --config configs/experiments/$c.yaml --workers 2 "$@"
done
echo "PQRS MAXDEG COMPLETE"
