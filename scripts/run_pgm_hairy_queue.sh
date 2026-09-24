#!/usr/bin/env bash
# After the free-component hairy-star sweep: the disjoint-pairs version
# (pgm_hairy_pairs.yaml), then the 1-way-only floor (pgm_baselines.yaml).
# Same disk guard and cores as run_pgm_hairy_star.sh.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=/home/golobs/miniconda3/envs/recon_/bin/python
until grep -q "HAIRY STAR COMPLETE" logs/pgm_hairy_star.log; do sleep 120; done
for c in pgm_hairy_pairs pgm_baselines; do
    taskset -c 12-23,36-47 $PY scripts/pgm_structure_sweep.py \
        --config configs/experiments/$c.yaml --workers ${WORKERS:-10} &
    MAIN=$!
    while kill -0 $MAIN 2>/dev/null; do
        free=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)
        pids=$(pgrep -f "$c.yaml")
        if [ "$free" -lt 3 ]; then kill -STOP $pids 2>/dev/null; echo "[guard] paused, ${free}G free"
        elif [ "$free" -gt 4 ]; then kill -CONT $pids 2>/dev/null; fi
        sleep 60
    done
    wait $MAIN
done
echo "HAIRY QUEUE COMPLETE"
