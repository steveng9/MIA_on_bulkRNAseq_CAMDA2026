#!/usr/bin/env bash
# DP-PGM "star with hairs" sweep (configs/experiments/pgm_hairy_star.yaml).
# CPU only, on our half of the box (cores 12-23 / 36-47).  ~3 GB of targets on
# a shared disk: a watcher pauses the sweep (SIGSTOP) below 3 GB free and
# resumes it above 4 GB.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=/home/golobs/miniconda3/envs/recon_/bin/python
taskset -c 12-23,36-47 $PY scripts/pgm_structure_sweep.py \
    --config configs/experiments/pgm_hairy_star.yaml --workers ${WORKERS:-8} "$@" &
MAIN=$!
while kill -0 $MAIN 2>/dev/null; do
    free=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)
    pids=$(pgrep -f "pgm_hairy_star.yaml")
    if [ "$free" -lt 3 ]; then kill -STOP $pids 2>/dev/null; echo "[guard] paused, ${free}G free"
    elif [ "$free" -gt 4 ]; then kill -CONT $pids 2>/dev/null; fi
    sleep 60
done
wait $MAIN
echo "HAIRY STAR COMPLETE"
