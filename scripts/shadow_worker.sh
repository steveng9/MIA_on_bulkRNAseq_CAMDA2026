#!/bin/bash
# One shadow-building worker for a MeLoMIA run.
#   shadow_worker.sh <gpu> <config> <attack-label> <shadow indices...>
# Workers share one cache directory and claim individual shadows with lock
# files, so several can run on different GPUs without stepping on each other.
set -euo pipefail
cd "$(dirname "$0")/.."
GPU=$1; CONFIG=$2; LABEL=$3; shift 3
export CUDA_VISIBLE_DEVICES=$GPU
/home/golobs/miniconda3/envs/recon_/bin/python -u scripts/run_experiment.py \
    "$CONFIG" --only "$LABEL" --prepare-only --no-meta --shadows "$@"
