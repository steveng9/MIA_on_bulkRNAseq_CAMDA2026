#!/usr/bin/env bash
# TODO 8, COMBINED arm: synth-shadow vs real-data shadows (EXPERIMENTS.tex open item).
# GPU 0 only -- GPU 1 stays free for other users -- and our half of the cores.
# MIA_N_JOBS is deliberately small: the DP-safe epsilon sweep holds the same
# cores until it finishes, and a running process cannot be retuned afterwards.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=0 MIA_N_JOBS=8
PY=/home/golobs/miniconda3/envs/recon_/bin/python
taskset -c 12-23,36-47 $PY scripts/run_experiment.py \
    configs/experiments/ablation_synth_shadow_combined.yaml
echo "SYNTH SHADOW COMBINED COMPLETE $(date -Is)"
