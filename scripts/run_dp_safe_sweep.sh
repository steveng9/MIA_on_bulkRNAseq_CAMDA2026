#!/usr/bin/env bash
# The epsilon curve for every DP-valid binning, then the attack-binning study.
#
# 3 binnings x 9 epsilons x 2 cohorts x 5 splits = 270 targets (20 already
# built).  Targets are cached by name, so this resumes if interrupted.
# 12 workers on cores 12-23 (+ siblings), leaving 12 physical cores free.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD
CORES=12-23,36-47
taskset -c $CORES conda run --no-capture-output -n recon_ python scripts/pgm_eps_sweep.py \
    --workers 12 --threads 2 \
    --variants "binning=uniform" "binning=dp_quantile" "binning=dp_uniform"
echo "DP-SAFE SWEEP COMPLETE $(date -Is)"
taskset -c $CORES conda run --no-capture-output -n recon_ python scripts/pgm_attack_binning.py \
    --workers 12
echo "ALL COMPLETE $(date -Is)"
