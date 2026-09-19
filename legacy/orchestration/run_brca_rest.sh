#!/bin/bash
# BRCA: build the remaining targets, then everything except MeLoMIA-ND.
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=0
PY=/home/golobs/miniconda3/envs/recon_/bin/python

echo "### building MVN / CVAE / ND targets"
$PY -u scripts/build_targets.py --dataset BRCA --generators mvn cvae nd

echo "### MeLoMIA-CVAE shadow stack + grid row"
$PY -u scripts/run_experiment.py configs/experiments/grid_brca.yaml \
    --only melomia_cvae --generators mvn cvae nd --device cuda
