#!/bin/bash
# DP-PGM targets are CPU-bound (Private-PGM / mbi), so they run alongside the GPU work.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/golobs/miniconda3/envs/recon_/bin/python
$PY -u scripts/build_targets.py --dataset BRCA --generators pgm --force
$PY -u scripts/build_targets.py --dataset COMBINED --generators pgm
