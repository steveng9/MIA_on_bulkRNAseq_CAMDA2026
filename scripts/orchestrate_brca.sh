#!/bin/bash
# Sequencing for the BRCA grid on a two-GPU box.
#
# GPU 1 is already building MeLoMIA-ND shadows 1-15 when this starts.  This
# script frees GPU 0 as soon as the CVAE work finishes, puts it on ND shadows
# 16-30, and once both halves are in, trains the meta-classifier and scores the
# whole grid.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/golobs/miniconda3/envs/recon_/bin/python
CFG=configs/experiments/grid_brca.yaml

wait_for() {  # wait_for <pgrep pattern> <label>
    while pgrep -f "$1" > /dev/null; do sleep 60; done
    echo "### $2 finished at $(date +%H:%M:%S)"
}

wait_for "only melomia_cvae --generators" "MeLoMIA-CVAE BRCA"

echo "### GPU0 joins ND shadow building (16-30) at $(date +%H:%M:%S)"
./scripts/shadow_worker.sh 0 $CFG melomia_nd $(seq 16 30) \
    > logs/brca_nd_worker_b.log 2>&1

wait_for "prepare-only --no-meta --shadows" "ND shadow building"

echo "### meta-classifier + ND grid row at $(date +%H:%M:%S)"
CUDA_VISIBLE_DEVICES=1 $PY -u scripts/run_experiment.py $CFG --only melomia_nd \
    > logs/brca_melomia_nd.log 2>&1

echo "### filling in remaining cells at $(date +%H:%M:%S)"
CUDA_VISIBLE_DEVICES=0 $PY -u scripts/run_experiment.py $CFG \
    --only mahalamia mamamia melomia_cvae > logs/brca_finish.log 2>&1

echo "### BRCA grid complete at $(date +%H:%M:%S)"
$PY scripts/make_tables.py --dataset BRCA --out results/tables
