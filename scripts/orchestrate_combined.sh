#!/bin/bash
# COMBINED grid, same shape as the BRCA one but with K=20 -- the cohort is four
# times larger, so each shadow costs about four times as much to train.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/golobs/miniconda3/envs/recon_/bin/python
CFG=configs/experiments/grid_combined.yaml
stamp() { date +%H:%M:%S; }

echo "### [$(stamp)] statistical attacks on whatever targets exist"
$PY -u scripts/run_experiment.py $CFG --only mahalamia mamamia \
    > logs/combined_statistical.log 2>&1

echo "### [$(stamp)] GPU1: ND shadows 1-10 | GPU0: MeLoMIA-CVAE"
./scripts/shadow_worker.sh 1 $CFG melomia_nd $(seq 1 10) \
    > logs/combined_nd_worker_a.log 2>&1 &
ND_A=$!
CUDA_VISIBLE_DEVICES=0 $PY -u scripts/run_experiment.py $CFG --only melomia_cvae \
    > logs/combined_melomia_cvae.log 2>&1
echo "### [$(stamp)] MeLoMIA-CVAE done"

./scripts/shadow_worker.sh 0 $CFG melomia_nd $(seq 11 20) \
    > logs/combined_nd_worker_b.log 2>&1 &
ND_B=$!
wait $ND_A $ND_B
echo "### [$(stamp)] ND shadow building done"

CUDA_VISIBLE_DEVICES=1 $PY -u scripts/run_experiment.py $CFG --only melomia_nd \
    > logs/combined_melomia_nd.log 2>&1

echo "### [$(stamp)] re-running the statistical attacks now that DP-PGM targets exist"
$PY -u scripts/run_experiment.py $CFG --only mahalamia mamamia \
    > logs/combined_statistical.log 2>&1

echo "### [$(stamp)] COMBINED grid complete"
$PY scripts/make_tables.py --dataset COMBINED --out results/tables
