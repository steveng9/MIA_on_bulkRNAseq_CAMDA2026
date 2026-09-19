#!/bin/bash
# Sequencing for the BRCA grid on a two-GPU box.
#
#   GPU 1: MeLoMIA-ND shadows 1-15, then the ND meta-classifier and grid row
#   GPU 0: MeLoMIA-CVAE end to end, then ND shadows 16-30
#
# The two ND workers share one cache and claim shadows with lock files, so the
# split is just load balancing -- either can finish the other's leftovers.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/golobs/miniconda3/envs/recon_/bin/python
CFG=configs/experiments/grid_brca.yaml
stamp() { date +%H:%M:%S; }

echo "### [$(stamp)] GPU1: ND shadows 1-15"
./scripts/shadow_worker.sh 1 $CFG melomia_nd $(seq 1 15) \
    > logs/brca_nd_worker_a.log 2>&1 &
ND_A=$!

echo "### [$(stamp)] GPU0: MeLoMIA-CVAE"
CUDA_VISIBLE_DEVICES=0 $PY -u scripts/run_experiment.py $CFG --only melomia_cvae \
    > logs/brca_melomia_cvae.log 2>&1
echo "### [$(stamp)] MeLoMIA-CVAE done"

echo "### [$(stamp)] GPU0: ND shadows 16-30"
./scripts/shadow_worker.sh 0 $CFG melomia_nd $(seq 16 30) \
    > logs/brca_nd_worker_b.log 2>&1 &
ND_B=$!

wait $ND_A $ND_B
echo "### [$(stamp)] ND shadow building done"

echo "### [$(stamp)] ND meta-classifier + grid row"
CUDA_VISIBLE_DEVICES=1 $PY -u scripts/run_experiment.py $CFG --only melomia_nd \
    > logs/brca_melomia_nd.log 2>&1
echo "### [$(stamp)] ND row done"

echo "### [$(stamp)] filling in the statistical attacks over all targets"
$PY -u scripts/run_experiment.py $CFG --only mahalamia mamamia \
    > logs/brca_statistical.log 2>&1

echo "### [$(stamp)] BRCA grid complete"
$PY scripts/make_tables.py --dataset BRCA --out results/tables
