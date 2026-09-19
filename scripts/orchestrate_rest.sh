#!/bin/bash
# Everything still outstanding after the BRCA statistical attacks, in order.
#
# Replaces orchestrate_brca.sh + orchestrate_after_brca.sh, which were written
# before the meta-classifier search was found to be thrashing the box (see
# mia/attacks/melomia/meta.py:n_jobs).  Each attack now gets a bounded share of
# the cores so two can search at once without oversubscribing.
#
#   1. BRCA MeLoMIA rows   -- the last two cells of the BRCA grid.  Shadow
#      stacks and features are already cached, so this is the meta-classifier
#      search plus one proxy per (generator, split).
#   2. BRCA tables         -- cheap.
#   3. tuned BRCA grid     -- statistical attacks only; MeLoMIA cells reuse the
#      caches from step 1.
#   4. synth-shadow ablation
#   5. COMBINED MeLoMIA rows, then tables and figures.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/golobs/miniconda3/envs/recon_/bin/python
export MIA_N_JOBS=${MIA_N_JOBS:-10}
export OMP_NUM_THREADS=$MIA_N_JOBS
export OPENBLAS_NUM_THREADS=$MIA_N_JOBS
export MKL_NUM_THREADS=$MIA_N_JOBS
stamp() { date +%H:%M:%S; }

echo "### [$(stamp)] BRCA MeLoMIA rows (MIA_N_JOBS=$MIA_N_JOBS per attack)"
CUDA_VISIBLE_DEVICES=0 $PY -u scripts/run_experiment.py \
    configs/experiments/grid_brca.yaml --only melomia_cvae \
    > logs/brca_melomia_cvae.log 2>&1 &
CVAE=$!
CUDA_VISIBLE_DEVICES=1 $PY -u scripts/run_experiment.py \
    configs/experiments/grid_brca.yaml --only melomia_nd \
    > logs/brca_melomia_nd.log 2>&1 &
ND=$!
wait $CVAE $ND
echo "### [$(stamp)] BRCA grid complete"

$PY scripts/make_tables.py --config configs/experiments/grid_brca.yaml \
    --out results/tables > logs/brca_tables.log 2>&1
$PY scripts/make_figures.py --config configs/experiments/grid_brca.yaml \
    --dataset BRCA >> logs/brca_tables.log 2>&1

echo "### [$(stamp)] tuned BRCA grid"
$PY -u scripts/run_experiment.py configs/experiments/grid_brca_tuned.yaml \
    > logs/brca_tuned.log 2>&1
$PY scripts/make_tables.py --config configs/experiments/grid_brca_tuned.yaml \
    --out results/tables >> logs/brca_tables.log 2>&1

echo "### [$(stamp)] synth-shadow ablation"
CUDA_VISIBLE_DEVICES=0 $PY -u scripts/run_experiment.py \
    configs/experiments/ablation_synth_shadow.yaml \
    --only melomia_cvae_real melomia_cvae_synth > logs/ablation_cvae.log 2>&1 &
A=$!
CUDA_VISIBLE_DEVICES=1 $PY -u scripts/run_experiment.py \
    configs/experiments/ablation_synth_shadow.yaml \
    --only melomia_nd_real melomia_nd_synth > logs/ablation_nd.log 2>&1 &
B=$!
wait $A $B
echo "### [$(stamp)] ablation done"

echo "### [$(stamp)] COMBINED MeLoMIA rows"
CUDA_VISIBLE_DEVICES=0 $PY -u scripts/run_experiment.py \
    configs/experiments/grid_combined.yaml --only melomia_cvae \
    > logs/combined_melomia_cvae.log 2>&1 &
C=$!
CUDA_VISIBLE_DEVICES=1 $PY -u scripts/run_experiment.py \
    configs/experiments/grid_combined.yaml --only melomia_nd \
    > logs/combined_melomia_nd.log 2>&1 &
D=$!
wait $C $D

$PY scripts/make_tables.py --config configs/experiments/grid_combined.yaml \
    --out results/tables > logs/combined_tables.log 2>&1
$PY scripts/make_figures.py --config configs/experiments/grid_combined.yaml \
    --dataset COMBINED >> logs/combined_tables.log 2>&1
echo "### [$(stamp)] all done"
