#!/bin/bash
# Everything to run once orchestrate_rest.sh has exited.
#
# Deliberately a separate file rather than an addition to orchestrate_rest.sh:
# bash reads a script incrementally by byte offset, so editing one that is
# currently executing can make it resume at the wrong place.  Whatever needs
# doing after a long run goes in a new file.
#
#   1. Tables and figures for the runs orchestrate_rest.sh produced but did not
#      aggregate: the synth-shadow ablation on BRCA.
#   2. The internal-proxy selection ablation (TODO item 10/11).  Held until now
#      so that every cell in the main grid was selected the same way; running it
#      earlier would have left a grid whose cells disagree about what "selected"
#      means.
#   3. Negative controls over everything, and the final tables.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/golobs/miniconda3/envs/recon_/bin/python
export MIA_N_JOBS=${MIA_N_JOBS:-10}
export OMP_NUM_THREADS=$MIA_N_JOBS
export OPENBLAS_NUM_THREADS=$MIA_N_JOBS
export MKL_NUM_THREADS=$MIA_N_JOBS
stamp() { date +%H:%M:%S; }

if pgrep -f "[o]rchestrate_rest.sh" >/dev/null; then
    echo "orchestrate_rest.sh is still running -- refusing to start."
    echo "Both would contend for the same two GPUs and the same shadow caches."
    exit 1
fi

echo "### [$(stamp)] tables for the synth-shadow ablation"
$PY scripts/make_tables.py --config configs/experiments/ablation_synth_shadow.yaml \
    --out results/tables > logs/ablation_tables.log 2>&1
$PY scripts/make_figures.py --config configs/experiments/ablation_synth_shadow.yaml \
    --dataset BRCA >> logs/ablation_tables.log 2>&1

echo "### [$(stamp)] internal-proxy selection ablation"
# The _grouped arm is a cache hit (its tag is the main grid's) and the _blockcv
# arm shares the same shadow stack, so this is one meta-classifier search per
# backend.  One GPU each.
CUDA_VISIBLE_DEVICES=0 $PY -u scripts/run_experiment.py \
    configs/experiments/ablation_internal_proxy.yaml \
    --only melomia_cvae_grouped melomia_cvae_blockcv \
    > logs/internal_proxy_cvae.log 2>&1 & A=$!
CUDA_VISIBLE_DEVICES=1 $PY -u scripts/run_experiment.py \
    configs/experiments/ablation_internal_proxy.yaml \
    --only melomia_nd_grouped melomia_nd_blockcv \
    > logs/internal_proxy_nd.log 2>&1 & B=$!
wait $A $B
$PY scripts/make_tables.py --config configs/experiments/ablation_internal_proxy.yaml \
    --out results/tables > logs/internal_proxy_tables.log 2>&1

echo "### [$(stamp)] negative controls"
$PY scripts/sanity_check.py --dataset BRCA
$PY scripts/sanity_check.py --dataset COMBINED

echo "### [$(stamp)] all done"
