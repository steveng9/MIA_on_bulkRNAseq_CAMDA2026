#!/bin/bash
# What runs once the BRCA grid is in, ordered by value per hour.
#
#   1. tuned-parameter BRCA grid  -- minutes.  The MeLoMIA cells reuse the
#      caches the main grid already built, so only the statistical attacks are
#      actually re-scored.
#   2. synth-shadow ablation      -- ~1.5h.  Justifies the central method claim,
#      and is cheap: the real-data-shadow arm skips base shadows entirely, while
#      the synth-shadow arm at K=15 reuses the K=30 stack.
#   3. COMBINED grid              -- 15h+.  More of the same on a larger cohort.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/golobs/miniconda3/envs/recon_/bin/python
stamp() { date +%H:%M:%S; }

# Wait for the BRCA orchestrator to release both GPUs.  The pattern is written
# so it cannot match this script's own command line.
while pgrep -f "orchestrate_brca[.]sh" > /dev/null; do sleep 120; done
echo "### [$(stamp)] BRCA grid finished"

echo "### [$(stamp)] tuned BRCA grid"
$PY -u scripts/run_experiment.py configs/experiments/grid_brca_tuned.yaml \
    > logs/brca_tuned.log 2>&1
$PY scripts/make_tables.py --config configs/experiments/grid_brca_tuned.yaml \
    --out results/tables > /dev/null 2>&1

echo "### [$(stamp)] synth-shadow ablation"
CUDA_VISIBLE_DEVICES=0 $PY -u scripts/run_experiment.py \
    configs/experiments/ablation_synth_shadow.yaml \
    --only melomia_cvae_real melomia_cvae_synth > logs/ablation_cvae.log 2>&1 &
ABL_CVAE=$!
CUDA_VISIBLE_DEVICES=1 $PY -u scripts/run_experiment.py \
    configs/experiments/ablation_synth_shadow.yaml \
    --only melomia_nd_real melomia_nd_synth > logs/ablation_nd.log 2>&1 &
ABL_ND=$!
wait $ABL_CVAE $ABL_ND
echo "### [$(stamp)] ablation done"

echo "### [$(stamp)] COMBINED grid"
exec ./scripts/orchestrate_combined.sh
