#!/usr/bin/env bash
# After run_pgg_fill.sh's builds: MahalaMIA variant rows and the 5-column figures
# on the new pgg targets; after its attacks: the slide table.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD
PY=/home/golobs/miniconda3/envs/recon_/bin/python
PIN="taskset -c 12-23,36-47"
until grep -q "=== builds done" logs/pgg_fill.log; do sleep 60; done
for l in brca combined; do
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 $PIN $PY scripts/run_experiment.py \
    configs/experiments/mahala_slides_$l.yaml --generators pgg
done
for k in pca umap; do CUDA_VISIBLE_DEVICES= $PIN $PY -W ignore scripts/umap_grid_slides.py --kind $k; done
echo "FIGURES DONE $(date -Is)"
until grep -q "PGG FILL COMPLETE" logs/pgg_fill.log; do sleep 60; done
CUDA_VISIBLE_DEVICES= $PIN $PY scripts/slide_table.py
echo "PGG FOLLOWUP COMPLETE $(date -Is)"
