#!/bin/sh
# After run_followups.sh: the BRCA gene-subset ablation with GSE58135 (through
# TCGA's fitted VST) as the auxiliary set, which gives BRCA the reference-based
# baselines and the vDE / dDE selections it otherwise cannot run.
PY=/home/golobs/miniconda3/envs/recon_/bin/python
export OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
while pgrep -f "perturb/run_followups.sh" > /dev/null; do sleep 20; done
taskset -c 12-23,36-47 $PY -W ignore scripts/perturb/gene_subsets.py --dataset BRCA --jobs 7 --logan \
    --reference /home/golobs/data/GSE58135/processed/aux_vst_frozen.tsv --tag _gse
echo FOLLOWUPS2_DONE
