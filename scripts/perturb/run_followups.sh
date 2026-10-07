#!/bin/sh
# Follow-ups that wait for the first passes: the adaptive-adversary variant of
# the donor-linked experiment, then the DP-PGM targets of the gene-subset
# ablation (re-run after the MAMA-MIA v2 edge cache was made view-safe).
PY=/home/golobs/miniconda3/envs/recon_/bin/python
PIN="taskset -c 12-23,36-47"
export OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""

while pgrep -f "perturb/donor_linked.py --dataset" > /dev/null; do sleep 20; done
$PIN $PY -W ignore scripts/perturb/donor_linked.py --dataset COMBINED --jobs 7 --adapt
$PIN $PY -W ignore scripts/perturb/donor_linked.py --dataset BRCA --jobs 7 --adapt

while pgrep -f "perturb/gene_subsets.py --dataset" > /dev/null; do sleep 20; done
rm -f results/perturb/gene_subsets_parts/*__pgm@*
PGM="pgm@binning=dp_quantile,edge_estimator=threshold,n_bins=16"
$PIN $PY -W ignore scripts/perturb/gene_subsets.py --dataset BRCA --jobs 5 --generators "$PGM"
$PIN $PY -W ignore scripts/perturb/gene_subsets.py --dataset COMBINED --jobs 5 --logan --generators "$PGM"
echo FOLLOWUPS_DONE
