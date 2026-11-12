#!/usr/bin/env bash
# Fidelity for every paper generator on the two TCGA cohorts, five splits, into results/fidelity.csv
cd /home/golobs/MIA_on_bulkRNAseq_CAMDA2026
PY=/home/golobs/miniconda3/envs/recon_/bin/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=
PGM="pgm@binning=dp_quantile,edge_estimator=threshold,n_bins=16"
taskset -c 20-23 $PY -W ignore scripts/eval_fidelity.py --dataset BRCA --generators mvn cvae nd pgg "$PGM" tabpfn
taskset -c 20-23 $PY -W ignore scripts/eval_fidelity.py --dataset COMBINED --generators mvn cvae nd pgg "$PGM"
echo "### [$(date -u +%m-%d\ %H:%M)] fidelity for all paper generators (TCGA) done" >> logs/paper_fill/progress.log
