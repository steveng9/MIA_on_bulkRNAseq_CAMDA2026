#!/bin/bash
# The paper's MAMA-MIA (black box: public star of tables, bin edges read off the
# release, class-centred) on the epsilon sweep of the corrected DP-PGM at 16 bins
# (targets built by scripts/pgm_eps_sweep.py on 2026-11-11).  CPU only.
cd ~/MIA_on_bulkRNAseq_CAMDA2026
PY=/home/golobs/miniconda3/envs/recon_/bin/python
L=logs/paper_fill
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
for ds in brca combined; do
  for e in 0.1 0.3 1.0 3.0 30.0 100.0 300.0 1000.0; do
    g="pgm@binning=dp_quantile,edge_estimator=threshold,epsilon=$e,n_bins=16"
    nice -n 5 taskset -c 12-19 $PY -W ignore scripts/run_experiment.py configs/experiments/paper_grid_$ds.yaml \
      --only mamamia_v2 --generators "$g" --device cpu > $L/eps_curve_mama_${ds}_$e.log 2>&1 &
  done
  wait
  echo "### [$(date -u +%m-%d\ %H:%M)] MAMA-MIA on the epsilon sweep, $ds" >> $L/progress.log
done
