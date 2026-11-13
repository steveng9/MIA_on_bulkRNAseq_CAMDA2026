#!/bin/bash
# White box against black box on TCGA-BRCA and TCGA-COMBINED (Steven, 2026-11-13).
# configs/experiments/whitebox_{brca,combined}.yaml say what each arm is.
#   lane S (CPU 16-17): MahalaMIA black/white box on the four generators, then the
#          three MAMA-MIA access levels on the corrected DP-PGM.  ~1 hour.
#   lane L (CPU 12-15): white-box MeLoMIA-CVAE: K shadows of the CVAE target's
#          recipe on real splits (GPU, ~1 GB each), features, classifier, then the
#          five targets.  BRCA (K=30) then COMBINED (K=20).
# Restart-safe: stacks, features and classifiers are cached; results append to
# results/index.csv.  Progress: logs/whitebox/progress.log.
cd ~/MIA_on_bulkRNAseq_CAMDA2026
PY=/home/golobs/miniconda3/envs/recon_/bin/python
L=logs/whitebox
mkdir -p $L
PGM="pgm@binning=dp_quantile,edge_estimator=threshold,n_bins=16"
stamp() { echo "### [$(date -u +%m-%d\ %H:%M)] $*" | tee -a $L/progress.log; }
run() {   # $1 = cores, rest = run_experiment arguments
  local c=$1; shift
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=0 \
    nice -n 5 taskset -c $c $PY -W ignore scripts/run_experiment.py "$@"
}

$PY -W ignore scripts/whitebox/materialise_mvn_models.py --datasets BRCA COMBINED > $L/materialise_mvn.log 2>&1 \
  || { stamp "white box: MVN models FAILED the identity checks (see $L/materialise_mvn.log)"; exit 1; }

(   # lane S
  for ds in brca combined; do
    run 16-17 configs/experiments/whitebox_$ds.yaml --only mahalamia mahalamia_wb > $L/mahalamia_$ds.log 2>&1 \
      && stamp "white box: MahalaMIA ${ds^^} done" || stamp "white box: MahalaMIA ${ds^^} FAILED"
    run 16-17 configs/experiments/whitebox_$ds.yaml --only mamamia_v2 mamamia_v2_wb_release mamamia_v2_wb_model \
      --generators "$PGM" > $L/mamamia_$ds.log 2>&1 \
      && stamp "white box: MAMA-MIA ${ds^^} done" || stamp "white box: MAMA-MIA ${ds^^} FAILED"
  done
) &

(   # lane L
  for ds in brca combined; do
    run 12-15 configs/experiments/whitebox_$ds.yaml --only melomia_cvae_wb --generators cvae > $L/melomia_cvae_$ds.log 2>&1 \
      && stamp "white box: MeLoMIA-CVAE ${ds^^} done" || stamp "white box: MeLoMIA-CVAE ${ds^^} FAILED"
  done
) &
wait
stamp "WHITEBOX_DONE"
