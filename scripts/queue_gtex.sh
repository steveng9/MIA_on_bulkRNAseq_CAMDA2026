#!/bin/bash
# GTEx cohort (configs/datasets/GTEX.yaml; Hakime's suggestion, 2026-10-22):
# targets for the challenge generators, the shadow-free grid, the donor-linked
# other-tissue attack and the report; then MeLoMIA with per-record calibration
# (K=20, as on COMBINED), CVAE first.  GPU 0 only; Steven's CausalFairness jobs
# share GPU 0 and the CPUs, so everything runs niced.
cd ~/MIA_on_bulkRNAseq_CAMDA2026
PY=/home/golobs/miniconda3/envs/recon_/bin/python
L=logs/gtex
NEW="pgm@binning=dp_quantile,edge_estimator=threshold,n_bins=16"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
stamp() { echo "### [$(date -u +%m-%d\ %H:%M)] $*" | tee -a $L/progress.log; }
build() { nice -n 5 $PY -W ignore scripts/build_targets.py --dataset GTEX "$@"; }
grid() { nice -n 5 $PY -W ignore scripts/run_experiment.py "$@"; }

stamp "targets start"
build --generators mvn cvae > $L/targets_mvn_cvae.log 2>&1 &
for s in 1 2 3 4 5; do build --generators nd --splits $s > $L/targets_nd_s$s.log 2>&1 & done
for s in 1 2 3 4 5; do build --generators pgg "$NEW" --splits $s > $L/targets_pgm_s$s.log 2>&1 & done
wait
stamp "targets done: $(ls -d artifacts/targets/GTEX/*/split_* 2>/dev/null | wc -l) built"

stamp "grid start"
for g in mvn cvae nd pgg "$NEW"; do
  grid configs/experiments/grid_gtex.yaml --generators "$g" > "$L/grid_${g:0:3}.log" 2>&1 &
done
wait
stamp "grid done"

OMP_NUM_THREADS=1 nice -n 5 $PY -W ignore scripts/perturb/gtex_donor_linked.py --jobs 6 > $L/donor_linked.log 2>&1
$PY scripts/perturb/report.py > $L/report.log 2>&1
stamp "donor-linked done"

export MIA_N_JOBS=8
grid configs/experiments/melomia_prc_gtex.yaml --only melomia_cvae > $L/melomia_cvae.log 2>&1
stamp "MeLoMIA-CVAE done"
grid configs/experiments/melomia_prc_gtex.yaml --only melomia_nd > $L/melomia_nd.log 2>&1
stamp "MeLoMIA-ND done"
stamp "GTEX_QUEUE_DONE"
