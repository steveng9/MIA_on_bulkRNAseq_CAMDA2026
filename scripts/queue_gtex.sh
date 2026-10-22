#!/bin/bash
# GTEx cohort (configs/datasets/GTEX.yaml; Hakime's suggestion, 2026-10-22):
# targets for the challenge generators, the shadow-free grid, the donor-linked
# other-tissue attack and the report, then MeLoMIA with per-record calibration
# (K=20, as on COMBINED).  Each generator's attacks start as soon as its five
# targets exist: the CAMDA-26 DP-PGM takes over two hours per split, the others
# minutes.  The MeLoMIA-CVAE shadow stack is built meanwhile (it needs no target).
#
#   WAIT_ND="pid ..." WAIT_PGM="pid ..." scripts/queue_gtex.sh
#
# waits for target builders already running (by PID) instead of starting them.
# GPU 0 only; Steven's CausalFairness jobs share GPU 0 and the CPUs, so
# everything is niced and pinned to cores 12-23 and 36-47.
cd ~/MIA_on_bulkRNAseq_CAMDA2026
PY=/home/golobs/miniconda3/envs/recon_/bin/python
L=logs/gtex
NEW="pgm@binning=dp_quantile,edge_estimator=threshold,n_bins=16"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
stamp() { echo "### [$(date -u +%m-%d\ %H:%M)] $*" | tee -a $L/progress.log; }
run() { nice -n 5 taskset -c 12-23,36-47 "$@"; }
build() { run $PY -W ignore scripts/build_targets.py --dataset GTEX "$@"; }
grid() { run $PY -W ignore scripts/run_experiment.py "$@"; }
have() {   # all five targets of generator $1 exist
  local d=artifacts/targets/GTEX/${1/@//}
  for s in 1 2 3 4 5; do [ -f "$d/split_$s/synthetic_data.csv" ] || return 1; done
}
await() {  # generator, then PIDs of builders running elsewhere; builds what is still missing
  local g=$1; shift
  for p in "$@"; do while kill -0 "$p" 2>/dev/null; do sleep 30; done; done
  have "$g" || build --generators "$g" > "$L/targets_${g:0:3}_rest.log" 2>&1
  have "$g" && stamp "targets $g ready" || stamp "targets $g MISSING after build"
}
attacks() {  # shadow-free grid and donor-linked attack for the generators given
  local pids=()
  for g in "$@"; do
    grid configs/experiments/grid_gtex.yaml --generators "$g" > "$L/grid_${g:0:3}.log" 2>&1 &
    pids+=($!)
  done
  wait "${pids[@]}"
  OMP_NUM_THREADS=1 run $PY -W ignore scripts/perturb/gtex_donor_linked.py --jobs 6 \
    --generators "$@" >> $L/donor_linked.log 2>&1
  $PY scripts/perturb/report.py > $L/report.log 2>&1
  stamp "shadow-free attacks and donor-linked done: $*"
}

stamp "queue v2 start"
MIA_N_JOBS=8 grid configs/experiments/melomia_prc_gtex.yaml --only melomia_cvae --prepare-only \
  > $L/melomia_cvae_prepare.log 2>&1 &
PREP=$!

have mvn || build --generators mvn > $L/targets_mvn.log 2>&1
have cvae || build --generators cvae > $L/targets_cvae.log 2>&1
await nd $WAIT_ND
attacks mvn cvae nd

wait $PREP
stamp "MeLoMIA-CVAE stack ready"
MIA_N_JOBS=8 grid configs/experiments/melomia_prc_gtex.yaml --only melomia_cvae \
  --generators mvn cvae nd > $L/melomia_cvae.log 2>&1
stamp "MeLoMIA-CVAE scored: mvn cvae nd"

await pgg $WAIT_PGM
await "$NEW"
attacks pgg "$NEW"
MIA_N_JOBS=8 grid configs/experiments/melomia_prc_gtex.yaml --only melomia_cvae \
  --generators pgg "$NEW" >> $L/melomia_cvae.log 2>&1
stamp "MeLoMIA-CVAE scored: DP-PGMs"

MIA_N_JOBS=8 grid configs/experiments/melomia_prc_gtex.yaml --only melomia_nd > $L/melomia_nd.log 2>&1
$PY scripts/perturb/report.py > $L/report.log 2>&1
stamp "MeLoMIA-ND done"
stamp "GTEX_QUEUE_DONE"
