#!/usr/bin/env bash
# MeLoMIA-TabPFN experiments, every stage on ONE GPU (default 0), two sota-env
# workers on it.  Progress: logs/melomia_tabpfn/queue.log; one log per stage.
#
#   nohup scripts/queue_melomia_tabpfn.sh 0 > /dev/null 2>&1 &
#
# Stages (each resumable: finished units are cached under
# artifacts/attacks/melomia_tabpfn/<cohort>/e2/, re-running skips them):
#   1. BRCA diagonal, paper setting + its no-shadow rows      (~5 GPU-h: 30 shadows)
#   2. BRCA all-gene ladder: pool + 5 releases on 978 genes   (~4 GPU-h)
#   3. COMBINED diagonal, paper setting + no-shadow rows      (shadows dominate; slow)
#   4. off-diagonal: the paper-grid generators, both cohorts  (200 genes each, reuses shadows)
# Never edit this file while it runs (bash reads it incrementally).
set -u
cd "$(dirname "$0")/.."
GPU=${1:-0}
PY=${CAMDA_MAIN_PYTHON:-$HOME/miniconda3/envs/recon_/bin/python}
export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONUNBUFFERED=1
LOG=logs/melomia_tabpfn
mkdir -p "$LOG"
Q=$LOG/queue.log
B=configs/experiments/melomia_tabpfn_brca.yaml
C=configs/experiments/melomia_tabpfn_combined.yaml
MAIN="melomia_tabpfn melomia_tabpfn_noshadow melomia_tabpfn_g200_raw"
LADDER="melomia_tabpfn_allgenes melomia_tabpfn_stride10 melomia_tabpfn_stride10_raw"
OFFDIAG=(mvn cvae nd
  "tabsyn@class_freq=train,diff_schedule=steps,latent_scale=std,preprocess=clip:0.001:0.999+standard"
  pgg "pgm@binning=dp_quantile,edge_estimator=threshold,n_bins=16")

run() {
  local name=$1; shift
  echo "[$(date '+%F %T')] start $name" >> "$Q"
  "$PY" scripts/run_experiment.py "$@" >> "$LOG/$name.log" 2>&1
  echo "[$(date '+%F %T')] end $name (exit $?)" >> "$Q"
}

run brca_main     "$B" --only $MAIN   --generators tabpfn
run brca_ladder   "$B" --only $LADDER --generators tabpfn
run combined_main "$C" --only $MAIN   --generators tabpfn
for g in "${OFFDIAG[@]}"; do
  safe=${g%%@*}
  run "brca_offdiag_$safe"     "$B" --only $MAIN --generators "$g"
  run "combined_offdiag_$safe" "$C" --only $MAIN --generators "$g"
done
echo "[$(date '+%F %T')] QUEUE DONE" >> "$Q"
