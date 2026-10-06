#!/bin/bash
# MeLoMIA-TabSyn end to end for one cohort: shadow stack, black box, white box.
#
#   scripts/run_melomia_tabsyn.sh BRCA [workers on GPU 0] [workers on GPU 1]
#
# Stages (each resumable; everything is cached per shadow / per target):
#   1  shadow stack to K=10            base shadows (target recipe) -> internal
#                                      synthetic data -> synth-shadows (probe
#                                      recipe) -> loss features
#   2  score K=10, tuned-TabSyn target black box (proxies built in parallel) and
#                                      white box (shadows reuse the base shadows)
#   3  shadow stack to full K, score   as 2
#   4  off-diagonal                    MeLoMIA-TabSyn against the other generators
# A TabSyn fit is ~1 GPU-hour on BRCA when 4 share a card, so the full BRCA
# stack (2 fits per shadow) is ~60 GPU-hours.  Progress: logs/melomia_tabsyn/.
set -uo pipefail
cd "$(dirname "$0")/.."
DS=${1:?cohort}; W0=${2:-4}; W1=${3:-2}
PY=/home/golobs/miniconda3/envs/recon_/bin/python
CFG=configs/experiments/melomia_tabsyn_$(echo "$DS" | tr A-Z a-z).yaml
T="tabsyn@class_freq=train,diff_schedule=steps,latent_scale=std,preprocess=clip:0.001:0.999+standard"
K=$(grep -m1 "n_shadows:" "$CFG" | awk '{print $2}')
LOG=logs/melomia_tabsyn/$DS; mkdir -p "$LOG"
export OMP_NUM_THREADS=2 MIA_N_JOBS=8
say() { echo "### [$(date -u +%m-%d\ %H:%M)] $*" | tee -a "$LOG/progress.log"; }

stack() {    # stack <label> <K>
    local pids=() i
    for i in $(seq 1 "$W0"); do
        ./scripts/shadow_loop.sh 0 "$CFG" "$1" "$2" $([ $((i % 2)) = 0 ] && echo down || echo up) >> "$LOG/stack_gpu0_w$i.log" 2>&1 & pids+=($!)
    done
    for i in $(seq 1 "$W1"); do
        ./scripts/shadow_loop.sh 1 "$CFG" "$1" "$2" $([ $((i % 2)) = 0 ] && echo up || echo down) >> "$LOG/stack_gpu1_w$i.log" 2>&1 & pids+=($!)
    done
    wait "${pids[@]}"
}

proxies() {  # proxies <label> <generators...>   five splits over the two GPUs
    local label=$1; shift
    local pids=() s
    for s in 1 2 3 4 5; do
        CUDA_VISIBLE_DEVICES=$(( s % 2 )) $PY -W ignore scripts/melomia_proxies.py "$CFG" "$label" \
            --generators "$@" --splits $s >> "$LOG/proxies_s$s.log" 2>&1 & pids+=($!)
    done
    wait "${pids[@]}"
}

score() {    # score <black-box label> <white-box label>
    proxies "$1" "$T"
    CUDA_VISIBLE_DEVICES=0 $PY -W ignore scripts/run_experiment.py "$CFG" --only "$1" --generators "$T" >> "$LOG/score_$1.log" 2>&1
    CUDA_VISIBLE_DEVICES=0 $PY -W ignore scripts/run_experiment.py "$CFG" --only "$2" --generators "$T" >> "$LOG/score_$2.log" 2>&1
    grep -h "MEAN" "$LOG/score_$1.log" "$LOG/score_$2.log" | tail -n 2 | tee -a "$LOG/progress.log"
}

say "stage 1: shadow stack to K=10 ($W0 + $W1 workers)"
stack melomia_tabsyn_k10 10
say "stage 2: score K=10 (black box, then white box)"
score melomia_tabsyn_k10 melomia_tabsyn_wb_k10
say "stage 3: shadow stack to K=$K"
stack melomia_tabsyn "$K"
score melomia_tabsyn melomia_tabsyn_wb
say "stage 4: off-diagonal targets"
proxies melomia_tabsyn mvn cvae nd pgg tabpfn
CUDA_VISIBLE_DEVICES=0 $PY -W ignore scripts/run_experiment.py "$CFG" --only melomia_tabsyn --generators mvn cvae nd pgg tabpfn >> "$LOG/score_offdiag.log" 2>&1
grep -h "MEAN" "$LOG/score_offdiag.log" | tee -a "$LOG/progress.log"
say "MELOMIA_TABSYN_DONE $DS"
