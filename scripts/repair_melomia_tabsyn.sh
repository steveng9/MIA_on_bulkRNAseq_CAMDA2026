#!/bin/bash
# Redo the black-box half of MeLoMIA-TabSyn after the probe-collapse fix
# (diff_warmup; docs/SOTA_GENERATORS.md, "Probe collapse").  White box is
# untouched: it never used a probe.
#   1  refit the denoiser of every synth-shadow probe (VAEs kept)
#   2  re-extract shadow features, retrain the meta-classifiers
#   3  retrain the final proxies (tuned-TabSyn targets first), score
#   scripts/repair_melomia_tabsyn.sh BRCA [gpu] [denoiser workers] [proxy workers]
set -uo pipefail
cd "$(dirname "$0")/.."
DS=${1:?cohort}; GPU=${2:-0}; W=${3:-4}; WP=${4:-2}   # WP: full proxy fits at once (~11 GB each)
PY=/home/golobs/miniconda3/envs/recon_/bin/python
CFG=configs/experiments/melomia_tabsyn_$(echo "$DS" | tr A-Z a-z).yaml
T="tabsyn@class_freq=train,diff_schedule=steps,latent_scale=std,preprocess=clip:0.001:0.999+standard"
K=$(grep -m1 "n_shadows:" "$CFG" | awk '{print $2}')
C=artifacts/attacks/melomia_tabsyn/$DS/n200
LOG=logs/melomia_tabsyn/$DS/repair; mkdir -p "$LOG"
export OMP_NUM_THREADS=2 MIA_N_JOBS=8 CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=.
say() { echo "### [$(date -u +%m-%d\ %H:%M)] $*" | tee -a "$LOG/progress.log"; }

# results of the collapsed probes: kept, out of the way
if [ ! -d "$C/broken_no_warmup" ]; then
    mkdir -p "$C/broken_no_warmup"
    mv "$C/proxy_features" "$C/meta" "$C/broken_no_warmup/" 2>/dev/null
fi

say "1: refit probe denoisers ($W workers)"
pids=()
for i in $(seq 0 $((W - 1))); do
    $PY -W ignore scripts/repair_tabsyn_probes.py "$DS" --worker $i --of $W >> "$LOG/probes_w$i.log" 2>&1 & pids+=($!)
done
wait "${pids[@]}"
say "1 done: $(ls $C/synth_shadows/*.rewarmed | wc -l) probes refitted"

say "2: shadow features + meta-classifiers"
$PY -W ignore scripts/run_experiment.py "$CFG" --only melomia_tabsyn --prepare-only >> "$LOG/prepare.log" 2>&1

proxies() {
    local pids=() s
    for s in 1 2 3 4 5; do
        $PY -W ignore scripts/melomia_proxies.py "$CFG" melomia_tabsyn --generators "$@" --splits $s >> "$LOG/proxies_s$s.log" 2>&1 & pids+=($!)
        [ $(( s % WP )) = 0 ] && wait "${pids[@]}"
    done
    wait "${pids[@]}"
}
say "3: final proxies, tuned-TabSyn targets"
proxies "$T"
$PY -W ignore scripts/run_experiment.py "$CFG" --only melomia_tabsyn melomia_tabsyn_k10 --generators "$T" >> "$LOG/score.log" 2>&1
grep -h "split [1-5]:\|MEAN" "$LOG/score.log" | cut -c1-8,95-200 | tee -a "$LOG/progress.log"
say "3: final proxies, other generators"
proxies mvn cvae nd pgg tabpfn
$PY -W ignore scripts/run_experiment.py "$CFG" --only melomia_tabsyn --generators mvn cvae nd pgg tabpfn >> "$LOG/score_offdiag.log" 2>&1
grep -h "MEAN" "$LOG/score_offdiag.log" | cut -c1-200 | tee -a "$LOG/progress.log"
say "REPAIR_DONE $DS"
