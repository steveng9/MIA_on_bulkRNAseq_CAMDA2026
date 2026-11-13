#!/bin/bash
# Fill the TabSyn / TabPFN gaps in the paper's tables (2026-11-13).
#   lane C (CPU 12-19): gene-subset sweeps for TabSyn and TabPFN on both TCGA cohorts,
#          and the TCGA robustness tables for TabPFN COMBINED split 2 (only split 1 was scored).
#   lane T (GPU 0, CPU 36-39): TabSyn on GTEx (built beside the TabSyn ablation, ~11 GB each),
#          then every GTEx table.  While it builds, logs/paper_fill/GPU_LOCK_TABSYN_GTEX
#          exists and scripts/queue_tabpfn_combined.sh waits before its next build.
#   lane P (after TABPFN_COMBINED_DONE): refresh every TabPFN COMBINED table, then TabPFN on
#          GTEx one split at a time, scored as each lands.
# Robustness experiments use the paper's MahalaMIA and write the _default outputs, as in
# scripts/perturb/run_default_mahalamia.sh.
# Restart-safe: builds skip existing targets; set SKIP_C=1 / BUILD_PID=<pid> to resume.
cd ~/MIA_on_bulkRNAseq_CAMDA2026
PY=/home/golobs/miniconda3/envs/recon_/bin/python
L=logs/paper_fill
TS="tabsyn@class_freq=train,diff_schedule=steps,latent_scale=std,preprocess=clip:0.001:0.999+standard"
stamp() { echo "### [$(date -u +%m-%d\ %H:%M)] $*" | tee -a $L/progress.log; }
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4

# robustness experiments with the paper's MahalaMIA, _default outputs; $1 = cores, rest = args
perturb() { local cores=$1 script=$2; shift 2
  PERTURB_TAG=_default PERTURB_ATTACKS="${ATT:-MahalaMIA}" CUDA_VISIBLE_DEVICES="" \
    nice -n 5 taskset -c $cores $PY -W ignore scripts/perturb/$script "$@"; }

tcga_tables_tabpfn_combined() {   # $1 = cores, $2 = splits
  local c=$1 sp=$2
  perturb $c aux_mismatch.py --dataset COMBINED --generators tabpfn --splits $sp --jobs 2
  perturb $c candidate_mismatch.py --dataset COMBINED --generators tabpfn --splits $sp --jobs 2
  ATT="MahalaMIA|MahalaMIA (no reference)" perturb $c donor_linked.py --dataset COMBINED --generators tabpfn --splits $sp --jobs 2
  perturb $c gene_subsets.py --dataset COMBINED --generators tabpfn --splits $sp --jobs 2 --tag _default
  perturb $c gene_subset_attack.py --dataset COMBINED --generators tabpfn --splits $sp --jobs 2
}

gtex_tables() {   # $1 = cores, $2 = generator, $3 = splits
  local c=$1 g=$2 sp=$3 C="nice -n 5 taskset -c $1"
  CUDA_VISIBLE_DEVICES=0 $C $PY -W ignore scripts/run_experiment.py configs/experiments/paper_grid_gtex.yaml --generators "$g" --splits $sp
  $C $PY -W ignore scripts/eval_fidelity.py --dataset GTEX --generators "$g" --splits $sp
  $C $PY -W ignore scripts/eval_bio.py --datasets GTEX --generators "$g" --splits $sp
  $C $PY -W ignore scripts/mahalamia_ablation.py --datasets GTEX --generators "$g" --splits $sp --workers 2
  ATT="MahalaMIA|MahalaMIA (no reference)" perturb $c gtex_donor_linked.py --generators "$g" --splits $sp --jobs 2
}

[ -z "$SKIP_C" ] && (   # lane C
  tcga_tables_tabpfn_combined 12-19 2 > $L/fill_tabpfn_combined_s2.log 2>&1
  stamp "fill: TCGA robustness tables for TabPFN COMBINED split 2"
) &

(   # lane T
  if [ -n "$BUILD_PID" ]; then
    while kill -0 $BUILD_PID 2>/dev/null; do sleep 60; done
  else
    touch $L/GPU_LOCK_TABSYN_GTEX
    CUDA_VISIBLE_DEVICES=0 nice -n 5 taskset -c 36-39 $PY -W ignore scripts/build_targets.py --dataset GTEX \
      --generators "$TS" --splits 1 2 3 4 5 > $L/fill_tabsyn_gtex_build.log 2>&1
  fi
  rm -f $L/GPU_LOCK_TABSYN_GTEX
  n=$(ls artifacts/targets/GTEX/tabsyn*/split_*/synthetic_data.csv 2>/dev/null | wc -l)
  stamp "fill: TabSyn GTEx built ($n of 5 splits)"
  gtex_tables 36-39 "$TS" "1 2 3 4 5" > $L/fill_tabsyn_gtex_score.log 2>&1
  stamp "FILL_TABSYN_GTEX_DONE"
) &

(   # lane P
  start=$(wc -l < $L/progress.log)
  until tail -n +$((start+1)) $L/progress.log | grep -q "TABPFN_COMBINED_DONE\|TabPFN COMBINED split . FAILED\|TabPFN COMBINED queue stopped"; do
    sleep 300
  done
  CUDA_VISIBLE_DEVICES=0 nice -n 5 taskset -c 16-19 $PY -W ignore scripts/mahalamia_ablation.py --datasets COMBINED \
    --generators tabpfn --workers 2 > $L/fill_tabpfn_combined_refresh.log 2>&1
  nice -n 5 taskset -c 16-19 $PY -W ignore scripts/eval_bio.py --datasets COMBINED --generators tabpfn >> $L/fill_tabpfn_combined_refresh.log 2>&1
  tcga_tables_tabpfn_combined 16-19 "3 4 5" >> $L/fill_tabpfn_combined_refresh.log 2>&1
  stamp "fill: TabPFN COMBINED tables refreshed"
  for s in 1 2 3 4 5; do
    [ -e $L/STOP_TABPFN ] && { stamp "TabPFN GTEx queue stopped before split $s"; break; }
    CUDA_VISIBLE_DEVICES=0 nice -n 5 taskset -c 16-19 $PY -W ignore scripts/build_targets.py --dataset GTEX \
      --generators tabpfn --splits $s > $L/fill_tabpfn_gtex_s$s.log 2>&1
    [ -f artifacts/targets/GTEX/tabpfn/split_$s/synthetic_data.csv ] || { stamp "TabPFN GTEx split $s FAILED"; break; }
    gtex_tables 16-19 tabpfn "$s" >> $L/fill_tabpfn_gtex_s$s.log 2>&1
    stamp "TabPFN GTEx split $s built and scored"
  done
  stamp "FILL_TABPFN_GTEX_DONE"
) &
wait
