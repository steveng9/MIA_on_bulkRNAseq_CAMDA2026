#!/bin/bash
# TabPFN on TCGA-COMBINED, splits 2 to 5 (2026-11-11), one build at a time on
# GPU 0 (a build grows to ~30 GB and took 14.7 h for split 1).  Each split is
# scored with the paper grid's attacks as soon as it exists, so the grid fills
# split by split.  Stop cleanly between splits with: touch logs/paper_fill/STOP_TABPFN
cd ~/MIA_on_bulkRNAseq_CAMDA2026
PY=/home/golobs/miniconda3/envs/recon_/bin/python
L=logs/paper_fill
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MIA_N_JOBS=4
stamp() { echo "### [$(date -u +%m-%d\ %H:%M)] $*" | tee -a $L/progress.log; }
for s in ${SPLITS:-2 3 4 5}; do
  [ -e $L/STOP_TABPFN ] && { stamp "TabPFN COMBINED queue stopped before split $s"; exit 0; }
  while [ -e $L/GPU_LOCK_TABSYN_GTEX ]; do sleep 120; done   # TabSyn GTEx shares GPU 0; wait for its builds
  if [ ! -f artifacts/targets/COMBINED/tabpfn/split_$s/synthetic_data.csv ]; then
    nice -n 5 taskset -c 22,46 $PY -W ignore scripts/build_targets.py --dataset COMBINED --generators tabpfn --splits $s \
      > $L/tabpfn_combined_s$s.log 2>&1
  fi
  [ -f artifacts/targets/COMBINED/tabpfn/split_$s/synthetic_data.csv ] || { stamp "TabPFN COMBINED split $s FAILED"; exit 1; }
  stamp "TabPFN COMBINED split $s built"
  nice -n 5 taskset -c 22,23,46,47 $PY -W ignore scripts/run_experiment.py configs/experiments/paper_grid_combined.yaml \
    --generators tabpfn --splits $s > $L/tabpfn_combined_score_s$s.log 2>&1
  nice -n 5 taskset -c 22,23,46,47 $PY -W ignore scripts/eval_fidelity.py --dataset COMBINED --generators tabpfn --splits $s \
    >> $L/tabpfn_combined_score_s$s.log 2>&1
  stamp "TabPFN COMBINED split $s scored"
done
stamp "TABPFN_COMBINED_DONE"
