#!/bin/bash
# Everything Steven asked for on 2026-10-05, in order, unattended:
#   a  wait for the tuned TabSyn targets on BRCA splits 2-5
#   b  the existing shadow-based attacks (MeLoMIA-ND, MeLoMIA-CVAE, MAMA-MIA) and
#      the cheap ones on the tuned-TabSyn and TabPFN targets (cached ND/CVAE stacks)
#   c  MeLoMIA-TabSyn on BRCA, black box + white box (scripts/run_melomia_tabsyn.sh)
#   d  tuned TabSyn targets on COMBINED, then MeLoMIA-TabSyn on COMBINED
cd "$(dirname "$0")/.."
PY=/home/golobs/miniconda3/envs/recon_/bin/python
T="tabsyn@class_freq=train,diff_schedule=steps,latent_scale=std,preprocess=clip:0.001:0.999+standard"
L=logs/melomia_tabsyn; mkdir -p $L
say() { echo "### [$(date -u +%m-%d\ %H:%M)] $*" | tee -a $L/queue.log; }

until grep -q TABSYN_FINAL_DONE logs/sota/tabsyn_final.log; do sleep 60; done
say "a done: tuned TabSyn targets on BRCA"

( export MIA_N_JOBS=8 OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1
  $PY -W ignore scripts/run_experiment.py configs/experiments/grid_sota_brca.yaml --generators "$T" > $L/existing_attacks_tabsyn.log 2>&1
  $PY -W ignore scripts/run_experiment.py configs/experiments/grid_sota_brca.yaml --generators tabpfn --splits 1 2 4 5 > $L/existing_attacks_tabpfn.log 2>&1
  echo "### existing attacks done" >> $L/queue.log ) &

say "c: MeLoMIA-TabSyn on BRCA"
./scripts/run_melomia_tabsyn.sh BRCA 4 1 > $L/run_brca.log 2>&1
wait

say "d: tuned TabSyn targets on COMBINED"
for s in 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=$(( s % 2 )) $PY -W ignore scripts/build_targets.py --dataset COMBINED --generators "$T" --splits $s > $L/target_combined_s$s.log 2>&1 &
done
wait
say "d: MeLoMIA-TabSyn on COMBINED"
./scripts/run_melomia_tabsyn.sh COMBINED 3 2 > $L/run_combined.log 2>&1
say "QUEUE_DONE"
