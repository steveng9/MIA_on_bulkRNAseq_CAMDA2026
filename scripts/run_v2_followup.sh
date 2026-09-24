#!/usr/bin/env bash
# MAMA-MIA v2 on every new DP-PGM target once the sweeps are done (2026-09-24):
# the hairy-star, disjoint-pairs, 1-way floor and PQRS max_degree targets, plus
# a backfill of structure-sweep targets rebuilt after the first v2 run.  Then
# rebuilds results/PGM_ARCHITECTURES.md.  Our cores only.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2
PY=/home/golobs/miniconda3/envs/recon_/bin/python
until grep -q "HAIRY QUEUE COMPLETE" logs/pgm_hairy_queue.log \
      && grep -q "PQRS MAXDEG COMPLETE" logs/pgm_pqrs_maxdeg.log \
      && ! pgrep -f "mamamia_v2_eval.py --workers 2 --paths public/grid" >/dev/null; do sleep 120; done
V2="taskset -c 12-23,36-47 $PY scripts/mamamia_v2_eval.py --workers 4"
$V2 --sweep results/pgm_structure_sweep.csv --splits 1 \
    --paths public/aux public/recovered public/known recovered/recovered recovered/aux aux/aux aux/recovered
for s in pgm_hairy_star pgm_hairy_pairs pgm_baselines pgm_pqrs_maxdeg16 pgm_pqrs_maxdeg; do
    [ -f results/$s.csv ] && $V2 --sweep results/$s.csv
done
$PY scripts/pgm_architectures.py
echo "V2 FOLLOWUP COMPLETE"
