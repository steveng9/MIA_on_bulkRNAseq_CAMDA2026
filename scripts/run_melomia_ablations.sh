#!/usr/bin/env bash
# MeLoMIA ablation queue (arms and order: scripts/melomia_ablations.py).  Two workers,
# 12 threads each, on the cores and GPU we are allowed.  Resumable: an arm with a .done
# marker is skipped; fitted classifiers are cached, so a restarted arm only re-scores.
# A cohort's arms start only once run_melomia_prc.sh has finished that cohort, because
# its headline arms reuse the classifiers that script fits.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD
PY=/home/golobs/miniconda3/envs/recon_/bin/python
D=logs/melomia_ablations; mkdir -p $D
Q=$D/queue.txt
[ -f $Q ] || $PY scripts/melomia_ablations.py queue > $Q

released() {  # cohort
  if [ "$1" = brca ]; then grep -q "=== brca melomia_cvae done" logs/melomia_prc.log
  else grep -q "MELOMIA PRC COMPLETE" logs/melomia_prc.log; fi
}
pop() {  # prints "cohort arm" of the first runnable arm and removes it from the queue
  ( flock 9
    $PY scripts/melomia_backfill_protocol.py > $D/backfill.log 2>&1
    while read -r c a; do
      [ -f $D/${c}_$a.done ] && { sed -i "/^$c $a\$/d" $Q; continue; }
      if released $c; then sed -i "/^$c $a\$/d" $Q; echo "$c $a"; exit 0; fi
    done < <(cat $Q)
  ) 9> $D/queue.lock
}
worker() {  # cores
  while [ -s $Q ]; do
    job=$(pop)
    if [ -z "$job" ]; then sleep 120; continue; fi
    set -- $job
    echo "$(date -Is) start $1 $2 (cores $WCORES)" >> $D/progress.log
    if CUDA_VISIBLE_DEVICES=0 MIA_N_JOBS=12 OMP_NUM_THREADS=12 taskset -c $WCORES $PY -W ignore \
        scripts/run_experiment.py configs/experiments/melomia_ablations_$1.yaml --only $2 \
        > $D/$1_$2.log 2>&1; then
      touch $D/$1_$2.done; echo "$(date -Is) done  $1 $2" >> $D/progress.log
    else
      echo "$(date -Is) FAILED $1 $2 (see $D/$1_$2.log)" >> $D/progress.log
    fi
    ( flock 9; $PY scripts/melomia_ablations.py table > $D/table.log 2>&1 ) 9> $D/table.lock
  done
}
WCORES=12-17,36-41 worker &
WCORES=18-23,42-47 worker &
wait
$PY scripts/melomia_ablations.py table
CUDA_VISIBLE_DEVICES= taskset -c 12-23,36-47 $PY scripts/slide_table.py > $D/slide_table.log 2>&1
echo "MELOMIA ABLATIONS COMPLETE $(date -Is)" | tee -a $D/progress.log
