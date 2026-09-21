#!/usr/bin/env bash
# Re-score every attack against the corrected DP-PGM targets, both cohorts.
# MeLoMIA's shadow stacks and meta-classifiers are reused; only the proxies on
# the released data are rebuilt (their cache now checks the target fingerprint).
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD
PY=/home/golobs/miniconda3/envs/recon_/bin/python
for cfg in grid_brca grid_combined; do
  echo "=== $cfg $(date -Is) ==="
  $PY scripts/run_experiment.py configs/experiments/$cfg.yaml \
      --only mahalamia mamamia melomia_nd melomia_cvae --generators pgm
done
echo "GRID CORRECTION COMPLETE $(date -Is)"
