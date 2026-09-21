#!/usr/bin/env bash
# Waits for the epsilon sweep to finish, then tests the two leak-free binning
# strategies on the same cores: eps 0.3 (where the leak was measured) and 10
# (the operating point), both cohorts, 5 splits.
set -u
cd "$(dirname "$0")/.."
until grep -q "SWEEP COMPLETE" logs/pgm_eps_sweep.log; do sleep 60; done
echo "sweep done, starting binning variants $(date -Is)"
PYTHONPATH=$PWD /home/golobs/miniconda3/envs/recon_/bin/python scripts/pgm_eps_sweep.py \
    --datasets COMBINED BRCA --eps 0.3 10 --workers 10 --threads 2 \
    --variants "binning=uniform" "binning=uniform,n_bins=16" "binning=dp_quantile"
echo "BINNING COMPLETE $(date -Is)"
