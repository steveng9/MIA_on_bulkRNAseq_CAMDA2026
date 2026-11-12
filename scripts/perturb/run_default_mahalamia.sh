#!/bin/bash
# Re-score every robustness experiment with the paper's MahalaMIA (own class,
# class covariance averaged with the pooled one; adopted 2026-11-11), so that no
# table mixes MahalaMIA variants.  Outputs carry the tag _default.  CPU only.
cd ~/MIA_on_bulkRNAseq_CAMDA2026
PY=/home/golobs/miniconda3/envs/recon_/bin/python
L=logs/paper_fill
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
export PERTURB_TAG=_default
stamp() { echo "### [$(date -u +%m-%d\ %H:%M)] $*" | tee -a $L/progress.log; }
run() { nice -n 5 taskset -c 20,21,23,44,45,47 $PY -W ignore "$@"; }
for ds in BRCA COMBINED; do
  PERTURB_ATTACKS="MahalaMIA" run scripts/perturb/aux_mismatch.py --dataset $ds --jobs 6 > $L/default_aux_$ds.log 2>&1
  stamp "default MahalaMIA: aux mismatch $ds"
  PERTURB_ATTACKS="MahalaMIA" run scripts/perturb/candidate_mismatch.py --dataset $ds --jobs 6 > $L/default_cand_$ds.log 2>&1
  stamp "default MahalaMIA: candidate mismatch $ds"
  PERTURB_ATTACKS="MahalaMIA|MahalaMIA (no reference)" run scripts/perturb/donor_linked.py --dataset $ds --jobs 6 > $L/default_donor_$ds.log 2>&1
  stamp "default MahalaMIA: donor-linked $ds"
done
PERTURB_ATTACKS="MahalaMIA|MahalaMIA (no reference)" run scripts/perturb/gtex_donor_linked.py --jobs 6 > $L/default_gtex_donor.log 2>&1
stamp "default MahalaMIA: GTEx other tissue"
for ds in BRCA COMBINED; do
  PERTURB_ATTACKS="MahalaMIA" run scripts/perturb/gene_subsets.py --dataset $ds --jobs 6 --tag _default > $L/default_genes_$ds.log 2>&1
  stamp "default MahalaMIA: gene subsets $ds"
done
stamp "DEFAULT_MAHALAMIA_PERTURB_DONE"
