# Adding a bulk RNA-seq cohort

One YAML file here registers a cohort under the file's name; nothing else needs
to change.  `python scripts/build_targets.py --dataset <NAME> ...` and every
attack, fidelity metric and experiment config then accept it.

```yaml
# configs/datasets/GTEX.yaml
expression: ~/data/GTEx/gene_counts.tsv   # matrix, first column = row ids
orientation: genes_x_samples              # or samples_x_genes (default)
labels: ~/data/GTEx/sample_meta.csv       # indexed by sample id
label_col: tissue
prepare: cpm+log1p+hvg:2000               # cohort-level preparation (below)
reference: ~/data/GTEx/held_out.tsv       # optional: known non-members (D_aux)
splits: ~/data/GTEx/splits.yaml           # optional: else drawn once and cached
n_splits: 5
test_frac: 0.2
split_seed: 42
```

## Two levels of preprocessing

**Cohort-level (`prepare`, here).**  Runs once on load and defines the space
everything else calls "raw" -- what the DESeq2 VST and the landmark-gene filter
were for the TCGA challenge data.  Steps, chained with `+`:

| step | effect | estimated across samples? |
|---|---|---|
| `cpm` | scale each sample to 10^6 counts | no |
| `log1p`, `log2p1` | log(1+x), log2(1+x) | no |
| `dropconst` | drop zero-variance genes | yes |
| `hvg:k` | keep the k most variable genes | yes |
| `genes:<file>` | keep the listed genes, in order (e.g. L1000 landmarks) | no |

Steps estimated across samples see members and non-members alike, as the
challenge's VST did, so they do not separate the two.  If a cohort already
arrives transformed (VST, log-TPM), leave `prepare` out.

**Model-level (`preprocess`, a generator parameter).**  Fitted on one training
split, inverted on the way out; see `mia/preprocessing.py`.  It is part of the
target's name (`tabsyn@preprocess=standard+pca:64`), so a preprocessing
ablation is an ordinary target sweep: `scripts/preprocess_ablation.py`.

## Things to check for a new cohort

- Value range.  `dpsynth` and `dpcvae@preprocess=fixed:lo:hi` take *public*
  bounds; the defaults (0, 24) are for DESeq2-VST.  log1p-CPM lies in about
  (0, 14).
- Class sizes.  MVN needs at least two training samples per class, and the
  ND recipe SMOTE-upsamples, which needs more neighbours than a class of
  three has.
- No published NoisyDiffusion data exists outside the challenge cohorts, so
  `nd` is always trained here.
