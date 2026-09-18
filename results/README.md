# Results

Every attack evaluation is recorded here.  Nothing in this directory is
regenerated implicitly: an attack that has been run is a file on disk with its
configuration, its per-sample scores and its metrics.

```
results/
  index.csv        one row per run -- load this with pandas to build anything
  runs/<run_id>/
    config.json    the fully resolved configuration that produced the run
    scores.csv     sample_id, score, y_member -- row level, for ROC curves and
                   for anyone who wants to re-score with a different metric
    metrics.json   AUC, AUPR, TPR at several FPRs, precision@5%
  tables/          rendered grids (scripts/make_tables.py)
  figures/         paper figures (scripts/make_figures.py)
```

## Run identity

`run_id` is `<dataset>__<attack>__<generator>__s<split>__<hash>`, where the hash
covers every attack parameter.  It is deterministic, so re-running the same
configuration overwrites its own directory instead of accumulating a near
duplicate, and two configurations that differ in any parameter never collide.

That also means a run belongs to a *configuration*, not to the experiment that
happened to trigger it: if two experiment files specify the same attack
parameters, they share the run.  When selecting runs, filter on the attack's
`tag` (which encodes the hyperparameters that distinguish variants) rather than
on the experiment name — that is what `--config` does.

## Reading it

```python
from mia import runs as R

df = R.load_index()
grid = df[df.dataset == "BRCA"].groupby(["attack", "tag", "generator"]).auc.mean()

scores = R.load_run("BRCA__mahalamia__mvn__s1__abcd1234")["scores"]
```

Or from the shell:

```bash
python scripts/make_tables.py --config configs/experiments/grid_brca.yaml
python scripts/make_figures.py --config configs/experiments/grid_brca.yaml
python scripts/reindex.py                      # rebuild index.csv from disk
python scripts/reindex.py --prune-attack mahalamia --dataset BRCA --yes
```

`reindex.py` matters after an attack gains a parameter: the old runs keep their
old hash and become orphans, and averaging across both versions of an attack
would be quietly wrong.  Prune them and re-run.

## Prevalence

The challenge splits are 80/20, so 80% of the candidates in every evaluation are
members.  AUPR and accuracy inherit that prior — an attack that guesses "member"
for everything scores 0.80 AUPR and 0.80 accuracy while learning nothing.  AUC
and TPR at a fixed FPR are the metrics to read; they are the ones the CAMDA
table reports.
