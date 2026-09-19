# Superseded one-off orchestration

These sequenced the first pass over the BRCA and COMBINED grids, one script per
situation, written as the work was discovered.  They are kept because the logs
they produced are referenced in `results/` and the commit history, not because
they should be run again — several predate the thread-capping fix in
`mia/attacks/melomia/meta.py:n_jobs()` and serialise the two MeLoMIA attacks
unnecessarily.

| file | what it did | replaced by |
|---|---|---|
| `orchestrate_brca.sh` | BRCA grid, splitting the ND shadow stack across two GPUs | `scripts/orchestrate_rest.sh` |
| `orchestrate_after_brca.sh` | tuned grid, ablation and COMBINED, ordered by value per hour | `scripts/orchestrate_rest.sh` |
| `orchestrate_combined.sh` | COMBINED grid at K=20 | `scripts/orchestrate_rest.sh` |
| `queue_combined.sh` | waited for the BRCA orchestrator to release both GPUs | `scripts/orchestrate_rest.sh` (one script, sequential stages) |
| `run_brca_rest.sh` | BRCA targets plus everything except MeLoMIA-ND | `scripts/run_experiment.py --only` |
| `run_pgm_targets.sh` | DP-PGM targets on CPU alongside the GPU work | `scripts/build_targets.py` |

The current entry points are:

* `scripts/orchestrate_rest.sh` — the outstanding queue, two attacks per stage,
  bounded core share per process.
* `scripts/after_orchestrate.sh` — what runs once that exits.
* `scripts/shadow_loop.sh` — extra workers to saturate a GPU during a shadow
  build; safe to run alongside a live orchestration, since shadows are claimed
  with lock files.
