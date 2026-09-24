# Steven's open items

One line per item Steven has asked for, with its status and where the work
lives.  The long-form record is `EXPERIMENTS.tex` and `results/FINDINGS.md`.
Last updated 2026-09-24 (evening).

## Running

| Item | Status | Where |
|---|---|---|
| MAMA-MIA v2 with the new "grid" edge estimator (black box), 108 dp_quantile targets | running | `logs/mamamia_v2_grid.log` → `results/mamamia_v2.csv` (edges=grid) |
| PQRS part 2 with your selector's `max_degree` cap: (50,15,5) at 16 bins; (200,50,10) at 8 bins | running, one fit at a time | `logs/pgm_pqrs_maxdeg.log` → `results/pgm_pqrs_maxdeg{,16}.csv` |
| Star with hairs (your 2026-09-24 idea), hairs free to grow into trees: l ∈ {0,20,50,100,200,400,977} × with/without 1-ways, 112 targets | running | `configs/experiments/pgm_hairy_star.yaml`, `logs/pgm_hairy_star.log` |
| Star with hairs, your literal version: hairs are disjoint gene pairs; l ∈ {20,50,100,200,489} × with/without 1-ways, 80 targets | queued behind the above | `configs/experiments/pgm_hairy_pairs.yaml`, `logs/pgm_hairy_queue.log` |
| 1-way-only floor (genes independent, no label link), 8 targets | queued behind the above | `configs/experiments/pgm_baselines.yaml` |
| MAMA-MIA v2 on all the new targets, then rebuild the one-place table | queued, starts when the above finish | `scripts/run_v2_followup.sh`, `logs/v2_followup.log` |

## Done, for Steven to read

| Item | Where |
|---|---|
| **One place for every DP-PGM architecture**: catalogue, what has been run under which conditions, head-to-head tables (same bins, ε, split) | `results/PGM_ARCHITECTURES.md` (regenerate: `python scripts/pgm_architectures.py`); per-target rows in `results/pgm_architectures.csv` |
| k/l forest sweep (288 targets): it does not beat the star; k (genes tied to subtype) is the only lever, and gene pairs barely help | FINDINGS §10k, `results/pgm_forest_sweep.csv` |
| MAMA-MIA v2 on the forests: black-box 0.58–0.67 at ε=10, 0.65–0.995 at ε=1000; white-box 0.74–0.78 at ε=10 (DP bound 0.89) | FINDINGS §10k |
| PQRS retry: (50,15,5) no better than the star; (200,50,10) cannot be fitted (15–16-gene junction-tree cliques) | FINDINGS §10k |
| Many-shadow selection on the structure-sweep targets | `results/mamamia_v2.csv` (cliques=shadow) |
| Why DP-PGM works now: the zCDP note for the group, Daniil, and the other agent | `docs/WHY_DP_PGM_WORKS_NOW.md` (also in the generator repo); page https://claude.ai/artifact/RfatVJKYn8saBiYnJy3KXT (share it first) |
| PCA / UMAP of real vs synthetic (CVAE, ND, four DP-PGM configs) | `results/figures/fidelity_{pca,umap}_s1.png`; script `scripts/fidelity_embeddings.py` |
| Edge recovery from density steps (task 18) | `mia/attacks/mamamia_v2.py` (`edges="steps"`); a negative result, see "Needs Steven" below |
| Disk pruning | 3.9 GB freed from BROKEN targets (`results/BROKEN.md`); then 7.5 GB of legacy shadow-generator weights in `mia_output/` (Steven OK'd; synthetic data, features, splits, classifiers kept; list in `mia_output/PRUNED_2026-09-24.txt`).  9.2 GB free |

## Needs Steven

1. **Which DP-PGM goes in the paper.** Leaning star (= forest k=978) at 16 bins; the forest sweep did cover k=500/978 and l up to 400, and none beat it.  Revisit once PQRS part 2 is in.
2. **Class-centring as headline or ablation.** It subtracts each subtype's mean attack score before ranking.

Decided 2026-09-24: BRCA's optimistic held-out aux stays for now (better aux strategy and more datasets later); build the grid edge estimator (done, now running); `mia_output/` pruned.

## Later

- Splits 2–3 for the forest sweep, once split 1 has been read.
- Classic MST with the label as a free node (the only architecture in the catalogue not yet run), if Steven wants it.
- Earlier open items: more ND synth-shadow splits; GPU items (MeLoMIA K sweep, sweep-axis ablation, matched base shadows).
