# Steven's open items

One line per item Steven has asked for, with its status and where the work
lives.  The long-form record is `EXPERIMENTS.tex` and `results/FINDINGS.md`.
Last updated 2026-09-24.

## Running

| Item | Status | Where |
|---|---|---|
| (nothing of ours is running) | | |

## Done, for Steven to read

| Item | Where |
|---|---|
| k/l forest sweep (288 targets): it does not beat the star; k (genes tied to subtype) is the only lever, and gene pairs barely help | FINDINGS §10k, `results/pgm_forest_sweep.csv` |
| MAMA-MIA v2 on the forests: black-box 0.58–0.67 at ε=10, 0.65–0.995 at ε=1000; white-box 0.74–0.78 at ε=10 (DP bound 0.89) | FINDINGS §10k |
| PQRS retry: (50,15,5) no better than the star; (200,50,10) cannot be fitted (15–16-gene junction-tree cliques) | FINDINGS §10k |
| Many-shadow selection on the structure-sweep targets | `results/mamamia_v2.csv` (cliques=shadow) |
| Why DP-PGM works now: the zCDP note for the group, Daniil, and the other agent | `docs/WHY_DP_PGM_WORKS_NOW.md` (also in the generator repo); page https://claude.ai/artifact/RfatVJKYn8saBiYnJy3KXT (share it first) |
| PCA / UMAP of real vs synthetic (CVAE, ND, four DP-PGM configs) | `results/figures/fidelity_{pca,umap}_s1.png`; script `scripts/fidelity_embeddings.py` |
| Edge recovery from density steps (task 18) | `mia/attacks/mamamia_v2.py` (`edges="steps"`); a negative result, see "Needs Steven" below |
| Disk pruning | 3.9 GB freed from BROKEN targets only (`results/BROKEN.md`); a guard pauses our sweeps below 3 GB free |

## Needs Steven

1. **Which DP-PGM goes in the paper.** Undecided; revisit after the forest sweep and the PCA/UMAP plots.
2. **BRCA held-out aux.** There is no real held-out set.  Non-member candidates stand in and are labelled optimistic.  Is that acceptable, or should BRCA's shadow results be left out?
3. **Edge recovery.** The white-box gain needs exact edges: σ = 0.03 of edge error erases it.  dp_quantile's edges form an exact arithmetic progression inside each public 0.5-wide grid cell.  Worth building a structured estimator on that?
4. **Disk: 1.9 GB free on the shared disk** (other users are writing too).  `mia_output/` (8.6 GB, legacy v1 pipeline, Feb–Apr 2026) is the largest thing I could free, but it may hold the abstract's original artifacts.  Delete, archive, or keep?
5. **Class-centring as headline or ablation.** It subtracts each subtype's mean attack score before ranking; see the report.

## Later

- Splits 2–3 for the forest sweep, once split 1 has been read.
- MST with the label as a node (the data picks k + l = 978), if the forest results make it worth running.
- Earlier open items: more ND synth-shadow splits; GPU items (MeLoMIA K sweep, sweep-axis ablation, matched base shadows).
