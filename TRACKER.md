# Steven's open items

One line per item Steven has asked for, with its status and where the work
lives.  The long-form record is `EXPERIMENTS.tex` and `results/FINDINGS.md`.
Last updated 2026-10-05.

## Running

| Item | Status | Where |
|---|---|---|
| **MeLoMIA-TabSyn, black box and white box** (2026-10-05), parallel to the MeLoMIA-ND / CVAE rows; plus the existing attacks on the tuned-TabSyn and TabPFN targets | **queued and running unattended.** Backend, white-box path and runner written and smoke-tested. Order: BRCA K=10 -> score -> K=30 -> score -> other generators as targets -> COMBINED (K=20). BRCA is ~90 GPU-hours, COMBINED about four times that. | `scripts/queue_melomia_tabsyn.sh`; `configs/experiments/melomia_tabsyn_{brca,combined}.yaml`; `logs/melomia_tabsyn/*/progress.log`; `docs/SOTA_GENERATORS.md` (last section) |
| **State-of-the-art generators for the white paper** (2026-10-03; your three decisions 2026-10-04): TabSyn, TabPFN, DPSynth, DP-CVAE as ordinary targets; preprocessing as one swept parameter | **TabPFN-3.5 gene by gene**: BRCA all five splits built (discriminator AUC ~0.63, MahalaMIA 0.572 on splits 1, 2, 4, 5; split 3 not yet scored); COMBINED split 1 building to time it. **TabSyn tuned**: diagnosed (latent scale + training schedule + rare-class under-sampling + quantile scaling) and fixed, TSTR F1 0.629 -> 0.776 and per-gene error 0.424 -> 0.089 on split 1, but GAN-leaks rises to 0.782. **Five splits (2026-10-05)**: utility ratio 0.89 (0.77-1.00), discriminator AUC 0.96, GAN-leaks 0.773, MahalaMIA 0.662, MAMA-MIA 0.555; split 1, which the recipe was tuned on, is one of the two best. **DPSynth**: runs on a 200-most-variable-gene cohort (`BRCA_HVG200`), every other generator rebuilt there; star beats MST, our DP-PGM star beats DPSynth's. MeLoMIA / MAMA-MIA not yet run on any of these. Not committed. | `docs/SOTA_GENERATORS.md`; `configs/experiments/grid_sota_*.yaml`, `grid_hvg200_brca.yaml`; `results/preprocess_ablation.csv`, `results/gene_subset_ablation.csv` |
| **RedSigma, the other winning CAMDA-26 attack** (2026-10-04; Tucker et al., code shared with us): get it working, reproduce their results, run it on our grid | **reproduced and run.** Their repo reports no metrics, only the 8 prediction files they submitted; our port regenerates all 8 (rank correlation 1.000). On our targets, 5 splits, mean AUC as submitted: BRCA MVN 1.000, CVAE 0.795, ND 0.516, both DP-PGMs 0.50-0.51; COMBINED MVN 0.681, CVAE 0.847, ND 0.506, both DP-PGMs 0.50. TabSyn / TabPFN / DP-CVAE: BRCA split 1 only (0.49-0.62). **Your two ideas (2026-10-05), also recorded:** their Gaussian rule on every generator lifts BRCA CVAE to 0.995 and ND to 0.813, but lowers COMBINED CVAE to 0.728 (ND 0.592); swapping rules helps ND (best 0.813 BRCA, 0.616 COMBINED) and never DP-PGM (every rule 0.50-0.51). What to report is undecided. To attack a new generator: add its name to the config and re-run it. **On main** | `mia/attacks/redsigma.py`, `tests/test_redsigma.py`, `configs/experiments/redsigma_{brca,combined}.yaml`; their code in `~/ELSA_REDSIGMA` |
| **MeLoMIA per-record calibration** (2026-10-03, your note from the TimeDiff audit): z-score each record against itself under the other synth-shadows. On by default; black box unchanged | **done** (FINDINGS §10q); slide table regenerated. MeLoMIA-ND on ND: 0.858 → 0.969 (BRCA), 0.648 → 0.802 (COMBINED). MeLoMIA-CVAE on CVAE: 0.731 → 0.870 (COMBINED), 0.798 → 0.807 (BRCA, where TPR at 1% FPR falls 0.33 → 0.25). DP-PGMs stay at chance | `notes/note_per_record_calibration.md` §9; `results/SLIDE_TABLE.md`; trial `results/per_record_calibration/` |
| **MeLoMIA ablations, one protocol** (2026-10-03): calibration on/off × shadow count × synth vs real shadows × selection folds; 52 arms, 5 generators × 5 splits each, every arm at 60 search trials | **done 2026-10-04, 52 of 52 arms, no discrepancies** (FINDINGS §10q). Both cohorts agree: 5 calibrated shadows beat the largest uncalibrated stack; synth-shadows beat real shadows on the probe's own family (real shadows ahead only for MeLoMIA-ND on MVN); selection folds make no difference. One exception: BRCA MeLoMIA-CVAE gains nothing and loses TPR at 1% FPR at every shadow count | `results/MELOMIA_ABLATIONS.md` (+ `melomia_ablations.csv`, `melomia_ablations_arms.csv`); arms defined in `scripts/melomia_ablations.py` |
| MAMA-MIA v2 with the new "grid" edge estimator (black box), 108 dp_quantile targets | **done** (108/108, no failures); write-up pending | `logs/mamamia_v2_grid.log` → `results/mamamia_v2.csv` (edges=grid) |
| PQRS part 2 with your selector's `max_degree` cap: (50,15,5) at 16 bins; (200,50,10) at 8 bins | **done**: fits now, ties the star (FINDINGS §10m); v2 queued | `logs/pgm_pqrs_maxdeg.log` → `results/pgm_pqrs_maxdeg{,16}.csv` |
| Star with hairs (your 2026-09-24 idea), hairs free to grow into trees: l ∈ {0,20,50,100,200,400,977} × with/without 1-ways, 112 targets | **done**: every hair makes correlation worse; the star's direct subtype links win (FINDINGS §10l). v2 queued | `configs/experiments/pgm_hairy_star.yaml`, `logs/pgm_hairy_star.log` |
| Star with hairs, your literal version: hairs are disjoint gene pairs; l ∈ {20,50,100,200,489} × with/without 1-ways, 80 targets | **done** (88 incl. 1-way floor, no failures); write-up after v2 | `configs/experiments/pgm_hairy_pairs.yaml`, `logs/pgm_hairy_queue.log` |
| 1-way-only floor (genes independent, no label link), 8 targets | queued behind the above | `configs/experiments/pgm_baselines.yaml` |
| **Group-meeting slide table** (2026-09-27, revised 2026-09-30): every attack on MVN/CVAE/ND, the challenge's real CAMDA-26 DP-PGM (`pgg`, replaces the earlier stand-in) and DP-PGM new; 3 fidelity rows; MahalaMIA PCA/ridge variants; 5-column PCA and UMAP grids | **done**. The new DP-PGM matches the release on utility and is worse on correlations; its gain is the end-to-end guarantee. Every attack except MAMA-MIA is at chance on both (FINDINGS §10o, §10p) | `results/SLIDE_TABLE.md` (`python scripts/slide_table.py`); page https://claude.ai/artifact/Kncd2sC1XLtYDNnAMKSgM5 |
| MAMA-MIA v2 on all the new targets, then rebuild the one-place table | **done** (214 evals, no failures); FINDINGS §10n, table rebuilt | `scripts/run_v2_followup.sh`, `logs/v2_followup.log` |

## Done, for Steven to read

| Item | Where |
|---|---|
| **MeLoMIA-E2E** (Steven's idea, 2026-10-05): proxy / synth-shadow weights and the classifier trained as one module. BRCA, CVAE targets, 5 splits: AUC 0.823, TPR@1%FPR 0.41, against MeLoMIA-CVAE 0.807 / 0.245 and its own frozen control 0.790 / 0.318. No change on MVN / ND targets. One seed, CVAE backend only; limited by label memorisation across 24 training shadows | attack `melomia_e2e_cvae` (`mia/attacks/melomia_e2e/`), `configs/experiments/melomia_e2e_brca.yaml`, `results/melomia_e2e_brca.csv` (the 60 runs are in `results/runs`, not yet in `results/index.csv`) |
| **One place for every DP-PGM architecture**: catalogue, what has been run under which conditions, head-to-head tables (same bins, ε, split) | `results/PGM_ARCHITECTURES.md` (regenerate: `python scripts/pgm_architectures.py`); per-target rows in `results/pgm_architectures.csv` |
| k/l forest sweep (288 targets): it does not beat the star; k (genes tied to subtype) is the only lever, and gene pairs barely help | FINDINGS §10k, `results/pgm_forest_sweep.csv` |
| MAMA-MIA v2 on the forests: black-box 0.58–0.67 at ε=10, 0.65–0.995 at ε=1000; white-box 0.74–0.78 at ε=10 (DP bound 0.89) | FINDINGS §10k |
| PQRS retry: (50,15,5) no better than the star; (200,50,10) cannot be fitted (15–16-gene junction-tree cliques) | FINDINGS §10k |
| Many-shadow selection on the structure-sweep targets | `results/mamamia_v2.csv` (cliques=shadow) |
| Why DP-PGM works now: the zCDP note for the group, Daniil, and the other agent | `docs/WHY_DP_PGM_WORKS_NOW.md` (also in the generator repo); page https://claude.ai/artifact/RfatVJKYn8saBiYnJy3KXT (share it first) |
| PCA / UMAP of real vs synthetic (CVAE, ND, four DP-PGM configs) | `results/figures/fidelity_{pca,umap}_s1.png`; script `scripts/fidelity_embeddings.py` |
| Edge recovery from density steps (task 18) | `mia/attacks/mamamia_v2.py` (`edges="steps"`); a negative result, see "Needs Steven" below |
| Disk pruning | 3.9 GB freed from BROKEN targets (`results/BROKEN.md`); then 7.5 GB of legacy shadow-generator weights in `mia_output/` (Steven OK'd; synthetic data, features, splits, classifiers kept; list in `mia_output/PRUNED_2026-09-24.txt`).  9.2 GB free.  2026-09-25: disk hit 2 GB and paused the hairy sweep; PGM target checkpoints now drop the fitted model (never reloaded; edges, tables and rho kept), 19 GB freed, `scripts/slim_pgm_checkpoints.py`, list in `artifacts/targets/SLIMMED.txt` |

## Needs Steven

0. **TabPFN on COMBINED.** 3,458 training rows instead of 871; split 1 is queued to measure the cost before committing to five. **Shadow-based attacks on the new targets** (MeLoMIA needs a shadow generator per target type: which ones do you want first?).

0b. **RedSigma: two things.** (a) Do you have their leaderboard scores? Their repo has none, so the only check was against their submitted prediction files. (b) Their attack picks its rule from the generator's name and has none for TabSyn / TabPFN, where their code falls back to the Gaussian rule: keep that fallback as "RedSigma" for the new generators, or report the best of their four rules?

1. **Which DP-PGM goes in the paper.** Leaning star (= forest k=978) at 16 bins; the forest sweep did cover k=500/978 and l up to 400, and none beat it.  PQRS part 2 is in and ties it (§10m).
2. **Class-centring as headline or ablation.** It subtracts each subtype's mean attack score before ranking.
3. **The abstract's DP-PGM targets and its MAMA-MIA v1 run**: where did they come from? It reports 0.53 / 0.56; our exact rebuild of the CAMDA-25 generator gives 0.51 / 0.50.
4. **The abstract's MeLoMIA-ND numbers (0.62 / 0.74)** came from a different setup than its Methods describe: the meta-classifier trained on the real ND targets of other splits with true labels; only splits 4–5 were scored, on a balanced subset. Ours follows the described method: 0.97 / 0.80 with per-record calibration (0.86 / 0.65 without). Proposed fix: report ours.

Decided 2026-09-24: BRCA's optimistic held-out aux stays for now (better aux strategy and more datasets later); build the grid edge estimator (done, now running); `mia_output/` pruned.

## Later

- Splits 2–3 for the forest sweep, once split 1 has been read.
- Classic MST with the label as a free node (the only architecture in the catalogue not yet run), if Steven wants it.
- Offered 2026-10-02 to 10-04, not run, waiting on Steven: donor-level spectral bump test on the scRNA-seq data; an attack that scores a record's distance off the synthetic data's span; MeLoMIA calibration variants (non-member-only reference, raw plus calibrated features, re-tuned search space); why BRCA MeLoMIA-CVAE does not gain from calibration; republish the group-meeting page (https://claude.ai/artifact/Kncd2sC1XLtYDNnAMKSgM5), which still shows uncalibrated MeLoMIA rows; fix the opening of `docs/WHY_DP_PGM_WORKS_NOW.md`.
- Earlier open items: more ND synth-shadow splits; GPU items (MeLoMIA K sweep, sweep-axis ablation, matched base shadows).
