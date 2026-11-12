# Side tasks for the 2026-11-13 team meeting

Four items Steven's team asked for, worked in parallel with the manuscript
session, 2026-11-12. Summary here; full detail in the sections below.

1. **GTEx**: ten tissues, matched to COMBINED's twelve TCGA projects; 4,168
   candidate+reference samples from 923 donors, about 81% of COMBINED's 5,147.
2. **Four new generators on COMBINED**: TabSyn done (5 splits); TabPFN
   building (another session's queue, splits 2-5 in progress, ETA hours);
   DP-CVAE and DPSynth were NOT started on COMBINED before today — both built
   today (see status below).
3. **Biological quality metrics**: `results/bio.csv` already covers two of
   Hakime's four methods (DE recovery, co-expression) for every generator,
   every cohort. The other two (discriminative / HVG genes between synthetic
   and reference) are not computed; mapped below with a recommendation.
4. **Gene-subset MIA sweep**: new result, `results/perturb/gene_subset_attack_{BRCA,COMBINED}.csv`.

## 1. GTEx cohort

Ten GTEx tissues, chosen to match COMBINED's twelve TCGA projects (`docs/GTEX.md`):

| GTEx tissue | TCGA project(s) |
|---|---|
| Breast - Mammary Tissue | BRCA |
| Kidney - Cortex | KIRC, KIRP |
| Lung | LUAD, LUSC |
| Prostate | PRAD |
| Colon - Transverse | COAD |
| Ovary | OV |
| Liver | LIHC |
| Esophagus - Mucosa | ESCA |
| Pancreas | PAAD |
| Skin - Sun Exposed (Lower leg) | SKCM |

Samples per tissue are not separately tabulated (the cohort build logs totals,
not a per-tissue breakdown); if the team wants a per-tissue table for the
slide, that's a five-minute re-read of `~/data/GTEx/processed` and I can add
it on request.

**Size.** 4,168 samples from 923 donors total (671 reference + 3,497 across
the five candidate splits). COMBINED is 5,147 samples (824 reference + 4,323
candidates). **GTEx is about 81% of COMBINED's sample size** (4,168 / 5,147).
GTEx is donor-grouped (each donor's ~4.5 of these ten tissues sit on one side
of a split only); COMBINED has no donor structure to preserve.

## 2. Four new generators on TCGA-COMBINED

| generator | status on COMBINED (978 genes) | where |
|---|---|---|
| TabSyn (tuned recipe) | **done**, 5 splits, utility_ratio 0.868 | `artifacts/targets/COMBINED/tabsyn@...` |
| TabPFN | split 1 done; **splits 2-5 building now** in another session's queue (`scripts/queue_tabpfn_combined.sh`, GPU 0, ~15h/split) — not touched, just reporting | `logs/paper_fill/` |
| DP-CVAE | was **not built on COMBINED before today** (only on BRCA / BRCA_HVG200). Built today on GPU 1 (idle), 5 splits, 113s/split | `artifacts/targets/COMBINED/dpcvae/` |
| DPSynth | 978 genes don't fit (JAX compile blows up at that width, documented wall). Built today on `COMBINED_HVG200` (200 most-variable genes), the same workaround already used for BRCA, 5 splits, CPU-only | `artifacts/targets/COMBINED_HVG200/dpsynth/` |

So as of this run: 3 of 4 are fully built on COMBINED (TabSyn, DP-CVAE, DPSynth@HVG200);
TabPFN is mid-build elsewhere. Not yet scored with the attack grid or
`eval_fidelity.py` / `eval_bio.py` for DP-CVAE / DPSynth — queued next (see
"Not done" below).

## 3. Biological quality metrics (Hakime's four methods)

Her reference, `Health-Privacy-Challenge/src/mia/models/baseline.py`:

| her method | what it measures | what we have |
|---|---|---|
| `get_supervised_DE_genes` | mutual information with class labels | **covered, different test**: `scripts/eval_bio.py` -> `results/bio.csv` does pairwise Wilcoxon DE (the CAMDA benchmark's own test, `diffexpression.R`), TPR/FPR/precision of synthetic DE genes against real DE genes, every generator, every cohort, 5 splits |
| co-expression (not in her list but in our pipeline) | — | `results/bio.csv` also has Pearson co-expression recall/precision |
| `get_discriminative_genes` | genes that best separate synthetic from reference | **not computed** |
| `get_HVG_genes` | genes whose synthetic variance diverges most from reference | **not computed** |
| `perform_pca` | — | we have PCA/UMAP *figures* (`scripts/fidelity_embeddings.py`, `scripts/umap_grid_slides.py`) but not a quantitative PCA-based quality score |

**Recommendation**: the two missing methods are about synthetic-vs-reference
divergence, which is exactly Hakime's R pipeline's job and she offered to run
it. The generators with the strongest utility (`results/fidelity.csv`,
`utility_ratio`, all cohorts) are **TabPFN (0.88-0.99) and NoisyDiffusion
(0.91-0.99)**, with CVAE and MVN close behind (0.87-0.99); the DP generators
(DP-PGM, DP-CVAE, DPSynth) trail badly (0.11-0.70). I'd send her the
synthetic CSVs for **TabPFN, ND, CVAE, MVN on BRCA and COMBINED** (already on
disk at `artifacts/targets/<DATASET>/<generator>/split_1/synthetic_data.csv`,
same format as the challenge's own releases) rather than reimplement her two
methods here — her R setup already has `scran` for the DE side we approximate
in Python, so running her own code end-to-end avoids a second
reimplementation for us to keep in sync. I have NOT packaged/sent these files;
that's a one-line "zip these five generators x two cohorts x split 1" when you
confirm.

## 4. Gene-subset MIA sweep: how many genes does an attacker need?

New script `scripts/perturb/gene_subset_attack.py` (committed), results in
`results/perturb/gene_subset_attack_{BRCA,COMBINED}.csv`. Runs the paper's
default MahalaMIA inside `mia/views.py`'s `view(genes=...)` on the existing
MVN / CVAE / NoisyDiffusion targets — no retraining, so this covers all 5
splits of both cohorts in under 5 minutes total.

Three gene-selection rules, k in {5,10,20,30,50,98,150,300,500,978}:
- **random** (the existing control)
- **hvg**: top-k highest-variance genes across the whole cohort
- **hallmark** (BRCA only, fixed k=27): a breast-cancer panel independent of
  this cohort's data — PAM50 intrinsic-subtype genes plus core driver/EMT
  genes (ESR1, BRCA1/2, TP53, PIK3CA, GATA3, ...), symbol -> Ensembl via
  NCBI's gene_info, intersected with the 978 landmark genes (27 of ~78
  candidate symbols are on the landmark panel — most famous BC genes, e.g.
  ESR1, PGR, MKI67, BRCA2, CDH1, are simply not among the 978).
  List: `configs/gene_sets/brca_hallmark_27.ensembl.txt` (with symbols in
  `brca_hallmark_27.annotated.tsv`).

**Mean AUC, MahalaMIA, 5 splits:**

| dataset | rule | k=10 | k=30 | k=50 | k=98 | k=150 | k=300 | k=978 |
|---|---|---|---|---|---|---|---|---|
| BRCA MVN | random | 0.517 | 0.561 | 0.600 | 0.676 | 0.732 | 0.860 | 1.000 |
| BRCA MVN | hvg | 0.517 | 0.554 | 0.598 | 0.694 | 0.759 | 0.878 | 1.000 |
| BRCA MVN | hallmark (k=27) | — | 0.551 | — | — | — | — | — |
| COMBINED MVN | random | 0.516 | 0.565 | 0.613 | 0.737 | 0.837 | 0.948 | 0.996 |
| COMBINED MVN | hvg | 0.509 | 0.549 | 0.603 | 0.742 | 0.838 | 0.953 | 0.996 |

(full table with CVAE/ND columns and TPR@1%FPR in the CSV.)

**Reading for the meeting:**
- **AUC climbs smoothly and roughly log-linearly with gene count** for this
  attack family, from chance at k~10 to the full-978 ceiling by k~500-978.
  "AUC 0.7" on MVN/COMBINED needs about k=90-100 genes; on the harder
  NoisyDiffusion target it needs closer to k=300.
- **Which genes barely matters, for this attack.** HVG and random are within
  0.01-0.03 AUC of each other at every k, on every generator, both cohorts —
  consistent with Hakime's existing finding that random subsets already
  capture most of the signal. The 27-gene breast-cancer panel (0.53-0.55) is
  *not* meaningfully ahead of random/HVG at a matched k (~30 genes: random
  0.56, hvg 0.55).
- **Caveat before reporting "k doesn't matter" as a general claim**: this is
  one attack family (the covariance-based MahalaMIA). The central idea Hakime
  and Janita want — genes-needed-for-AUC as an interpretable privacy metric —
  should be checked against at least one more attack shape (e.g. a
  nearest-neighbour attack, GAN-leaks) before claiming the gene identity
  genuinely doesn't matter; it may matter more for an attack that reads
  specific marker genes' values rather than global covariance structure.
- Not done: COMBINED has no single "hallmark" panel (it's pan-cancer across
  12 projects) — if the team wants a biologically-targeted subset for
  COMBINED, the natural analogue is a per-project driver-gene union, or the
  12 projects' PAM50-style panels pooled. I didn't build that without a
  decision on which disease panels matter there.

## Master TODO (lower priority, not urgent — long runway to deadline)

From the meeting notes, not actioned today:

**Normalization**
- [ ] Document the TPM-to-VST distribution alignment transform in enough
      detail for Hakime (what `align_to_release` in
      `scripts/perturb/aux_mismatch.py` actually does: per-gene quantile
      mapping, nothing more).
- [ ] Biological implications of normalization mismatch for DE vs
      co-expression analysis specifically (not just attack AUC).
- [ ] The donor-mismatched-normalization experiment already has preliminary
      numbers (alignment partially recovers attack accuracy, per
      `docs/DONOR_LINKED_AND_PERTURBATIONS.md` section 7) — write this up as
      its own experiment rather than a side note.

**Donor/sample overlap**
- [ ] Verify TCGA barcode relationships (same sample / same tumor, different
      vial / different tissue sample, same donor) systematically rather than
      by the ad hoc categories already used in the donor-linked work.
- [ ] GTEx as the cleaner version of the "same donor, different tissue"
      question — already built and run (`docs/GTEX.md`, "another tissue of a
      training donor" section); needs validation before it goes in the paper,
      not a new build.

**Bigger ideas (A-D from the meeting)**
- [ ] **A. Genes-needed-for-AUC as a privacy metric.** Section 4 above is the
      first data point; needs: more attacks, more datasets, and a clean
      write-up of the metric itself (e.g. "k such that AUC crosses 0.7").
- [ ] **B. Biologically-informed threat models.** Same experiment, reframed:
      what if the fixed panel is itself the adversary's whole aux knowledge,
      not file a subset for comparison. Needs a decision on which disease
      panels to use for non-BRCA cohorts.
- [ ] **C. Normalization mismatch as an attack barrier/workaround** — already
      has results (`docs/DONOR_LINKED_AND_PERTURBATIONS.md` sec. 2-4, 7);
      needs promotion from side-finding to its own write-up.
- [ ] **D. Broader policy contribution / higher-impact venue.** Depends on A
      maturing into something rigorous across attacks and datasets; no code
      work yet, a writing/positioning decision for Steven once A has legs.
