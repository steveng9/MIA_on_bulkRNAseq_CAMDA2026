# Known-broken artifacts — do not use

## BROKEN_DP_EDGES (found 2026-09-23)

**What:** every DP-PGM target built with `binning=dp_quantile` or `binning=dp_uniform`
before generator commit `cd5d1d5` (private-pgm-rnaseq-camda2026), and every attack
run, CSV row and FINDINGS number computed on them.

**Why:** the DP bin edges were read off a noisy per-gene histogram with negative
cells clipped to zero. Each empty cell kept its positive noise, and ~40 of the
48 cells are empty for a typical gene. That phantom mass pushed the 0.5% / 99.5%
bounds to the ends of the grid even at ε=1000: median **2.0 / 22.0 against a
true 9.5 / 13.5** (BRCA). The data is effectively 4 equal bins over (2, 22), so
~85% of each gene lands in one cell. The DP accounting was correct; the
release was simply useless, and every quality and attack number on it
measures that bug, not DP.

**How to recognise them:** the target name contains `binning=dp_` and does **not**
contain `edge_estimator=threshold`. The rule is `mia.targets.broken_reason()`.

**Guards in place:**
- `mia.targets.load_target` / `build_target` raise `BrokenTargetError` (pass
  `allow_broken=True` only to reproduce the bug);
- `mia.runs.load_index()` adds a computed `status` column (`ok` / `BROKEN_DP_EDGES`);
- stamped on disk by `scripts/mark_broken.py` (idempotent):
  - target dirs: `BROKEN_DP_EDGES.txt`, plus `meta.json` `status` / `broken_reason`;
  - run dirs: `BROKEN_DP_EDGES.txt`, plus `config.json` notes prefixed `BROKEN_DP_EDGES; `;
  - `results/index.csv`: notes prefixed `BROKEN_DP_EDGES; ` (2220 runs);
  - `results/pgm_eps_sweep.csv` (180/380 rows) and `results/pgm_attack_binning.csv`
    (540/1140 rows): a `status` column.

**Affected write-ups:** FINDINGS §10e, 10f, 10g and 10h for their dp_* numbers
(each carries a banner), and the killed ε=1000 4-attack grid
(`configs/experiments/grid_dpsafe*_*.yaml`, `scripts/run_dpsafe_grid*.sh`,
which now refuse to run).

**Not affected:**
- `binning=uniform` (public fixed range, no estimation);
- legacy `binning=quantile` (non-DP by design, documented in §10a);
- non-PGM generators;
- everything with `edge_estimator=threshold`, including all of
  `results/pgm_structure_sweep.csv`.

**Replacement:** `scripts/pgm_structure_sweep.py` (FINDINGS §10i).
