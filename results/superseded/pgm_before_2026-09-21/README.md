# DP-PGM runs superseded on 2026-09-21

Every run here attacked a `pgm` target but predates target fingerprints, so it
cannot be told apart from a current run by name or tag alone.  They are kept for
the record, out of `runs/` so that `index.csv` and `scripts/reindex.py` no
longer see them.

* Timestamped before 2026-09-20 ~22:00 UTC (564 runs): attacked the broken
  generator -- permuted gene order and basic composition (FINDINGS 7), 12 of
  them unlabelled ad-hoc runs from 2026-09-18.  These
  are the "DP-PGM at chance" numbers, and are void.
* Timestamped 2026-09-20 22:02-22:03 (10 runs, MAMA-MIA): attacked the
  corrected targets under an earlier MAMA-MIA parameter set; superseded by
  identical re-runs that carry the fingerprint.

Current DP-PGM runs record `target.fingerprint` in their config.json.
