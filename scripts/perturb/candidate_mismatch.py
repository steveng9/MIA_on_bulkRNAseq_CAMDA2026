"""Attacks when the attacked profiles themselves arrive in another
representation than the release (TRACKER item 0d; the "high priority" note in
the paper's auxiliary-mismatch appendix).

The release stays as published (VST).  The candidates -- the records whose
membership the adversary tests -- are replaced by the same samples in another
normalisation or from another pipeline
(scripts/external/candidate_representations.py):

  vst             as distributed (control: the grid's setting)
  gdc_tpm         GDC's STAR pipeline, TPM
  gdc_log2tpm     the same, log2(TPM + 1)
  toil_log2tpm    UCSC TOIL recompute (STAR + RSEM), log2(TPM + 1)
  toil_vst_own    TOIL expected counts through a VST fitted on the adversary's
                  own samples with the challenge's recipe
  *_aligned       the adversary maps each gene of its candidate pool onto the
                  release's distribution of that gene (needs only the
                  release); `vst_aligned` shows what the mapping does when
                  nothing is mismatched

Auxiliary data: BRCA has none.  COMBINED runs three ways: no auxiliary set;
the reference set as distributed (VST); and the reference set in the
candidates' representation, aligned the same way (all of the adversary's own
data from one pipeline).

Candidates a representation lacks (TOIL misses a few samples) keep their
distributed values and are left out of that condition's metrics.  MeLoMIA is
not run: its features are cached per target and its per-record calibration
would need every synth-shadow proxy re-applied to the new profiles.

    python scripts/perturb/candidate_mismatch.py --dataset COMBINED --jobs 6
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mia import datasets as D  # noqa: E402
from mia import metrics as M  # noqa: E402
from mia import paths  # noqa: E402
from mia import targets as T  # noqa: E402
from mia import views as V  # noqa: E402
from aux_mismatch import GENERATORS, NO_AUX, align_to_release, domias_baseline  # noqa: E402

TCGA = Path("~/data/TCGA_GDC/processed").expanduser()
REPS = ["vst", "gdc_tpm", "gdc_log2tpm", "toil_log2tpm", "toil_vst_own"]
ALIGN = {"vst", "gdc_log2tpm", "toil_log2tpm", "toil_vst_own"}   # TPM aligns as log-TPM does


def representation(dataset: str, rep: str) -> pd.DataFrame | None:
    """Every distributed sample of the cohort (candidates and reference) in `rep`."""
    genes = D.load_expression(dataset).columns
    if rep == "vst":
        ref = D.load_reference(dataset)
        return pd.concat([D.load_expression(dataset)] + ([] if ref is None else [ref]))
    src = {"gdc_tpm": "gdc_tpm", "gdc_log2tpm": "gdc_tpm"}.get(rep, rep)
    f = TCGA / f"{dataset}_all_{src}.tsv"
    if not f.exists():
        return None
    X = pd.read_csv(f, sep="\t", index_col=0).loc[:, list(genes)].astype(np.float64)
    return np.log2(X + 1) if rep == "gdc_log2tpm" else X


def conditions(dataset: str, Xs: np.ndarray) -> list:
    """[(name, candidates frame, reference frame or None or KEEP, ids to score)]."""
    cand_ids = D.load_expression(dataset).index
    ref_ids = None if D.load_reference(dataset) is None else D.load_reference(dataset).index
    out = []
    for rep in REPS:
        F = representation(dataset, rep)
        if F is None:
            continue
        for aligned in ([False, True] if rep in ALIGN else [False]):
            name = rep + ("_aligned" if aligned else "")
            present = cand_ids.intersection(F.index)
            C = F.loc[present]
            if aligned:
                C = align_to_release(C, Xs)
            C = pd.concat([C, D.load_expression(dataset).loc[cand_ids.difference(present)]])
            if ref_ids is None:
                out.append((f"{name}|none", C, None, present))
                continue
            R = F.loc[ref_ids.intersection(F.index)]
            if aligned:
                R = align_to_release(R, Xs)
            out.append((f"{name}|none", C, None, present))
            out.append((f"{name}|ref_vst", C, V.KEEP, present))
            if name != "vst":
                out.append((f"{name}|ref_same", C, R, present))
    return out


def one_target(args):
    dataset, generator, split, out_dir = args
    part = Path(out_dir) / f"{dataset}__{generator[:60]}__s{split}.csv"
    if not T.exists(dataset, generator, split):
        return f"{dataset}/{generator}/s{split} not built"
    done = pd.read_csv(part) if part.exists() else pd.DataFrame(columns=["condition"])
    t0 = time.time()
    Xs = T.load_target(dataset, generator, split)["X"].astype(np.float64)
    y = D.membership_labels(dataset, split)
    ids = D.load_expression(dataset).index
    rows = []
    for cond, C, ref, present in conditions(dataset, Xs):
        if cond in set(done.condition):
            continue
        aux = cond.split("|")[1]
        labels = (NO_AUX + (["RedSigma"] if dataset == "BRCA" else [])) if aux == "none" else V.USES_AUX
        keep = ids.isin(present)
        with V.view(dataset, reference=None if aux == "none" else ref, candidates=C):
            for label in labels:
                try:
                    if label == "DOMIAS-KDE":
                        s = domias_baseline(dataset, generator, split)
                    else:
                        s = V.build(label).score(dataset, generator, split)
                except Exception:
                    print(f"  FAILED {dataset}/{generator}/s{split} {label} {cond}\n"
                          + traceback.format_exc(limit=2), flush=True)
                    continue
                m = M.evaluate(y[keep], np.asarray(s)[keep])
                rows.append(dict(dataset=dataset, generator=generator, split=split,
                                 condition=cond, attack=label, n_scored=int(keep.sum()),
                                 auc=m["auc"], tpr_at_fpr_0_01=m["tpr_at_fpr_0.01"],
                                 tpr_at_fpr_0_1=m["tpr_at_fpr_0.1"]))
    pd.concat([done, pd.DataFrame(rows)]).to_csv(part, index=False)
    return f"{part.name}: +{len(rows)} rows in {time.time() - t0:.0f}s"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--splits", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--generators", nargs="+", default=GENERATORS)
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args()
    out_dir = paths.RESULTS / "perturb" / "candidate_mismatch_parts"
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(args.dataset, g, s, str(out_dir)) for g in args.generators for s in args.splits]
    pool = ProcessPoolExecutor(args.jobs)
    for msg in pool.map(one_target, jobs):
        print(msg, flush=True)
    parts = [pd.read_csv(f) for f in sorted(out_dir.glob("*.csv"))]
    out = paths.RESULTS / "perturb" / "candidate_mismatch.csv"
    pd.concat(parts).to_csv(out, index=False)
    print(f"wrote {out}", flush=True)
    # a worker that died after its last task once left the pool's shutdown
    # waiting forever; every result is on disk by now, so stop the workers
    # ourselves instead of waiting for them
    for proc in list(pool._processes.values()):
        proc.terminate()
    os._exit(0)


if __name__ == "__main__":
    main()
