#!/usr/bin/env python
"""MAMA-MIA v2 on the DP-PGM structure-sweep targets: every access path, every arm.

    python scripts/mamamia_v2_eval.py --workers 2                  # all finished targets
    python scripts/mamamia_v2_eval.py --datasets BRCA --only dp_quantile16/tree_label --eps 1000

For each target in results/pgm_structure_sweep.csv (which lists only targets
the sweep has finished building), runs `mamamia_v2.family_scores` for every
(cliques, edges) access path and derives the aggregation arms from the
per-family scores, so each path costs one pass.  Arms:

    all            clique-weighted mean of every family's mean log-ratio
    <family>       one family alone (1way, gl, tree)
    star           1way + gl (v1's tables on v2's bins)
    ... each also class-centred ("_cc").

Access paths (see mia/attacks/mamamia_v2.py):
    black-box   cliques=public|recovered|aux  x  edges=recovered|aux
    white-box   cliques=true, or edges=known on a data-dependent binning
                -- DIAGNOSTICS, labelled `access=white-box` in every row.

Also recorded, as white-box bookkeeping (no attack reads them): how many of the
target's tree pairs the black-box recovery found, and how far the recovered
edges are from the target's.

Writes results/mamamia_v2.csv (one row per target x path x arm, keyed) and saves
the headline arm of each path ("all_cc") as an ordinary run in results/runs
(experiment=mamamia_v2, variant=<config>|<path>).
"""

import argparse
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import time  # noqa: E402
import traceback  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mia import csvlock  # noqa: E402
from mia import datasets as D  # noqa: E402
from mia import metrics as M  # noqa: E402
from mia import runs as R  # noqa: E402
from mia import targets as T  # noqa: E402
from mia.attacks.mamamia_v2 import (MAMAMIAv2, center_by_class,  # noqa: E402
                                    recover_edges, true_cliques)
from mia.attacks.mahalamia import sigmoid_calibrate  # noqa: E402

SWEEP = ROOT / "results" / "pgm_structure_sweep.csv"
OUT = ROOT / "results" / "mamamia_v2.csv"
EXPERIMENT = "mamamia_v2"
KEY = ["dataset", "target", "split", "fingerprint", "cliques", "edges", "arm"]
PATHS = [("public", "aux"), ("public", "recovered"), ("recovered", "aux"),
         ("recovered", "recovered"), ("aux", "aux"), ("aux", "recovered"),
         ("shadow", "recovered"),
         ("public", "known"), ("true", "known"), ("true", "recovered")]


def arms(fs: dict) -> dict:
    s, n, y = fs["scores"], fs["n_tables"], fs["y"]

    def mix(fams):
        fams = [f for f in fams if f in s]
        if not fams:
            return None
        w = np.array([n[f] for f in fams], float)
        return sum(wi * s[f] for wi, f in zip(w, fams)) / w.sum()

    out = {"all": mix(list(s)), "star": mix(["1way", "gl"])}
    out.update({f: s[f] for f in s})
    out = {k: v for k, v in out.items() if v is not None}
    out.update({f"{k}_cc": center_by_class(v, y) for k, v in list(out.items())})
    return out


def run_one(job):
    ds, name, split, label, eps, paths = job
    t0 = time.time()
    try:
        rec = T.target_record(ds, name, split)
        params = rec["params"]
        ym = D.membership_labels(ds, split).astype(int)
        ids = list(D.load_expression(ds).index)
        Xs = T.load_target(ds, name, split)["X"].astype(np.float64)
        Xr = D.load_expression(ds).values.astype(np.float64)
        structure = params.get("structure", "hierarchical")

        # White-box bookkeeping, computed once.
        tc = true_cliques(ds, name, split) if structure != "hierarchical" else None
        truth = set(tc["pairs"]) if tc else set()
        truth_gl = set(tc["gl"]) if tc else set()
        E_true = (recover_edges(Xs, params, "known", Xr, ds, name, split)
                  if params.get("binning") != "uniform" else None)

        rows = []
        for cl, ed in PATHS:
            if paths and f"{cl}/{ed}" not in paths:
                continue
            if cl == "true" and tc is None:
                continue
            if cl == "shadow" and structure == "hierarchical":
                continue        # the star is fixed by config: nothing to guess
            atk = MAMAMIAv2(cliques=cl, edges=ed)
            fs = atk.family_scores(ds, name, split)
            extra = {"n_tree": len(fs["tree"]), "aux_set": fs["aux"]}
            c = fs["cliques"]
            if cl in ("recovered", "aux", "shadow") and tc is not None:
                w = c.get("weights", {})
                chosen = (set(p for p, wi in zip(c["pairs"], w["pairs"]) if wi >= 0.5)
                          if "pairs" in w else set(c["pairs"]))
                if truth:
                    extra["tree_recall"] = len(chosen & truth) / len(truth)
                gl = (set(g for g, wi in zip(c["gl"], w["gl"]) if wi >= 0.5)
                      if "gl" in w else set(c["gl"]))
                if truth_gl and len(truth_gl) < len(D.gene_names(ds)):
                    extra["gl_recall"] = len(gl & truth_gl) / len(truth_gl)
            if E_true is not None:
                extra["edge_mae_vs_target"] = float(np.abs(fs["edges"] - E_true).mean())
            for arm, sc in arms(fs).items():
                met = M.evaluate(ym, sc)
                rows.append({"dataset": ds, "config": label, "epsilon": eps,
                             "split": split, "target": name,
                             "fingerprint": rec["fingerprint"],
                             "binning": params.get("binning"),
                             "n_bins": params.get("n_bins"), "structure": structure,
                             "cliques": cl, "edges": ed, "access": fs["access"],
                             "arm": arm, "auc": met["auc"],
                             "tpr_at_fpr_0.01": met.get("tpr_at_fpr_0.01"),
                             **extra})
                if arm == "all_cc":
                    R.save_run(dataset=ds, attack="mamamia_v2", generator=name,
                               split=split, params={**atk.params(), "arm": arm},
                               sample_ids=ids, scores=sigmoid_calibrate(sc, log_transform=False),
                               y_member=ym, metrics=met, tag=atk.tag(),
                               experiment=EXPERIMENT,
                               variant=f"{label}|{cl}/{ed}",
                               notes=f"{fs['access']} {label} epsilon={eps}",
                               target=rec)
        return f"{ds}/{label}/eps{eps:g}/s{split}", rows, time.time() - t0, None
    except Exception:
        return f"{ds}/{label}/eps{eps:g}/s{split}", None, 0, traceback.format_exc(limit=8)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--datasets", nargs="+")
    p.add_argument("--only", nargs="+", help="config labels")
    p.add_argument("--eps", nargs="+", type=float)
    p.add_argument("--splits", nargs="+", type=int)
    p.add_argument("--redo", action="store_true", help="re-run targets already in OUT")
    p.add_argument("--sweep", default=str(SWEEP), help="sweep CSV listing the targets")
    p.add_argument("--paths", nargs="+",
                   help="only these cliques/edges paths, e.g. shadow/recovered; "
                        "targets are skipped only if they already have them")
    args = p.parse_args()

    sw = pd.read_csv(args.sweep)
    if args.datasets:
        sw = sw[sw.dataset.isin(args.datasets)]
    if args.only:
        sw = sw[sw.config.isin(args.only)]
    if args.eps:
        sw = sw[sw.epsilon.isin(args.eps)]
    if args.splits:
        sw = sw[sw.split.isin(args.splits)]
    if OUT.exists() and not args.redo:
        o = pd.read_csv(OUT)
        if args.paths:
            o = o[(o.cliques + "/" + o.edges).isin(args.paths)]
        done = set(o[["target", "split", "fingerprint"]]
                   .itertuples(index=False, name=None))
        sw = sw[[(t, s, f) not in done for t, s, f in
                 zip(sw.target, sw.split, sw.fingerprint)]]
    jobs = [(r.dataset, r.target, int(r.split), r.config, r.epsilon,
             tuple(args.paths or ())) for r in sw.itertuples()]
    print(f"{len(jobs)} targets on {args.workers} workers -> {OUT}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(run_one, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            tag, rows, secs, err = fut.result()
            if err:
                print(f"[{i}/{len(jobs)}] {tag} FAILED\n{err}", flush=True)
                continue
            for r in rows:
                csvlock.append_row(OUT, r, key=KEY)
            best = {}
            for r in rows:
                if r["arm"] == "all_cc":
                    best[f"{r['cliques'][:3]}/{r['edges'][:3]}"] = r["auc"]
            print(f"[{i}/{len(jobs)}] {tag} ({secs:.0f}s) all_cc: "
                  + " ".join(f"{k}={v:.3f}" for k, v in best.items()), flush=True)
    print("MAMAMIA V2 COMPLETE", flush=True)


if __name__ == "__main__":
    main()
