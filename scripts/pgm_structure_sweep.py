#!/usr/bin/env python
"""DP-PGM structure x binning sweep: quality and MahalaMIA on the same targets.

    python scripts/pgm_structure_sweep.py --workers 12
    python scripts/pgm_structure_sweep.py --references      # MahalaMIA on mvn/cvae/nd
    python scripts/pgm_structure_sweep.py --dry-run         # list the jobs
    python scripts/pgm_structure_sweep.py --config configs/experiments/pgm_forest_sweep.yaml

`--config` points the same machinery at another design (its `name` is the
experiment and names the output CSV; `structure_grid` expands to one structure
per combination, labelled k<k>_l<l>).

Everything the sweep does is declared in configs/experiments/
pgm_structure_sweep.yaml; see its header for why.  For every (cohort, config,
epsilon, split) one target is built through the ordinary target store, under
its explicit variant name (e.g. `pgm@binning=dp_quantile,edge_estimator=
threshold,epsilon=1000,n_bins=16,structure=tree_label`), so any later
experiment can reuse it.  Then, in the same worker:

  * fidelity (mia.fidelity.evaluate -- the same call as the eps sweep and
    results/fidelity_grid.csv), the DP bound Phi(sqrt(rho)), the rho spent per
    stage, the tree selection's diagnostics and how far the (DP) bin edges
    are from the members' true percentiles -> results/pgm_structure_sweep.csv,
    one row per target, with every attack's AUC in `auc_*` columns;
  * every MahalaMIA variant in the YAML, shipped MAMA-MIA and the nine MAMA-MIA
    aggregation arms (black-box aux edges) -> ordinary runs in results/runs/ and
    results/index.csv, experiment=pgm_structure_sweep, variant=<config label>.

Idempotent: targets are cached by name, runs overwrite their own directory, CSV
rows are keyed on the target's fingerprint.  Re-run the same command to resume.
"""

import argparse
import os
import sys

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--threads", type=int, default=1)
_threads = str(_pre.parse_known_args()[0].threads)
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, _threads)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")       # CPU-only by design

import math  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from scipy.stats import norm  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from mia import attacks as A  # noqa: E402
from mia import csvlock  # noqa: E402
from mia import datasets as D  # noqa: E402
from mia import fidelity as F  # noqa: E402
from mia import metrics as M  # noqa: E402
from mia import runs as R  # noqa: E402
from mia import targets as T  # noqa: E402
from mamamia_aggregation import arms_for_split  # noqa: E402

_pre.add_argument("--config", default=str(ROOT / "configs" / "experiments"
                                          / "pgm_structure_sweep.yaml"))
CONFIG = Path(_pre.parse_known_args()[0].config)
EXPERIMENT = yaml.safe_load(CONFIG.read_text())["name"]
OUT = ROOT / "results" / f"{EXPERIMENT}.csv"
KEY = ["dataset", "target", "split", "fingerprint"]


def load_config() -> dict:
    return yaml.safe_load(CONFIG.read_text())


def configs(cfg: dict) -> dict:
    """label -> overrides, for every binning x structure plus the extras."""
    out = {}
    structures = dict(cfg.get("structures", {}))
    grid = cfg.get("structure_grid")
    if grid:
        for k in grid["k_label"]:
            for l in grid["l_pairs"]:
                structures[f"k{k}_l{l}"] = {**grid.get("base", {}),
                                            "k_label": k, "l_pairs": l}
    for b, bo in cfg["binnings"].items():
        for s, so in structures.items():
            out[f"{b}/{s}"] = {**bo, **so}
    for label, o in cfg.get("extra", {}).items():
        out[label] = dict(o)
    # The generator's own default; leaving it out keeps star targets' names the
    # same as the eps sweep's (so e.g. binning=uniform,n_bins=48 is shared).
    for o in out.values():
        if o.get("structure") == "hierarchical":
            del o["structure"]
    return out


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def run_mahalamia(cfg, ds, name, split, label, notes) -> dict:
    aucs = {}
    for v, params in cfg["mahalamia"].items():
        atk = A.build("mahalamia", **{**params, "device": "cpu", "verbose": False})
        met = atk.evaluate(ds, name, split, experiment=EXPERIMENT,
                           variant=f"{label}|{v}", notes=notes)
        aucs[f"mahalamia_{v}"] = met["auc"]
        aucs[f"t1_mahalamia_{v}"] = met.get("tpr_at_fpr_0.01", float("nan"))
    return aucs


def edge_diagnostics(ds, name, split) -> dict:
    """How far the released edges are from the members' own percentiles.

    White-box bookkeeping only (reads generator.pt); no attack uses it.
    """
    import pickle
    from mia.generators.pgm import _import_upstream
    _import_upstream()
    try:
        d = pickle.load(open(T._files(ds, name, split)["model"], "rb"))._discretizer
    except Exception:
        return {}
    X = D.load_expression(ds).values.astype(np.float64)
    m = D.membership_labels(ds, split).astype(bool)
    E = [np.asarray(e) for e in d._edges]
    lo = np.percentile(X[m], 0.5, axis=0)
    hi = np.percentile(X[m], 99.5, axis=0)
    Z = d.transform(X[m])
    K = d.n_bins
    modal = np.mean([np.bincount(Z[:, j], minlength=K).max() / len(Z)
                     for j in range(Z.shape[1])])
    occupied = np.mean([(np.bincount(Z[:, j], minlength=K) > 0).sum()
                        for j in range(Z.shape[1])])
    return {"edge_lo_err": float(np.median(np.abs([e[0] for e in E] - lo))),
            "edge_hi_err": float(np.median(np.abs([e[-1] for e in E] - hi))),
            "modal_cell_share": float(modal), "occupied_bins": float(occupied)}


def run_one(job: dict):
    ds, eps, split, label = job["dataset"], job["epsilon"], job["split"], job["label"]
    cfg = load_config()
    over = {**cfg["base"], **job["overrides"], "epsilon": eps}
    name = T.variant_name("pgm", ds, over)
    tag = f"{ds}/{label}/eps{eps:g}/s{split}"
    t0 = time.time()
    try:
        T.build_target(ds, name, split, params=over, device="cpu")
        t_fit = time.time() - t0
        tg = T.load_target(ds, name, split)
        rec = T.target_record(ds, name, split)
        rp = rec["params"]

        X = D.load_expression(ds).values.astype(np.float64)
        y = D.encode_subtypes(ds, D.load_subtypes(ds).values)
        m = D.membership_labels(ds, split).astype(bool)
        row = F.evaluate(tg["X"].astype(np.float64), tg["y_int"],
                         X[~m], y[~m], X[m], y[m])

        from mia.generators.pgm import _import_upstream
        _import_upstream()
        from pgm_fitter import rho_from_eps_delta
        row["dp_bound_auc"] = float(norm.cdf(math.sqrt(
            rho_from_eps_delta(eps, rp.get("delta", 1e-5)))))
        row["dp_valid"] = rp.get("binning", "quantile") != "quantile"
        row.update(edge_diagnostics(ds, name, split))
        try:
            import pickle
            g = pickle.load(open(T._files(ds, name, split)["model"], "rb"))
            row.update({f"rho_{k}": v for k, v in getattr(g, "rho_breakdown", {}).items()})
            row.update({f"sel_{k}": v for k, v in
                        getattr(g, "selection_diagnostics", {}).items()})
        except Exception:
            pass

        notes = f"{label} epsilon={eps}"
        aucs = run_mahalamia(cfg, ds, name, split, label, notes)

        spec = yaml.safe_load((ROOT / "configs" / "experiments" /
                               f"grid_{ds.lower()}.yaml").read_text())["attacks"]["mamamia"]
        atk = A.build(spec["class"], **{**spec.get("params", {}), "device": "cpu",
                                        "verbose": False})
        aucs["mamamia"] = atk.evaluate(ds, name, split, experiment=EXPERIMENT,
                                       variant=f"{label}|shipped", notes=notes)["auc"]

        n_bins = rp["n_bins"]
        _, arms = arms_for_split(ds, split, n_bins, eps, rp.get("delta", 1e-5),
                                 generator=name, edges="aux")
        ids = list(D.load_expression(ds).index)
        ym = D.membership_labels(ds, split).astype(int)
        for arm, sc in arms.items():
            met = M.evaluate(ym, sc)
            R.save_run(dataset=ds, attack="mamamia_arm", generator=name, split=split,
                       params={"attack": "mamamia_arm", "arm": arm, "n_bins": n_bins,
                               "epsilon": eps, "delta": rp.get("delta", 1e-5),
                               "attack_edges": "aux"},
                       sample_ids=ids, scores=sc, y_member=ym, metrics=met,
                       tag=slug(arm) + "_aux", experiment=EXPERIMENT,
                       variant=f"{label}|{arm}", notes=notes, target=rec)
            aucs[f"arm_{slug(arm)}"] = met["auc"]

        row.update(dataset=ds, config=label, epsilon=eps, split=split, target=name,
                   binning=rp.get("binning"), n_bins=n_bins,
                   structure=rp.get("structure", "hierarchical"),
                   k_label=rp.get("k_label"), l_pairs=rp.get("l_pairs"),
                   edge_estimator=rp.get("edge_estimator", "clip"),
                   fingerprint=rec["fingerprint"], seed=rec["seed"],
                   fit_seconds=round(t_fit, 1), seconds=round(time.time() - t0, 1),
                   **{f"auc_{k}": v for k, v in aucs.items()})
        return tag, row, None
    except Exception:
        return tag, None, traceback.format_exc(limit=6)


def run_reference(job):
    """MahalaMIA variants on a non-DP generator's canonical target."""
    ds, gen, split = job
    cfg = load_config()
    try:
        aucs = run_mahalamia(cfg, ds, gen, split, gen, "reference for pgm_structure_sweep")
        return f"{ds}/{gen}/s{split}", aucs, None
    except Exception:
        return f"{ds}/{gen}/s{split}", None, traceback.format_exc(limit=6)


def main():
    p = argparse.ArgumentParser(description=__doc__, parents=[_pre],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--datasets", nargs="+")
    p.add_argument("--eps", nargs="+", type=float)
    p.add_argument("--splits", nargs="+", type=int)
    p.add_argument("--only", nargs="+", help="config labels to run")
    p.add_argument("--references", action="store_true",
                   help="run the MahalaMIA variants on mvn/cvae/nd instead")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    cfg = load_config()
    datasets = args.datasets or cfg["datasets"]
    splits = args.splits or cfg["splits"]

    if args.references:
        jobs = [(d, g, s) for d in datasets for g in ("mvn", "cvae", "nd")
                for s in splits if T.exists(d, g, s)]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for tag, aucs, err in pool.map(run_reference, jobs):
                print(tag, "FAILED\n" + err if err else
                      " ".join(f"{k}={v:.3f}" for k, v in aucs.items()
                               if not k.startswith("t1_")), flush=True)
        print("REFERENCES COMPLETE", flush=True)
        return

    cfgs = configs(cfg)
    if args.only:
        cfgs = {k: v for k, v in cfgs.items() if k in args.only}
    jobs = [dict(dataset=d, epsilon=e, split=s, label=lbl, overrides=o)
            for e in (args.eps or cfg["epsilons"]) for s in splits
            for lbl, o in cfgs.items() for d in datasets]
    # eps=1000 split 1 of every config first, so a first look at the whole
    # design arrives early; COMBINED before BRCA within that since it is slower.
    first = cfg.get("first_epsilon", 1000)
    jobs.sort(key=lambda j: (j["split"], j["epsilon"] != first,
                             j["dataset"] != "COMBINED"))
    if args.dry_run:
        for j in jobs:
            print(j["dataset"], j["label"], j["epsilon"], j["split"],
                  T.variant_name("pgm", j["dataset"],
                                 {**cfg["base"], **j["overrides"], "epsilon": j["epsilon"]}))
        print(len(jobs), "jobs")
        return

    print(f"{len(jobs)} jobs on {args.workers} workers -> {OUT}", flush=True)
    t0, ok, failed = time.time(), 0, 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, j): j for j in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            tag, row, err = fut.result()
            if err is not None:
                failed += 1
                print(f"[{i}/{len(jobs)}] {tag}  FAILED\n{err}", flush=True)
                continue
            csvlock.append_row(OUT, row, key=KEY)
            ok += 1
            best = max((v, k) for k, v in row.items()
                       if k.startswith("auc_mahalamia_"))
            print(f"[{i}/{len(jobs)}] {tag}  ratio={row['utility_ratio']:.3f} "
                  f"W1={row['wasserstein_mean']:.3f} corr_mae={row['corr_mae']:.3f} "
                  f"disc={row['discriminator_auc']:.3f} "
                  f"mahala-best={best[0]:.3f} ({best[1][14:]}) "
                  f"mama={row['auc_mamamia']:.3f} "
                  f"(fit {row['fit_seconds']:.0f}s, total {row['seconds']:.0f}s)",
                  flush=True)
    print(f"\n{ok} ok, {failed} failed, {time.time() - t0:.0f}s wall", flush=True)
    print("STRUCTURE SWEEP COMPLETE", flush=True)


if __name__ == "__main__":
    main()
