#!/usr/bin/env python
"""DP-PGM privacy curve: epsilon sweep, utility and attacks on the same targets.

    python scripts/pgm_eps_sweep.py --workers 10
    python scripts/pgm_eps_sweep.py --datasets BRCA --eps 0.1 1 --splits 1

For every (cohort, epsilon, split) this builds one target through the ordinary
target store, so it is reusable by any later experiment under the name
`pgm@epsilon=<eps>` (epsilon=10 is the canonical `pgm` target and is reused, not
rebuilt).  Then, in the same worker:

  * utility and fidelity  -> results/pgm_eps_sweep.csv, one row per target,
    with sigma per marginal family and the analytical AUC ceiling for a
    marginals-based attack (FINDINGS 9a);
  * MahalaMIA and MAMA-MIA with the grid's parameters -> results/runs/ and
    results/index.csv, like any other experiment (experiment=pgm_eps_sweep);
  * the nine MAMA-MIA aggregation arms of scripts/mamamia_aggregation.py, also
    as ordinary runs (attack=mamamia_arm, variant=<arm>), so their row-level
    scores are kept.

The ordinary attacks are skipped for the canonical epsilon=10 target: those runs
already exist under the grid's configuration and share its run ids.

Every step is idempotent -- targets are cached by name and checked by content,
runs overwrite their own directory, CSV rows are keyed -- so an interrupted
sweep is resumed by re-running the same command.

The PGM fit is single-threaded (`mbi.FactoredInference`), so parallelism is
across targets.  Size `--workers` to the cores you are allowed; each worker
also gets `--threads` BLAS threads for the utility metrics.
"""

import argparse
import os
import sys

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--threads", type=int, default=2)
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
from mamamia_aggregation import arms_for_split, family_sigma  # noqa: E402

EXPERIMENT = "pgm_eps_sweep"
OUT = ROOT / "results" / "pgm_eps_sweep.csv"
KEY = ["dataset", "epsilon", "split", "fingerprint"]
GRID_CONFIG = {"BRCA": "grid_brca.yaml", "COMBINED": "grid_combined.yaml"}
DEFAULT_EPS = [0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]


def grid_config(dataset: str) -> dict:
    return yaml.safe_load((ROOT / "configs" / "experiments"
                           / GRID_CONFIG[dataset]).read_text())


def ceiling(dataset: str, eps: float, delta: float, n_bins: int, n_syn: int) -> dict:
    """Best AUC any attack on the released marginals can reach (FINDINGS 9a).

    A member adds +1 to one cell of each of the 978 one-way and 978 gene x label
    marginals.  Against that, each cell carries DP noise sigma and sampling noise
    n_syn * p * (1 - p); the per-cell squared SNRs add, and a Gaussian
    likelihood-ratio test with separation d reaches AUC Phi(d / sqrt(2)).
    """
    n_genes = len(D.gene_names(dataset))
    y = D.encode_subtypes(dataset, D.load_subtypes(dataset).values)
    pc = np.bincount(y, minlength=D.n_classes(dataset)) / len(y)
    p1, p2 = 1.0 / n_bins, (pc / n_bins).mean()
    s1 = family_sigma(n_genes, 0, eps, delta)
    s2 = family_sigma(n_genes, 1, eps, delta)
    d2 = (n_genes / (s1 ** 2 + n_syn * p1 * (1 - p1))
          + n_genes / (s2 ** 2 + n_syn * p2 * (1 - p2)))
    return {"sigma_1way": s1, "sigma_2way": s2,
            "ceiling_auc": float(norm.cdf(math.sqrt(d2) / math.sqrt(2)))}


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def run_one(job: dict):
    """Build, measure and attack one target.  Returns (label, row, error)."""
    ds, eps, split = job["dataset"], job["epsilon"], job["split"]
    name = T.variant_name("pgm", ds, {"epsilon": eps, **job["overrides"]})
    label = f"{ds}/{name}/s{split}"
    t0 = time.time()
    try:
        cfg = grid_config(ds)
        T.build_target(ds, name, split, params=cfg["target_params"]["pgm"],
                       device="cpu")
        t_fit = time.time() - t0
        tg = T.load_target(ds, name, split)
        rec = T.target_record(ds, name, split)
        rp = rec["params"]

        # ── utility / fidelity ────────────────────────────────────────────
        X = D.load_expression(ds).values.astype(np.float64)
        y = D.encode_subtypes(ds, D.load_subtypes(ds).values)
        m = D.membership_labels(ds, split).astype(bool)
        row = F.evaluate(tg["X"].astype(np.float64), tg["y_int"],
                         X[~m], y[~m], X[m], y[m])
        row.update(ceiling(ds, eps, rp.get("delta", 1e-5), rp["n_bins"],
                           len(tg["X"])))
        # What epsilon itself guarantees for ANY attack: a composition of
        # Gaussian mechanisms at total rho is mu-GDP with mu = sqrt(2 rho), and
        # no test beats AUC Phi(mu / sqrt 2).  Only binding if the whole
        # pipeline is inside the accounting (binning != "quantile").
        from mia.generators.pgm import _import_upstream
        _import_upstream()
        from pgm_fitter import rho_from_eps_delta
        row["dp_bound_auc"] = float(norm.cdf(math.sqrt(
            rho_from_eps_delta(eps, rp.get("delta", 1e-5)))))

        # ── attacks ───────────────────────────────────────────────────────
        aucs = {}
        if name != "pgm":
            for label_ in ("mahalamia", "mamamia"):
                spec = cfg["attacks"][label_]
                atk = A.build(spec["class"], **{**spec.get("params", {}),
                                                "device": "cpu", "verbose": False})
                met = atk.evaluate(ds, name, split, experiment=EXPERIMENT,
                                   variant=label_, notes=f"epsilon={eps}")
                aucs[label_] = met["auc"]

        n_bins = rp["n_bins"]
        yr, arms = arms_for_split(ds, split, n_bins, eps, rp.get("delta", 1e-5),
                                  generator=name)
        ids = list(D.load_expression(ds).index)
        ym = D.membership_labels(ds, split).astype(int)
        for arm, sc in arms.items():
            met = M.evaluate(ym, sc)
            R.save_run(dataset=ds, attack="mamamia_arm", generator=name, split=split,
                       params={"attack": "mamamia_arm", "arm": arm, "n_bins": n_bins,
                               "epsilon": eps, "delta": rp.get("delta", 1e-5)},
                       sample_ids=ids, scores=sc, y_member=ym, metrics=met,
                       tag=slug(arm), experiment=EXPERIMENT, variant=arm,
                       notes=f"epsilon={eps}", target=rec)
            aucs[arm] = met["auc"]

        row.update(dataset=ds, epsilon=eps, split=split, target=name,
                   binning=rp.get("binning", "quantile"), n_bins=n_bins,
                   fingerprint=rec["fingerprint"], seed=rec["seed"],
                   composition=rp.get("composition"),
                   neighboring=rp.get("neighboring"),
                   fit_seconds=round(t_fit, 1),
                   seconds=round(time.time() - t0, 1),
                   **{f"auc_{slug(k)}": v for k, v in aucs.items()})
        return label, row, None
    except Exception:
        return label, None, traceback.format_exc(limit=4)


def main():
    p = argparse.ArgumentParser(description=__doc__, parents=[_pre],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", default=["COMBINED", "BRCA"])
    p.add_argument("--eps", nargs="+", type=float, default=DEFAULT_EPS)
    p.add_argument("--splits", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--variants", nargs="+", default=[""],
                   help='extra target overrides, one set per argument, e.g. '
                        '"binning=uniform" "binning=uniform,n_bins=16"; '
                        '"" is the canonical generator')
    args = p.parse_args()
    variants = [T.split_name("pgm@" + v)[1] for v in args.variants]

    # Uncached fits first so the cheap cached jobs fill the tail.
    jobs = [dict(dataset=d, epsilon=e, split=s, overrides=o)
            for o in variants for d in args.datasets
            for e in args.eps for s in args.splits]
    jobs.sort(key=lambda j: T.exists(
        j["dataset"],
        T.variant_name("pgm", j["dataset"], {"epsilon": j["epsilon"], **j["overrides"]}),
        j["split"]))

    print(f"{len(jobs)} jobs on {args.workers} workers "
          f"({os.environ['OMP_NUM_THREADS']} BLAS threads each) -> {OUT}",
          flush=True)
    t0, ok, failed = time.time(), 0, 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, j): j for j in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            label, row, err = fut.result()
            if err is not None:
                failed += 1
                print(f"[{i}/{len(jobs)}] {label}  FAILED\n{err}", flush=True)
                continue
            csvlock.append_row(OUT, row, key=KEY)
            ok += 1
            print(f"[{i}/{len(jobs)}] {label}  "
                  f"ratio={row['utility_ratio']:.3f} W1={row['wasserstein_mean']:.3f} "
                  f"ceiling={row['ceiling_auc']:.3f} "
                  f"mama={row.get('auc_mamamia', float('nan')):.3f} "
                  f"best-arm={row.get('auc_log_class_centred', float('nan')):.3f} "
                  f"mahala={row.get('auc_mahalamia', float('nan')):.3f} "
                  f"(fit {row['fit_seconds']:.0f}s, total {row['seconds']:.0f}s)",
                  flush=True)
    print(f"\n{ok} ok, {failed} failed, {time.time() - t0:.0f}s wall", flush=True)
    print("SWEEP COMPLETE", flush=True)


if __name__ == "__main__":
    main()
