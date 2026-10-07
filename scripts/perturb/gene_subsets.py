"""Adversary restricted to a subset of genes, and the reduction-dimension
ablation of the challenge baselines (Hakime's experiment 3).

Two blocks, one CSV (`results/perturb/gene_subsets.csv`), one row per
(dataset, generator, split, block, attack, rule, k):

  baseline_dr   DOMIAS-KDE, GAN-leaks calibrated and LOGAN-D1 exactly as the
                challenge baseline runs them (each set on its own
                StandardScaler; PCA fitted on the reference set; vDE / sDE /
                dDE gene selections), with the reduction dimension swept
                instead of fixed at 100.
  gene_subset   our attacks when the adversary holds only k genes of every
                candidate.  The release and the auxiliary set are cut to the
                same k genes; the target generator still trained on all 978.

Gene-selection rules, all computable by the adversary from the release, its
labels and the auxiliary set:

  vde      variance(release) / variance(aux), on the per-set standardised
           arrays, as the baseline does.  After per-set standardisation every
           gene has variance 1 in both, so this ranking is numerical noise;
           kept because it is what the baseline runs.
  vde_raw  the same ratio on the unstandardised values
  sde      mutual information between gene and class in the release
  dde      |coefficient| of an L1 logistic regression, release vs aux
  hvg      variance in the release
  random   a fixed random order per split (nested across k)

    python scripts/perturb/gene_subsets.py --dataset COMBINED --jobs 8
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mia import datasets as D  # noqa: E402
from mia import metrics as M  # noqa: E402
from mia import paths  # noqa: E402
from mia import targets as T  # noqa: E402
from mia import views as V  # noqa: E402

KS = (10, 25, 50, 100, 200, 400, 600, 800)
GENERATORS = ["mvn", "cvae", "nd", "pgg",
              "pgm@binning=dp_quantile,edge_estimator=threshold,n_bins=16",
              "tabsyn@class_freq=train,diff_schedule=steps,latent_scale=std,"
              "preprocess=clip:0.001:0.999+standard", "tabpfn"]
SUBSET_ATTACKS = ["MahalaMIA (as submitted)", "MahalaMIA (ridge 1e-4)", "MAMA-MIA v1",
                  "MAMA-MIA v2", "RedSigma", "GAN-leaks", "GAN-leaks cal."]


def std(Z):
    from sklearn.preprocessing import StandardScaler
    return StandardScaler().fit_transform(np.asarray(Z, dtype=np.float64))


def rankings(Xg, yg, Xr, seed: int) -> dict:
    """Gene orderings, most preferred first, for every rule that can be computed."""
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.linear_model import LogisticRegression
    out = {}
    Sg = std(Xg)
    out["sde"] = np.argsort(mutual_info_classif(Sg, yg, random_state=42), kind="stable")[::-1]
    out["hvg"] = np.argsort(Xg.var(0), kind="stable")[::-1]
    out["random"] = np.random.default_rng(seed).permutation(Xg.shape[1])
    if Xr is not None:
        Sr = std(Xr)
        out["vde"] = np.argsort(Sg.var(0) / (Sr.var(0) + 1e-10), kind="stable")[::-1]
        out["vde_raw"] = np.argsort(Xg.var(0) / (Xr.var(0) + 1e-10), kind="stable")[::-1]
        lr = LogisticRegression(C=0.1, penalty="l1", solver="liblinear", random_state=42)
        lr.fit(np.vstack([Sg, Sr]), np.r_[np.ones(len(Sg)), np.zeros(len(Sr))])
        out["dde"] = np.argsort(np.abs(lr.coef_[0]), kind="stable")[::-1]
    return out


def kde_log_ratio(t, g, r):
    """DOMIAS with Gaussian KDEs, as a log ratio so high dimensions do not underflow."""
    from scipy import stats
    return stats.gaussian_kde(g.T).logpdf(t.T) - stats.gaussian_kde(r.T).logpdf(t.T)


def baseline_dr(Xt, Xg, Xr, yg, ranks, ks, logan: bool):
    from sklearn.decomposition import PCA

    from mia.attacks.generic import gan_leaks_cal
    St, Sg, Sr = std(Xt), std(Xg), std(Xr)
    for k in ks:
        views = {}
        if k <= min(Sr.shape) - 1:
            pca = PCA(n_components=k, random_state=0).fit(Sr)
            views["pca"] = (pca.transform(St), pca.transform(Sg), pca.transform(Sr))
        for rule in ("vde", "sde", "dde"):
            idx = ranks[rule][:k]
            views[rule] = (St[:, idx], Sg[:, idx], Sr[:, idx])
        for rule, (t, g, r) in views.items():
            yield "GAN-leaks cal.", rule, k, gan_leaks_cal(t, g, r)
            try:
                yield "DOMIAS-KDE", rule, k, kde_log_ratio(t, g, r)
            except Exception as exc:               # singular KDE covariance
                print(f"    KDE failed at {rule} k={k}: {exc}", flush=True)
            if logan:
                import torch
                from domias.baselines import LOGAN_D1
                np.random.seed(42)
                torch.manual_seed(42)
                yield "LOGAN-D1", rule, k, np.asarray(LOGAN_D1(t, g, r), dtype=float)
    yield "GAN-leaks cal.", "all", St.shape[1], gan_leaks_cal(St, Sg, Sr)


def one_target(args):
    dataset, generator, split, out_dir, logan, reference_tsv = args
    part = Path(out_dir) / f"{dataset}__{generator.replace('/', '_')[:60]}__s{split}.csv"
    if part.exists():
        return f"{part.name} cached"
    if not T.exists(dataset, generator, split):
        return f"{dataset}/{generator}/s{split} not built"
    t0 = time.time()
    y = D.membership_labels(dataset, split)
    tg = T.load_target(dataset, generator, split)
    Xg, yg = tg["X"].astype(np.float64), np.asarray(tg["y_int"])
    Xt = D.load_expression(dataset).values.astype(np.float64)
    ref = D.load_reference(dataset)
    if reference_tsv:
        ref = pd.read_csv(reference_tsv, sep="\t", index_col=0)
        ref = ref.loc[:, list(D.load_expression(dataset).columns)]
    Xr = None if ref is None else ref.values.astype(np.float64)
    ranks = rankings(Xg, yg, Xr, seed=1000 + split)
    rows = []

    def add(block, attack, rule, k, scores):
        m = M.evaluate(y, scores)
        rows.append(dict(dataset=dataset, generator=generator, split=split, block=block,
                         attack=attack, rule=rule, k=k, auc=m["auc"],
                         tpr_at_fpr_0_01=m["tpr_at_fpr_0.01"],
                         tpr_at_fpr_0_1=m["tpr_at_fpr_0.1"]))

    if Xr is not None:
        for attack, rule, k, s in baseline_dr(Xt, Xg, Xr, yg, ranks, KS, logan):
            add("baseline_dr", attack, rule, k, s)

    cells = [("all", Xt.shape[1], None)]
    cells += [(rule, k, ranks[rule][:k]) for rule in ranks if rule != "vde" for k in KS]
    for rule, k, idx in cells:
        with V.view(dataset, reference=ref if reference_tsv else V.KEEP, genes=idx):
            for label in SUBSET_ATTACKS:
                if label in V.USES_AUX and Xr is None and label != "RedSigma" \
                        and "MahalaMIA" not in label:
                    continue
                try:
                    add("gene_subset", label, rule, k, V.build(label).score(dataset, generator, split))
                except Exception:
                    print(f"  FAILED {dataset}/{generator}/s{split} {label} {rule} k={k}\n"
                          + traceback.format_exc(limit=2), flush=True)
    pd.DataFrame(rows).to_csv(part, index=False)
    return f"{part.name}: {len(rows)} rows in {time.time() - t0:.0f}s"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--splits", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--generators", nargs="+", default=GENERATORS)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--logan", action="store_true")
    ap.add_argument("--reference", default=None,
                    help="samples x genes TSV to use as the auxiliary set instead of the cohort's")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    out_dir = paths.RESULTS / "perturb" / ("gene_subsets_parts" + args.tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(args.dataset, g, s, str(out_dir), args.logan, args.reference)
            for g in args.generators for s in args.splits]
    with ProcessPoolExecutor(args.jobs) as pool:
        for msg in pool.map(one_target, jobs):
            print(msg, flush=True)
    parts = [pd.read_csv(f) for f in sorted(out_dir.glob("*.csv"))]
    if parts:
        out = paths.RESULTS / "perturb" / f"gene_subsets{args.tag}.csv"
        pd.concat(parts).to_csv(out, index=False)
        print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
