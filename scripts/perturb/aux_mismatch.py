"""Attacks when the adversary's auxiliary data does not match the release
(Hakime's experiments 1 and 2).

The release and the candidates stay in the challenge's VST space; only the
auxiliary set changes.  One row per (dataset, generator, split, condition,
attack) in `results/perturb/aux_mismatch.csv`.

BRCA conditions (experiment 1; TCGA-BRCA ships no auxiliary set):
  none               no auxiliary set at all
  tcga_heldout84     84 non-member candidates of the split, class-stratified:
                     same cohort, same normalisation, same size as GSE58135.
                     The matched control.  These 84 are left out of every
                     BRCA metric, so all conditions score the same candidates.
  gse_vst_frozen     GSE58135 through the TCGA cohort's own fitted VST
  gse_vst_own        GSE58135 through a VST fitted on GSE58135 alone
  gse_log2tpm        GSE58135, log2(TPM + 1)
  gse_tpm            GSE58135, TPM as distributed
  *_aligned          the same set after the adversary maps each gene onto the
                     release's distribution of that gene (quantile mapping):
                     a repair that needs nothing but the release

COMBINED conditions (experiment 2; the challenge reference set, 824 samples):
  none, ref_vst (as distributed, the current setting), ref_vst_own (VST
  refitted on the 824 alone), ref_log2tpm, ref_tpm, *_aligned, and
  ref_vst_n84 (84 of the 824, to separate size from mismatch).

    python scripts/perturb/aux_mismatch.py --dataset BRCA --jobs 8
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

GSE = Path("~/data/GSE58135/processed").expanduser()
TCGA = Path("~/data/TCGA_GDC/processed").expanduser()
GENERATORS = ["mvn", "cvae", "nd", "pgg",
              "pgm@binning=dp_quantile,edge_estimator=threshold,n_bins=16",
              "tabsyn@class_freq=train,diff_schedule=steps,latent_scale=std,"
              "preprocess=clip:0.001:0.999+standard", "tabpfn"]
NO_AUX = ["MahalaMIA (no aux)", "MahalaMIA (no aux, ridge 1e-4)", "GAN-leaks", "MC",
          "MAMA-MIA v1", "MAMA-MIA v2"]
ALIGNABLE = ("gse_vst_frozen", "gse_vst_own", "gse_log2tpm", "gse_tpm", "ref_vst_own", "ref_log2tpm", "ref_tpm")


def read(path: Path, genes) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path, sep="\t", index_col=0).loc[:, list(genes)]


def align_to_release(ref: pd.DataFrame, Xs: np.ndarray) -> pd.DataFrame:
    """Per gene, send the auxiliary values to the release's quantiles at the same rank."""
    R = ref.values.astype(np.float64)
    ranks = (R.argsort(0).argsort(0) + 0.5) / len(R)
    out = np.empty_like(R)
    srt = np.sort(Xs, axis=0)
    grid = (np.arange(len(srt)) + 0.5) / len(srt)
    for j in range(R.shape[1]):
        out[:, j] = np.interp(ranks[:, j], grid, srt[:, j])
    return pd.DataFrame(out, index=ref.index, columns=ref.columns)


def domias_baseline(dataset, generator, split):
    """DOMIAS-KDE as the challenge baseline runs it: every set on its own scaler,
    PCA fitted on the auxiliary set (100 components, or a quarter of its size)."""
    from scipy import stats
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    std = lambda Z: StandardScaler().fit_transform(np.asarray(Z, dtype=np.float64))  # noqa: E731
    Xr = std(D.load_reference(dataset).values)
    Xt = std(D.load_expression(dataset).values)
    Xg = std(T.load_target(dataset, generator, split)["X"])
    pca = PCA(n_components=min(100, len(Xr) // 4), random_state=0).fit(Xr)
    t, g, r = pca.transform(Xt), pca.transform(Xg), pca.transform(Xr)
    return stats.gaussian_kde(g.T).logpdf(t.T) - stats.gaussian_kde(r.T).logpdf(t.T)


def heldout(dataset: str, split: int, n: int = 84) -> list:
    """`n` non-member candidates of the split, class-stratified, fixed per split."""
    y = D.membership_labels(dataset, split)
    ids = np.array(D.load_expression(dataset).index)
    lab = D.load_subtypes(dataset).values
    rng = np.random.default_rng(500 + split)
    non = np.where(y == 0)[0]
    take = []
    for c in np.unique(lab[non]):
        rows = non[lab[non] == c]
        take += list(rng.choice(rows, max(1, round(n * len(rows) / len(non))), replace=False))
    return list(ids[sorted(take)][:n])


def conditions(dataset: str, split: int) -> tuple:
    """({name: reference frame or None}, ids to leave out of the metrics)."""
    expr = D.load_expression(dataset)
    genes = expr.columns
    conds, skip = {"none": None}, []
    if dataset == "BRCA":
        skip = heldout(dataset, split)
        conds["tcga_heldout84"] = expr.loc[skip]
        for name in ("vst_frozen", "vst_own", "log2tpm", "tpm"):
            conds[f"gse_{name}"] = read(GSE / f"aux_{name}.tsv", genes)
    elif dataset == "COMBINED":
        ref = D.load_reference(dataset)
        conds["ref_vst"] = ref
        conds["ref_vst_n84"] = ref.sample(84, random_state=500 + split)
        for name in ("vst_own", "log2tpm", "tpm"):
            conds[f"ref_{name}"] = read(TCGA / f"COMBINED_reference_{name}.tsv", genes)
    else:
        raise SystemExit(f"no conditions defined for {dataset}")
    return {k: v for k, v in conds.items() if k == "none" or v is not None}, skip


def one_target(args):
    dataset, generator, split, out_dir = args
    part = Path(out_dir) / f"{dataset}__{generator[:60]}__s{split}.csv"
    if not T.exists(dataset, generator, split):
        return f"{dataset}/{generator}/s{split} not built"
    done = pd.read_csv(part) if part.exists() else pd.DataFrame(columns=["condition"])
    t0 = time.time()
    conds, skip = conditions(dataset, split)
    Xs = T.load_target(dataset, generator, split)["X"].astype(np.float64)
    for name in [c for c in conds if c in ALIGNABLE]:
        conds[name + "_aligned"] = align_to_release(conds[name], Xs)
    y = D.membership_labels(dataset, split)
    keep = ~D.load_expression(dataset).index.isin(skip)
    rows = []
    for cond, ref in conds.items():
        if cond in set(done.condition):
            continue
        labels = NO_AUX if cond == "none" else V.USES_AUX
        with V.view(dataset, reference=ref):
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
                m = M.evaluate(y[keep], s[keep])
                rows.append(dict(dataset=dataset, generator=generator, split=split,
                                 condition=cond, attack=label, n_aux=0 if ref is None else len(ref),
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
    out_dir = paths.RESULTS / "perturb" / "aux_mismatch_parts"
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(args.dataset, g, s, str(out_dir)) for g in args.generators for s in args.splits]
    with ProcessPoolExecutor(args.jobs) as pool:
        for msg in pool.map(one_target, jobs):
            print(msg, flush=True)
    parts = [pd.read_csv(f) for f in sorted(out_dir.glob("*.csv"))]
    out = paths.RESULTS / "perturb" / "aux_mismatch.csv"
    pd.concat(parts).to_csv(out, index=False)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
