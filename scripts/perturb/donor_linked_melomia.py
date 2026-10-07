"""Donor-linked membership inference with MeLoMIA (see donor_linked.py).

MeLoMIA's stack is built around the candidate pool, so a second sample B of a
donor cannot simply be appended to it.  Instead B is read through the stack
that already exists, exactly as a candidate is at attack time:

  1. every saved synth-shadow model scores B  -> B's loss signature under the
     attacker's own models, standardised with that model's candidate statistics;
  2. a proxy is trained on the released data (as the attack does) and scores B
     and the candidates;
  3. per-record calibration: B's proxy signature is z-scored against B's own
     signatures under the synth-shadows;
  4. the saved meta-classifiers score the result.

No model is retrained except the proxy, which the attack trains per target
anyway and does not keep.  The candidates go through the same code in the same
pass, and their AUC is printed next to the one recorded in results/index.csv.

Writes npz files in the layout of donor_linked.py to
artifacts/perturb/donor_linked_scores_melomia/.

    python scripts/perturb/donor_linked_melomia.py --dataset BRCA --backend nd --generators nd
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mia import attacks as A  # noqa: E402
from mia import datasets as D  # noqa: E402
from mia import metrics as M  # noqa: E402
from mia import paths  # noqa: E402
from mia import targets as T  # noqa: E402
from mia.attacks.melomia import features as F  # noqa: E402
from mia.attacks.melomia import meta as MM  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from donor_linked import extras  # noqa: E402

SETTINGS = {  # the stacks the grid reports (configs/experiments/melomia_prc_*.yaml)
    ("BRCA", "nd"): dict(n_shadows=30, n_noise=600),
    ("BRCA", "cvae"): dict(n_shadows=30, n_noise=50),
    ("COMBINED", "nd"): dict(n_shadows=20, n_noise=600),
    ("COMBINED", "cvae"): dict(n_shadows=20, n_noise=50),
}
EPS = 1e-6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--backend", required=True, choices=["nd", "cvae"])
    ap.add_argument("--generators", nargs="+", required=True)
    ap.add_argument("--splits", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    ds = args.dataset
    label = f"MeLoMIA-{args.backend.upper()}"

    atk = A.build(f"melomia_{args.backend}", device=args.device, optuna_trials=60,
                  classifiers=("xgb", "rf", "lgbm", "cat", "mlp"),
                  per_record_calibration=True, **SETTINGS[(ds, args.backend)])
    if not atk._meta_path(ds).exists():
        raise SystemExit(f"no trained stack at {atk._meta_path(ds)}; this script never builds one")
    meta = atk._ensure_meta(ds)
    be = atk._backend(ds)
    K = atk.n_shadows

    XB, mB = extras(ds)
    XB = XB.values.astype(np.float32)
    cand = list(D.load_expression(ds).index)
    X_real = D.load_expression(ds).values.astype(np.float32)
    out_dir = paths.ARTIFACTS / "perturb" / "donor_linked_scores_melomia"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. B and the candidates under every synth-shadow, per classifier spec
    specs = {c: meta["classifiers"][c] for c, w in meta["ensemble_weights"].items() if w >= 1e-6}
    zB = {c: [] for c in specs}          # standardised B signatures, one per shadow
    zC = {c: [] for c in specs}
    t0 = time.time()
    for k in range(1, K + 1):
        gen = atk._read_model(be, atk._synth_shadow_path(ds, k))
        lB, eB = be.extract(gen, XB)
        lC, eC, _, ids = atk._load_features(ds, k)
        assert list(ids) == cand
        for c, spec in specs.items():
            fC = F.prepare(lC, eC, spec["sweep_indices"], spec["noise_budget"]).astype(np.float64)
            fB = F.prepare(np.log(lB), eB, spec["sweep_indices"], spec["noise_budget"])
            mu, sd = fC.mean(0), fC.std(0)
            zB[c].append((fB - mu) / (sd + EPS))
            zC[c].append((fC - mu) / (sd + EPS))
        del gen
        atk._free_gpu()
    print(f"[{label}/{ds}] {K} synth-shadows read in {time.time() - t0:.0f}s", flush=True)
    ref = {}
    for c in specs:
        PB, PC = np.stack(zB[c]), np.stack(zC[c])
        ref[c] = (PB.mean(0), PB.std(0), PC.mean(0), PC.std(0))
        saved = np.load(atk._clf_dir(ds, c) / "record_reference.npz", allow_pickle=True)
        gap = float(np.abs(PC.mean(0) - saved["mean"]).max())
        print(f"  [{c}] candidates' per-record reference vs the stack's own: max gap {gap:.2e}",
              flush=True)
    del zB, zC

    # 2-4. one proxy per target
    for generator in args.generators:
        for split in args.splits:
            out = out_dir / f"{ds}__{generator[:60]}__s{split}__{args.backend}.npz"
            if out.exists() or not T.exists(ds, generator, split):
                continue
            t0 = time.time()
            tg = T.load_target(ds, generator, split)
            gen = be.probe()
            gen.seed = atk.seed + 7000 + split
            gen.fit(tg["X"], tg["y_int"], D.n_classes(ds))
            lC, eC = be.extract(gen, X_real)
            lB, eB = be.extract(gen, XB)
            del gen
            atk._free_gpu()
            sC, sB = np.zeros(len(X_real)), np.zeros(len(XB))
            for c, spec in specs.items():
                w = meta["ensemble_weights"][c]
                fC = F.prepare(np.log(lC), eC, spec["sweep_indices"], spec["noise_budget"]).astype(np.float64)
                fB = F.prepare(np.log(lB), eB, spec["sweep_indices"], spec["noise_budget"])
                mu, sd = fC.mean(0), fC.std(0)
                mB_, sB_, mC_, sC_ = ref[c]
                XC = (((fC - mu) / (sd + EPS) - mC_) / (sC_ + EPS)).astype(np.float32)
                XBf = (((fB - mu) / (sd + EPS) - mB_) / (sB_ + EPS)).astype(np.float32)
                clf = MM.get(c).load(atk._clf_dir(ds, c))
                sC += w * MM.get(c).predict(clf, XC)
                sB += w * MM.get(c).predict(clf, XBf)
            y = D.membership_labels(ds, split)
            payload = {"candidates": np.array(cand), "y_member": y, "attacks": np.array([label]),
                       "scores_alone": sC[None]}
            for i, (tier, m) in enumerate(mB.groupby("tier")):
                rows = mB.index.get_indexer(m.index)
                payload[f"tier{i}_name"] = tier
                payload[f"tier{i}_ids"] = np.array(m.index)
                payload[f"tier{i}_candidates"] = sC[None]
                payload[f"tier{i}_extras"] = sB[rows][None]
            np.savez_compressed(out, **payload)
            print(f"  {generator} split {split}: candidates AUC {M.evaluate(y, sC)['auc']:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
