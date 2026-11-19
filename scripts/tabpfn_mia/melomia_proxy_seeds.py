"""Is MeLoMIA's split-to-split spread the proxy's training randomness?

Retrains the proxy (a NoisyDiffusion model fitted to the release) with other
seeds and rescores with the *same* trained stack and meta-classifier, exactly
as `MeLoMIA.score` does, without touching the cache.  Source artifacts/pgg/env.sh.

    python scripts/tabpfn_mia/melomia_proxy_seeds.py --base_generator tabpfn --splits 1,2,4 --seeds 0,1,2
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from mia import datasets as D, metrics as M, targets as TG  # noqa: E402
from mia.attacks import build  # noqa: E402
from mia.attacks.melomia import features as F, meta as MM  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="BRCA_G200")
    ap.add_argument("--generator", default="tabpfn")
    ap.add_argument("--base_generator", default="tabpfn")
    ap.add_argument("--splits", default="1,2,4")
    ap.add_argument("--seeds", default="0,1,2", help="offsets added to the default proxy seed")
    a = ap.parse_args()
    atk = build("melomia_nd", n_shadows=30, n_noise=600, optuna_trials=60,
                classifiers=["xgb", "rf", "lgbm", "cat", "mlp"], base_generator=a.base_generator or None)
    ds = a.dataset
    meta = atk.prepare(ds)                       # cached: loads, builds nothing
    be = atk._backend(ds)
    X_real = D.load_expression(ds).values.astype(np.float32)
    for split in [int(s) for s in a.splits.split(",")]:
        target = TG.load_target(ds, a.generator, split)
        y = D.membership_labels(ds, split)
        for off in [int(s) for s in a.seeds.split(",")]:
            gen = be.probe()
            gen.seed = atk.seed + 7000 + split + 100_000 * off
            gen.fit(target["X"], target["y_int"], D.n_classes(ds))
            losses, extra = be.extract(gen, X_real, atk._labels(ds))
            sharp = losses.mean(2).mean(0)[-1]           # raw loss at the lowest noise level
            losses = atk._calibrate_grid(losses, None)
            total = np.zeros(len(losses))
            for name, w in meta["ensemble_weights"].items():
                if w < 1e-6:
                    continue
                spec = meta["classifiers"][name]
                X = F.prepare(losses, extra, spec["sweep_indices"], spec["noise_budget"])
                if atk.per_record_calibration:
                    ref = np.load(atk._clf_dir(ds, name) / "record_reference.npz", allow_pickle=True)
                    X = F.per_record_apply(X, ref["mean"], ref["sd"])
                clf = MM.get(name).load(atk._clf_dir(ds, name))
                total += MM.get(name).predict(clf, X) * w
            m = M.evaluate(y, total)
            print(f"split {split} proxy seed +{off}: AUC {m['auc']:.3f}  TPR@1% {m['tpr_at_fpr_0.01']:.3f}  "
                  f"(lowest-noise loss {sharp:.3f})", flush=True)
            del gen
            atk._free_gpu()


if __name__ == "__main__":
    main()
