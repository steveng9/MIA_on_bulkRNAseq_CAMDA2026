"""MeLoMIA-ND on the Pro-GENE-GEN-MPC TabPFN releases, for a head-to-head with ICL-MIA.

Settings are those of the repository's earlier MeLoMIA-ND runs on TabPFN
(`configs/experiments/grid_sota_brca.yaml`: K = 30, 600 noise draws, 60 Optuna
trials, xgb / rf / lgbm / cat / mlp, per-record calibration, synth-shadows).
Needs the link tree from `pgg_shim.py`; source its env.sh first.  Runs in the
main environment (camda-mia).

    source artifacts/pgg/env.sh
    python scripts/tabpfn_mia/run_melomia_nd.py [--shadows a:b] [--splits 1,2,3,4,5]

`--shadows a:b` builds only that share of the stack (no meta-classifier, no
scoring), so several processes can build it in parallel.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from mia.attacks import build  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="BRCA")
    ap.add_argument("--generator", default="tabpfn")
    ap.add_argument("--splits", default="1,2,3,4,5")
    ap.add_argument("--shadows", default="", help="a:b, 1-based inclusive range of shadows to build only")
    ap.add_argument("--n_shadows", type=int, default=30)
    ap.add_argument("--optuna_trials", type=int, default=60)
    ap.add_argument("--classifiers", default="xgb,rf,lgbm,cat,mlp")
    ap.add_argument("--experiment", default="pgg_tabpfn35_melomia_nd")
    ap.add_argument("--base_generator", default="", help="'' = NoisyDiffusion base shadows (default); "
                    "'tabpfn' = the TabPFN shadows injected by melomia_tabpfn_shadows.py")
    a = ap.parse_args()

    atk = build("melomia_nd", n_shadows=a.n_shadows, n_noise=600, optuna_trials=a.optuna_trials,
                classifiers=a.classifiers.split(","), base_generator=a.base_generator or None)
    t0 = time.time()
    if a.shadows:
        lo, hi = (int(v) for v in a.shadows.split(":"))
        atk.prepare(a.dataset, shadows=range(lo, hi + 1), meta=False)
        print(f"built shadows {lo}..{hi} in {time.time()-t0:.0f}s", flush=True)
        return
    atk.prepare(a.dataset)
    print(f"stack + meta-classifier ready in {time.time()-t0:.0f}s", flush=True)
    for s in [int(v) for v in a.splits.split(",")]:
        m = atk.evaluate(a.dataset, a.generator, s, experiment=a.experiment, variant="melomia_nd" + (f"_base{a.base_generator}" if a.base_generator else ""),
                         notes="MeLoMIA-ND on the Pro-GENE-GEN-MPC tabpfn_35 releases (ICL-MIA head-to-head)")
        print(f"split {s}: AUC {m['auc']:.3f}  TPR@1%FPR {m['tpr_at_fpr_0.01']:.3f}  "
              f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
