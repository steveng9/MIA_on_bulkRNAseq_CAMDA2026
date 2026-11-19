"""TabPFN shadows + TabPFN probe, but MeLoMIA's decision step.

Same shadows (shadows.py) and same per-candidate TabPFN features as ICL-MIA;
only step 3 changes, using MeLoMIA's own code
(`mia.attacks.melomia.features`, `.meta`):
  * per-record calibration: each feature z-scored against the same candidate
    across the other shadows (training rows, leave-one-out) or all shadows
    (the release's rows) -- in and out pooled, as MeLoMIA does;
  * a classifier trained on all (candidate, shadow) rows with shadow membership,
    grouped by candidate; xgb / rf / lgbm / cat / mlp with MeLoMIA's default
    hyperparameters (no Optuna search), and their rank-averaged ensemble.
Feature sets: "genes" = the 200 per-gene terms l_rel - l_ref; "typ" = the
single ICL-MIA score.  Main environment (camda-mia).
"""
import glob
import os
import sys

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
from attack import scores  # noqa: E402
from calibrate import tpr_at  # noqa: E402
from mia.attacks.melomia import features as F, meta as MM  # noqa: E402

G = 200
ref = np.load("results/tabpfn_mia/full/ref_BRCA_n2_c1-978.npz")
l_ref = ref["l_ref"][:, :G]
sh = [np.load(f) for f in sorted(glob.glob(f"results/tabpfn_mia/shadows/shadow_BRCA_g{G}_k*.npz"))]
K, n = len(sh), len(l_ref)
y = np.concatenate([z["member"] for z in sh]).astype(int)
groups = np.tile(np.arange(n), K)
targets = [np.load(f"results/tabpfn_mia/full/rel_BRCA_s{s}_n2_c1-978.npz") for s in range(1, 6)]


def feats(l_rel, kind):
    if kind == "genes":
        return l_rel - l_ref
    return scores(l_rel, l_ref)["typ"][:, None]


for kind in ("typ", "genes"):
    X = np.concatenate([feats(z["l_rel"], kind) for z in sh])
    Xtr = F.per_record_train(X, K)
    mean, sd = F.per_record_reference(X, K)
    Xt = [F.per_record_apply(feats(t["l_rel"][:, :G], kind), mean, sd) for t in targets]
    preds = {}
    for name in ("xgb", "rf", "lgbm", "cat", "mlp"):
        clf = MM.get(name).train(Xtr, y, groups, None, None)
        preds[name] = [MM.get(name).predict(clf, x) for x in Xt]
    preds["ensemble"] = [np.mean([rankdata(preds[c][i]) for c in ("xgb", "rf", "lgbm", "cat", "mlp")], 0)
                         for i in range(5)]
    for name, p in preds.items():
        a = np.array([(roc_auc_score(t["member"], v), tpr_at(t["member"], v, 0.01)) for t, v in zip(targets, p)])
        print(f"features={kind:5s} {name:8s} AUC {a[:,0].mean():.3f}  TPR@1% {a[:,1].mean():.3f}   "
              f"per split {np.round(a[:,0], 3)}", flush=True)
