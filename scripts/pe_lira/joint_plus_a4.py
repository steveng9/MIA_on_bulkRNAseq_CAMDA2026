"""Ensemble: standardised joint score + standardised A4 score, per world, on the testbed worlds.
A4 per-world scores come from score_worlds.py (results/pe_lira/<setting>/<world>.npz)."""
import glob, os, sys
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from joint_sparse import joint_scores, load_world
z = lambda v: (v - v.mean()) / (v.std() + 1e-12)
def tpr(y, s, f):
    a, b, _ = roc_curve(y, s); return float(np.interp(f, a, b))
setting = sys.argv[1]
Y, S = [], {"joint": [], "A4": [], "joint+A4": []}
for w in sorted(glob.glob(os.path.expanduser(f"~/privacy/nfree_worlds/{setting}_w*"))):
    tr, view, ctx, cls, lab = load_world(w)
    j, _ = joint_scores(view, ctx, cls, float(tr["meta"]["sigma"]))
    a4 = np.load(f"results/pe_lira/{setting}/{os.path.basename(w)}.npz", allow_pickle=True)
    assert (a4["labels"] == lab).all()
    Y.append(lab); S["joint"].append(z(j)); S["A4"].append(z(a4["A4"])); S["joint+A4"].append(z(j) + z(a4["A4"]))
y = np.concatenate(Y)
print(setting + ": " + " | ".join(f"{k} AUC {roc_auc_score(y, np.concatenate(v)):.3f} TPR@1% {tpr(y, np.concatenate(v), 0.01):.3f}" for k, v in S.items()))
