"""Decision-rule ablation: LiRA on MeLoMIA-ND's own cached features.

Same shadows, same probes (the proxies MeLoMIA itself trained), only the
decision rule changes: MeLoMIA's tuned classifier ensemble -> per-candidate
Gaussian in/out likelihood ratio, as in ICL-MIA.  Statistic per candidate and
noise level t: -mean over noise draws of log denoising loss.
    python scripts/tabpfn_mia/lira_on_melomia.py
"""
import glob, os, sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibrate import lira, tpr_at

ROOT = "artifacts/pgg/attacks/melomia_nd/BRCA_G200"
idx = pd.read_csv("results/tabpfn_mia/melomia_runs/index.csv")
for arm, variant in (("n600_basetabpfn", "melomia_nd_basetabpfn"), ("n600", "melomia_nd")):
    files = sorted(glob.glob(f"{ROOT}/{arm}/features/shadow_*.npz"), key=lambda f: int(f.split("_")[-1][:-4]))
    sh = [np.load(f, allow_pickle=True) for f in files]
    M = np.stack([z["y_member"].astype(bool) for z in sh])
    S = np.stack([-np.log(z["losses"]).mean(2) for z in sh])          # (K, n, T)
    K, n, T = S.shape
    rounds = np.arange(K) // 5
    # best single noise level, chosen on held-out shadow rounds only
    held = np.zeros(T)
    for r in range(6):
        tr = rounds != r
        for i in np.flatnonzero(~tr):
            held += [roc_auc_score(M[i], lira(S[i, :, t], S[tr][:, :, t], M[tr])[0]) for t in range(T)]
    t_best = int(np.argmax(held))
    rows = []
    for s in range(1, 6):
        z = np.load(f"{ROOT}/{arm}/proxy_features/tabpfn_split_{s}.npz", allow_pickle=True)
        P = -np.log(z["losses"]).mean(1 if False else 2)                 # (n, T)
        run = idx[(idx.variant == variant) & (idx.split == s)].iloc[-1]
        sc = pd.read_csv(f"results/tabpfn_mia/melomia_runs/runs/{run.run_id}/scores.csv")
        m = sc["y_member"].values
        per_t = np.stack([lira(P[:, t], S[:, :, t], M)[0] for t in range(T)], 1)
        summ = per_t.sum(1); best = per_t[:, t_best]
        rows.append([run.auc, run["tpr_at_fpr_0.01"], roc_auc_score(m, summ), tpr_at(m, summ, 0.01),
                     roc_auc_score(m, best), tpr_at(m, best, 0.01)])
    r = np.array(rows)
    print(f"{variant}: classifier AUC {r[:,0].mean():.3f} TPR@1% {r[:,1].mean():.3f} | "
          f"LiRA summed over noise levels AUC {r[:,2].mean():.3f} TPR@1% {r[:,3].mean():.3f} | "
          f"LiRA at best level (t index {t_best}) AUC {r[:,4].mean():.3f} TPR@1% {r[:,5].mean():.3f}")
    print("   per split, LiRA summed:", np.round(r[:, 2], 3), " classifier:", np.round(r[:, 0], 3))
