"""Speed / fidelity of cheaper TabPFN settings for ICL-MIA (split 1, a few columns)."""
import os, sys, time
import numpy as np, torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load
from tabpfn import TabPFNRegressor
os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
t = load("BRCA", 1)
order = np.argsort(t.ids); inv = np.argsort(order)
P = np.column_stack([t.y.astype(np.float32), t.X]); R = np.column_stack([t.yS.astype(np.float32), t.S])
folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(P[order], t.y[order]))
folds = [(order[a], order[b]) for a, b in folds]

def run(c, n_est=None, topk=None):
    kw = {} if n_est is None else {"n_estimators": n_est}
    reg = TabPFNRegressor(device="cuda", random_state=0, ignore_pretraining_limits=True, **kw)
    feats = np.arange(c)
    if topk is not None and c - 1 > topk:
        g = P[:, 1:c]; tgt = P[:, c]
        r = np.abs([np.corrcoef(g[:, j], tgt)[0, 1] for j in range(g.shape[1])])
        feats = np.concatenate([[0], 1 + np.argsort(-np.nan_to_num(r))[:topk]])   # label + top-k genes
    def lp(ctx, q):
        reg.fit(ctx[:, feats], ctx[:, c]); p = reg.predict(q[:, feats], output_type="full")
        lg = torch.as_tensor(p["logits"]).double(); y = torch.as_tensor(q[:, c], dtype=lg.dtype, device=lg.device)
        return -p["criterion"].double().forward(lg, y).detach().cpu().numpy().ravel()
    torch.cuda.synchronize(); t0 = time.time()
    lr = lp(R, P); lf = np.zeros(len(P))
    for tr, te in folds: lf[te] = lp(P[tr], P[te])
    torch.cuda.synchronize()
    return lr - lf, (time.time() - t0) / 6, reg

for c in (300, 600, 900):
    ref, tref, reg = run(c)
    print(f"col {c}: default ({getattr(reg, 'n_estimators', '?')} -> {len(getattr(reg, 'ensemble_configs_', [])) or '?'} members) "
          f"{tref:.1f}s/fit AUC {roc_auc_score(t.member, ref):.3f}", flush=True)
    for name, kw in [("n_est=2", {"n_est": 2}), ("n_est=1", {"n_est": 1}), ("top128", {"topk": 128}),
                     ("top128,n_est=2", {"topk": 128, "n_est": 2})]:
        d, tt, _ = run(c, **kw)
        print(f"    {name:15s} {tt:.1f}s/fit ({tref/tt:.1f}x)  corr with default {np.corrcoef(d, ref)[0,1]:.3f}  "
              f"AUC {roc_auc_score(t.member, d):.3f}", flush=True)
