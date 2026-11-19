"""Probe: per-column TabPFN log-density of candidates under three contexts.

Runs in the camda_sota env.  For a few gene columns c (generation order:
label, gene 1, ..., gene d), feature set = [label, genes < c].
  members : context = the training rows (query access, upper bound)
  release : context = the synthetic release (black box)
  pool    : context = the other 4 of 5 folds of the candidate pool (calibration)
"""
import os, sys, time
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load
import torch
from tabpfn import TabPFNRegressor

os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
ds, split = sys.argv[1], int(sys.argv[2])
cols = [int(c) for c in sys.argv[3].split(",")]
t = load(ds, split)
reg = TabPFNRegressor(device="cuda", random_state=0, ignore_pretraining_limits=True)

def table(X, y):
    return np.column_stack([y.astype(np.float32), X])

C, R = table(t.X, t.y), table(t.S, t.yS)
M = C[t.member == 1]

def logp(ctx, query, c):
    reg.fit(ctx[:, :c], ctx[:, c])
    pred = reg.predict(query[:, :c], output_type="full")
    lg = torch.as_tensor(pred["logits"]).double()
    crit = pred["criterion"].double()
    y = torch.as_tensor(query[:, c], dtype=lg.dtype, device=lg.device)
    return -crit.forward(lg, y).detach().cpu().numpy().ravel()

folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(C, t.member * 10 + t.y))
for c in cols:
    t0 = time.time()
    lm, lr = logp(M, C, c), logp(R, C, c)
    lp = np.zeros(len(C))
    for tr, te in folds:
        lp[te] = logp(C[tr], C[te], c)
    a = lambda s: roc_auc_score(t.member, s)
    print(f"col {c:4d}  members {a(lm):.3f} cal {a(lm-lp):.3f} | release {a(lr):.3f} cal {a(lr-lp):.3f}"
          f" | pool {a(lp):.3f}  ({time.time()-t0:.0f}s)", flush=True)
