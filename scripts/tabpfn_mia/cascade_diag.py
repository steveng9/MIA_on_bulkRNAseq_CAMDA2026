"""Diagnostic: does a TabPFN release copy members more in late columns?

Per block of genes (in generation order), AUC of -min distance from each
candidate to the release; and, with neighbours matched on the first quarter of
genes only, AUC of -residual on each later block.  A copy cascade shows as AUC
rising with block index.
"""
import sys
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from data import load

ds = sys.argv[1] if len(sys.argv) > 1 else "BRCA"
for split in range(1, 6):
    t = load(ds, split)
    mu, sd = t.X.mean(0), t.X.std(0) + 1e-6
    A = torch.tensor((t.X - mu) / sd, device="cuda")
    B = torch.tensor((t.S - mu) / sd, device="cuda")
    d = A.shape[1]
    blocks = [torch.tensor(b, device="cuda") for b in np.array_split(np.arange(d), 8)]
    per = [roc_auc_score(t.member, -torch.cdist(A[:, b], B[:, b]).min(1).values.cpu().numpy())
           for b in blocks]
    nn = torch.cdist(A[:, : d // 4], B[:, : d // 4]).topk(5, largest=False).indices
    res = [roc_auc_score(t.member, -((A[:, None, b] - B[nn][:, :, b]) ** 2).mean(-1).min(1).values.cpu().numpy())
           for b in blocks]
    full = roc_auc_score(t.member, -torch.cdist(A, B).min(1).values.cpu().numpy())
    print(f"split {split} full-NN {full:.3f} | per-block " + " ".join(f"{a:.3f}" for a in per)
          + " | Q1-matched " + " ".join(f"{a:.3f}" for a in res), flush=True)
