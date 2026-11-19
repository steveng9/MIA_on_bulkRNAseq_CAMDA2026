"""Shadow releases for calibrating ICL-MIA (LiRA-style, membership known).

Each shadow k:
  1. context  = the candidate pool minus one fold.  Shadows come in rounds of
     a 5-fold partition (stratified by subtype, fresh shuffle per round), so
     every candidate is left out of exactly one shadow per round: K = 30 gives
     24 "in" and 6 "out" shadows per candidate.  ~871 rows, the target's size.
  2. release  = the real generator (`mia.generators.tabpfn.TabPFNGenerator`,
     TabPFN-3.5, default ensemble, 3 orderings, t = 1) on that context, with
     its own seed, as many rows as context rows, restricted to the first
     --genes genes.  The generator fills genes in order, so this is exactly
     the first --genes columns of a full release.
  3. l_rel    = ICL-MIA's release term for every candidate on those genes,
     computed exactly as for the target (`attack.py --part rel`: TabPFN
     regressor, n_estimators = 2, random_state = 0, sample-id order).

The pool term is shared with the target (`attack.py --part ref`).
Run disjoint --k ranges in parallel; each shadow is one .npz.  Runs in camda_sota.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
from data import load  # noqa: E402

os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")


def shadow_folds(y_sorted: np.ndarray, k: int) -> np.ndarray:
    """Boolean 'in context' mask over the pool (sample-id order) for shadow k."""
    r, f = divmod(k, 5)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=100 + r).split(y_sorted, y_sorted))
    mask = np.ones(len(y_sorted), bool)
    mask[folds[f][1]] = False
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="BRCA")
    ap.add_argument("--k", default="0:30", help="shadow index range a:b")
    ap.add_argument("--genes", type=int, default=200)
    ap.add_argument("--n_estimators", type=int, default=2, help="for scoring, as for the target")
    ap.add_argument("--out", default="results/tabpfn_mia/shadows")
    args = ap.parse_args()

    import torch
    from tabpfn import TabPFNRegressor
    from mia.generators.tabpfn import TabPFNGenerator

    reg = TabPFNRegressor(device="cuda", random_state=0, n_estimators=args.n_estimators,
                          ignore_pretraining_limits=True)

    def logp(ctx, query, c):
        reg.fit(ctx[:, :c], ctx[:, c])
        pred = reg.predict(query[:, :c], output_type="full")
        lg = torch.as_tensor(pred["logits"]).double()
        y = torch.as_tensor(query[:, c], dtype=lg.dtype, device=lg.device)
        return -pred["criterion"].double().forward(lg, y).detach().cpu().numpy().ravel()

    t = load(args.dataset, 1)                       # any split: the pool is the same
    order = np.argsort(t.ids)
    ids, X, y = np.array(t.ids)[order], t.X[order], t.y[order]
    P = np.column_stack([y.astype(np.float32), X])
    G = args.genes
    os.makedirs(args.out, exist_ok=True)
    a, b = (int(v) for v in args.k.split(":"))
    for k in range(a, b):
        path = f"{args.out}/shadow_{args.dataset}_g{G}_k{k:03d}.npz"
        if os.path.exists(path):
            continue
        inn = shadow_folds(y, k)
        t0 = time.time()
        gen = TabPFNGenerator(seed=5000 + k, device="cuda", verbose=False)
        gen.fit(X[inn][:, :G], y[inn], n_classes=len(t.classes))
        S, yS = gen.sample(int(inn.sum()))
        t_gen = time.time() - t0
        R = np.column_stack([yS.astype(np.float32), S])
        l_rel = np.stack([logp(R, P, c) for c in range(1, G + 1)], axis=1)
        np.savez_compressed(path, l_rel=l_rel, member=inn.astype(int), ids=ids, y=y,
                            S=S.astype(np.float32), yS=yS, seconds_gen=t_gen,
                            seconds_score=time.time() - t0 - t_gen)
        print(f"shadow {k}: {int(inn.sum())} in context, generated {G} genes in {t_gen:.0f}s, "
              f"scored in {time.time()-t0-t_gen:.0f}s", flush=True)


if __name__ == "__main__":
    main()
