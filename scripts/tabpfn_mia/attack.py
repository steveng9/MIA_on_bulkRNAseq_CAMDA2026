"""ICL-MIA: a membership attack tailored to TabPFN's in-context generator.

The generator (tabpfn_extensions' TabPFNUnsupervisedModel, as wrapped by
`mia/generators/tabpfn.py`) fills columns in a fixed order -- label, gene 1,
..., gene d -- sampling each from TabPFN's predictive p(x_c | x_<c) with the
training rows as context.  The adversary holds the same public weights, so it
can evaluate the same conditionals under any context it likes.

Per candidate x and column c (in generation order, each on its full prefix):

    l_rel(x, c) = log p_TabPFN(x_c | x_<c ; context = the release)
    l_ref(x, c) = log p_TabPFN(x_c | x_<c ; context = candidate pool minus x's fold)

Scores (see `scores`):
    zsum   sum_c zscore_over_candidates(l_rel - l_ref)
    typ    zsum with the candidate's typicality, sum_c l_ref, regressed out
           (label-free; removes the per-candidate component shared across
           columns that otherwise caps the AUC as columns are added)

Two stages, because the pool is the same in every split:
    --part ref   l_ref, once per dataset.  Folds are drawn on the candidates
                 sorted by sample id, stratified by subtype only (the adversary
                 does not know membership), so they do not depend on the split.
    --part rel   l_rel, one fit per column per split.
Matrices are saved in sample-id order; `--cols a:b` computes a column range
so long runs can be split across GPUs and merged by `score.py`.

Runs in the camda_sota environment.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load  # noqa: E402

os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")


def _z(a):
    return (a - a.mean(0)) / (a.std(0) + 1e-12)


def scores(l_rel: np.ndarray, l_ref: np.ndarray) -> dict:
    zsum = _z(l_rel - l_ref).sum(1)
    T = np.column_stack([np.ones(len(zsum)), l_ref.sum(1)])
    typ = zsum - T @ np.linalg.lstsq(T, zsum, rcond=None)[0]
    return {"zsum": zsum, "typ": typ}


def parse_cols(spec: str, d: int, stride: int) -> list:
    all_cols = sorted(set(range(stride, d + 1, stride)) | {d})
    if not spec:
        return all_cols
    a, b = (int(v) for v in spec.split(":"))
    return [c for c in all_cols if a <= c < b]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["ref", "rel"], required=True)
    ap.add_argument("--dataset", default="BRCA")
    ap.add_argument("--splits", default="1,2,3,4,5", help="--part rel only")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--cols", default="", help="column range a:b (table index; genes are 1..d)")
    ap.add_argument("--n_estimators", type=int, default=2)
    ap.add_argument("--run", default="tabpfn_35")
    ap.add_argument("--out", default="results/tabpfn_mia/full")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    from tabpfn import TabPFNRegressor
    reg = TabPFNRegressor(device="cuda", random_state=args.seed, n_estimators=args.n_estimators,
                          ignore_pretraining_limits=True)

    def logp(ctx, query, c):
        reg.fit(ctx[:, :c], ctx[:, c])
        pred = reg.predict(query[:, :c], output_type="full")
        lg = torch.as_tensor(pred["logits"]).double()
        y = torch.as_tensor(query[:, c], dtype=lg.dtype, device=lg.device)
        return -pred["criterion"].double().forward(lg, y).detach().cpu().numpy().ravel()

    os.makedirs(args.out, exist_ok=True)
    splits = [int(s) for s in args.splits.split(",")] if args.part == "rel" else [1]
    for split in splits:
        t = load(args.dataset, split, args.run)
        order = np.argsort(t.ids)                                   # sample-id order
        ids = np.array(t.ids)[order]
        P = np.column_stack([t.y.astype(np.float32), t.X])[order]   # table column 0 = label
        d = P.shape[1] - 1
        cols = parse_cols(args.cols, d, args.stride)
        tag = f"{cols[0]}-{cols[-1]}"
        L = np.zeros((len(P), len(cols)))
        t0 = time.time()
        if args.part == "ref":
            folds = list(StratifiedKFold(5, shuffle=True, random_state=args.seed).split(P, t.y[order]))
            for j, c in enumerate(cols):
                for tr, te in folds:
                    L[te, j] = logp(P[tr], P[te], c)
                if (j + 1) % 50 == 0:
                    print(f"ref {args.dataset} {j+1}/{len(cols)} cols ({time.time()-t0:.0f}s)", flush=True)
            path = f"{args.out}/ref_{args.dataset}_n{args.n_estimators}_c{tag}.npz"
            np.savez_compressed(path, l_ref=L, cols=np.array(cols), ids=ids, y=t.y[order])
        else:
            R = np.column_stack([t.yS.astype(np.float32), t.S])
            for j, c in enumerate(cols):
                L[:, j] = logp(R, P, c)
                if (j + 1) % 50 == 0:
                    print(f"rel {args.dataset} s{split} {j+1}/{len(cols)} cols ({time.time()-t0:.0f}s)",
                          flush=True)
            path = f"{args.out}/rel_{args.dataset}_s{split}_n{args.n_estimators}_c{tag}.npz"
            np.savez_compressed(path, l_rel=L, cols=np.array(cols), ids=ids,
                                member=t.member[order], y=t.y[order])
        print(f"wrote {path} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
