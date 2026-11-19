"""Oracle ceiling for passive membership inference on a TabPFN release.

The oracle knows every training row other than x, and compares how well two
contexts of the same size explain the release R:

    score(x) = log p(R | D_-x + {x}) - log p(R | D_-x + {u})
             = sum_j sum_c [ log p(R_jc | R_j,<c ; D_-x + {x}) - log p(R_jc | R_j,<c ; D_-x + {u}) ]

D_-x is the training set without x, u a fixed filler record (a non-member not
among the scored candidates), j runs over release rows and c over the chosen
gene columns, each conditioned on its full prefix (label + earlier genes) as
in the generator.

Why a swap and not add/remove: any one-row change of context moves TabPFN's
total log-likelihood of the release by ~1 nat, non-additively (data-dependent
preprocessing).  Comparing two contexts of equal size, both one row away from
D_-x, keeps that jitter symmetric between members and non-members.

--rs sets TabPFN's random_state.  With the generator's own (42 + split) a
member's "with x" context is exactly the generating context, so the score
also rewards matching the generator's randomness: the formal ceiling, which
needs D_-x exactly.  Any other seed removes that and leaves the smooth
influence of x on the release.

Approximation: one feature ordering instead of the generator's three.

Writes per-(candidate, column, release row) differences so other aggregations
(e.g. only the release rows nearest x) can be scored later.  Runs in camda_sota.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load  # noqa: E402

os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="BRCA")
    ap.add_argument("--split", type=int, default=1)
    ap.add_argument("--cols", default="10,20,30,40,50,60,70,80,90,100")
    ap.add_argument("--n_per_class", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0, help="candidate subsample")
    ap.add_argument("--out", default="results/tabpfn_mia")
    ap.add_argument("--rs", type=int, default=None, help="TabPFN random_state (default: the generator's)")
    ap.add_argument("--design", default="subset", choices=["subset", "swap"],
                    help="swap (first run) treats members and non-members differently; "
                         "subset gives every candidate the same operation")
    args = ap.parse_args()

    import torch
    from tabpfn import TabPFNRegressor
    # The generator's estimators used random_state = its seed (42 + split).
    rs = 42 + args.split if args.rs is None else args.rs
    reg = TabPFNRegressor(device="cuda", random_state=rs, ignore_pretraining_limits=True)

    def logp(ctx, query, c):
        reg.fit(ctx[:, :c], ctx[:, c])
        pred = reg.predict(query[:, :c], output_type="full")
        lg = torch.as_tensor(pred["logits"]).double()
        y = torch.as_tensor(query[:, c], dtype=lg.dtype, device=lg.device)
        return -pred["criterion"].double().forward(lg, y).detach().cpu().numpy().ravel()

    t = load(args.dataset, args.split)
    C = np.column_stack([t.y.astype(np.float32), t.X])
    R = np.column_stack([t.yS.astype(np.float32), t.S])
    cols = [int(c) for c in args.cols.split(",")]
    rng = np.random.default_rng(args.seed)
    mem_idx = np.flatnonzero(t.member == 1)
    non_idx = np.flatnonzero(t.member == 0)
    pick = np.concatenate([rng.choice(mem_idx, args.n_per_class, replace=False),
                           rng.choice(non_idx, args.n_per_class, replace=False)])
    D = C[t.member == 1]
    # Row of each member inside D, to swap it out.
    pos_in_D = {i: k for k, i in enumerate(mem_idx)}
    u = C[rng.choice(np.setdiff1d(non_idx, pick))]

    # design "subset": every candidate gets the identical operation, and no
    # evaluated context equals the generating one.  S_x is a random subset of
    # D_-x of size |D| - 2 (drop 1 more member for a member x, 2 for a
    # non-member); score = L(S_x + x) - L(S_x + u), x and u in the last row.
    # Fixed per candidate across columns.
    if args.design == "subset":
        S_idx = []
        for i in pick:
            rest = np.delete(np.arange(len(D)), pos_in_D[i]) if t.member[i] else np.arange(len(D))
            S_idx.append(np.sort(np.random.default_rng(1000 + int(i)).choice(
                rest, len(D) - 2, replace=False)))

    diff = np.zeros((len(pick), len(cols), len(R)), dtype=np.float32)
    t0 = time.time()
    for jc, c in enumerate(cols):
        if args.design == "subset":
            for k, i in enumerate(pick):
                S = D[S_idx[k]]
                diff[k, jc] = logp(np.vstack([S, C[i]]), R, c) - logp(np.vstack([S, u]), R, c)
        else:
            base = logp(D, R, c)                        # members:     D_-x + {x} = D
            base_u = logp(np.vstack([D, u]), R, c)      # non-members: D_-x + {u} = D + u
            for k, i in enumerate(pick):
                if t.member[i]:
                    Du = D.copy()
                    Du[pos_in_D[i]] = u                                    # D_-x + {u}
                    diff[k, jc] = base - logp(Du, R, c)
                else:
                    diff[k, jc] = logp(np.vstack([D, C[i]]), R, c) - base_u   # (D + x) - (D + u)
        s = diff[:, : jc + 1].sum((1, 2))
        print(f"col {c} ({jc+1}/{len(cols)}, {time.time()-t0:.0f}s) oracle AUC "
              f"{roc_auc_score(t.member[pick], s):.3f}", flush=True)

    suffix = "" if args.design == "swap" else "_subset"
    np.savez_compressed(f"{args.out}/oracle_{args.dataset}_s{args.split}_rs{rs}{suffix}.npz", diff=diff,
                        pick=pick, rs=rs, cols=np.array(cols), member=t.member[pick], y=t.y[pick])


if __name__ == "__main__":
    main()
