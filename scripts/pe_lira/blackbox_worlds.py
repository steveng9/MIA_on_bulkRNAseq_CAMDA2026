"""Black-box attacks on the PE-Tabular worlds: final released synthetic data only (no counts,
no selections, no transcript), on the same targets as the count/selection attacks.

  dcr      -distance from the target to its nearest synthetic record of the same class
  knn_dr   log(d_k(target, Ref) / d_k(target, synthetic)), same class, k = 5: a DOMIAS-style
           density ratio (synthetic vs reference) with k-NN density estimates

Embedding: the attacker-side public-bounds embedder (bounds from the released candidates),
as in the evo-mia attacks.  Pooled over worlds after per-world standardisation, as in
world_lira.py; also re-reports A4 / B5 / C1 from score_worlds.py on the same targets.

    python scripts/pe_lira/blackbox_worlds.py breast-cancer_eps10
"""
import glob, os, sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.neighbors import NearestNeighbors

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "artifacts", "evo_mia_head"))
from attacks.tabular.transcript_tabular import (LABEL, TAB, _checkpoints, info_from_candidates,  # noqa: E402
                                                make_embedder, rows_from_csv, run_metadata)


def tpr(y, s, f):
    a, b, _ = roc_curve(y, s)
    return float(np.interp(f, a, b))


def kdist(ref, q, k):
    nn = NearestNeighbors(n_neighbors=min(k, len(ref))).fit(ref)
    return nn.kneighbors(q)[0][:, -1]


def world(wdir, k=5):
    run = os.path.join(wdir, "run")
    md = run_metadata(run)
    embed = make_embedder(info_from_candidates(run, md), md)
    cks = _checkpoints(run)
    final = pd.read_pickle(os.path.join(cks[max(cks)], "data_frame.pkl"))   # the released synthetic data
    S, sc = embed(final[TAB].tolist()), final[LABEL].to_numpy().astype(int)
    mem, mc = rows_from_csv(os.path.join(wdir, "members.csv"), md)
    non, nc = rows_from_csv(os.path.join(wdir, "nonmembers.csv"), md)
    ref, rc = rows_from_csv(os.path.join(wdir, "ref.csv"), md)
    X, cls = embed(mem + non), np.concatenate([mc, nc])
    R = embed(ref)
    y = np.r_[np.ones(len(mem)), np.zeros(len(non))].astype(int)
    dcr, kdr = np.zeros(len(X)), np.zeros(len(X))
    for c in np.unique(cls):
        i = cls == c
        Sc, Rc = S[sc == c], R[rc == c]
        if len(Sc) == 0 or len(Rc) == 0:
            continue
        dcr[i] = -kdist(Sc, X[i], 1)
        kdr[i] = np.log(kdist(Rc, X[i], k) + 1e-9) - np.log(kdist(Sc, X[i], k) + 1e-9)
    return y, {"dcr": dcr, "knn_dr": kdr}


def main():
    setting = sys.argv[1]
    W = sorted(glob.glob(os.path.expanduser(f"~/privacy/nfree_worlds/{setting}_w*")),
               key=lambda p: int(p.rsplit("_w", 1)[1]))
    Y, Sc = [], {}
    for w in W:
        y, s = world(w)
        Y.append(y)
        prev = np.load(f"{REPO}/results/pe_lira/{setting}/{os.path.basename(w)}.npz", allow_pickle=True)
        assert (prev["labels"] == y).all(), "targets differ from score_worlds.py"
        for v in ("A4", "B5", "C1"):
            s[v] = prev[v]
        for name, v in s.items():
            Sc.setdefault(name, []).append((v - v.mean()) / (v.std() + 1e-12))
    y = np.concatenate(Y)
    view = {"dcr": "final synthetic data only", "knn_dr": "final synthetic data + Ref",
            "B5": "survival (selection, no counts)", "C1": "rank (selection order, no counts)",
            "A4": "noised counts (per record)"}
    print(f"{setting}: {len(W)} worlds, {len(y)} targets")
    for name in ("dcr", "knn_dr", "B5", "C1", "A4"):
        s = np.concatenate(Sc[name])
        print(f"  {name:7s} {view[name]:36s} AUC {roc_auc_score(y, s):.3f}  TPR@1% {tpr(y, s, 0.01):.3f}  "
              f"TPR@0.1% {tpr(y, s, 0.001):.3f}")


if __name__ == "__main__":
    main()
