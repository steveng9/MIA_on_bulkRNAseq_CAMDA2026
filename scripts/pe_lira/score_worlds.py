"""Per-world, per-target scores of the EVO-MIA attack variants on the PE-Tabular testbed worlds.

Uses evo-mia's own scorer (`attacks.tabular.run_worlds_eval.score_world`) from a read-only
export of its last commit (artifacts/evo_mia_head, made with `git archive 8176c9b`), with
cache=False so nothing is written into the world folders.  One .npz per world, here.

    python scripts/pe_lira/score_worlds.py --worlds '~/privacy/nfree_worlds/breast-cancer_eps10_w*' \
        --out results/pe_lira/breast-cancer_eps10 --workers 8
"""
import argparse, glob, os, sys, time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EVO = os.path.join(REPO, "artifacts", "evo_mia_head")
sys.path.insert(0, EVO)


def job(a):
    wdir, out = a
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from attacks.tabular.run_worlds_eval import DEFAULT_VARIANTS, score_world
    t0 = time.time()
    res = score_world(wdir, DEFAULT_VARIANTS, embedder="public", cache=False)
    keep = {k: v for k, v in res.items() if not k.endswith("__per_round") and k not in ("fits",)}
    path = os.path.join(out, os.path.basename(wdir) + ".npz")
    np.savez(path, **keep)
    return os.path.basename(wdir), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    W = sorted(glob.glob(os.path.expanduser(a.worlds)), key=lambda p: int(p.rsplit("_w", 1)[1]))
    todo = [(w, a.out) for w in W if not os.path.exists(os.path.join(a.out, os.path.basename(w) + ".npz"))]
    with ProcessPoolExecutor(a.workers) as ex:
        for name, s in ex.map(job, todo):
            print(f"{name}: {s:.0f}s", flush=True)


if __name__ == "__main__":
    main()
