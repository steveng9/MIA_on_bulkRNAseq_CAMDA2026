"""Give the MVN targets the generator.pt a white-box attack loads.

The MVN targets were built before MVNGenerator could be saved.  Its fit is a
deterministic function of the training split (per-class mean and covariance),
so the model can be rebuilt exactly: refit with the recipe and seed recorded in
the target's meta.json.  Two checks that it is *the* target before saving:

  labels  sampling with the recorded seed re-emits the release's label
          sequence exactly (it depends only on the class counts and the seed);
  span    every released row of class c lies in mu_c + span(class c's centred
          training rows), to float32 precision.  The class covariance has rank
          n_c - 1 < 978, so the release of the true model lives in that affine
          subspace and the release of any other model does not (BRCA split 1:
          relative residual 5e-7 against its own split's fit, 0.62 against
          split 2's).

The rows themselves cannot be re-emitted: numpy draws a multivariate normal
through an SVD of the covariance, whose singular-vector signs depend on the
LAPACK build and thread count, so the same model gives the same distribution
but different draws.  A model that fails either check is reported and not saved.

    python scripts/whitebox/materialise_mvn_models.py --datasets BRCA COMBINED
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mia import datasets as D  # noqa: E402
from mia import generators as G  # noqa: E402
from mia import targets as T  # noqa: E402

#: Largest relative distance of a released row from its class's affine span
#: that still counts as inside it.  float32 rounding gives ~1e-6; another
#: split's model gives ~0.6.
SPAN_TOL = 1e-4


def span_residual(X_train, y_train, X_rel, y_rel) -> float:
    """Max over released rows of |x - mu_c - P_c(x - mu_c)| / |x - mu_c|."""
    worst = 0.0
    for c in np.unique(y_rel):
        Xc = X_train[y_train == c].astype(np.float64)
        mu = Xc.mean(axis=0)
        _, s, Vt = np.linalg.svd(Xc - mu, full_matrices=False)
        V = Vt[s > s[0] * 1e-10]
        R = X_rel[y_rel == c].astype(np.float64) - mu
        r = np.linalg.norm(R - (R @ V.T) @ V, axis=1) / np.linalg.norm(R, axis=1)
        worst = max(worst, float(r.max()))
    return worst


def materialise(dataset: str, split: int, force: bool = False) -> str:
    f = T._files(dataset, "mvn", split)
    if f["model"].exists() and not force:
        return "exists"
    meta = json.loads(f["meta"].read_text())
    gen = G.build("mvn", seed=meta["seed"], device="cpu", verbose=False, **meta["params"])
    X, y, _ = D.training_subset(dataset, split)
    gen.fit(X, y, D.n_classes(dataset))

    tg = T.load_target(dataset, "mvn", split)
    _, y_re = gen.sample(len(tg["X"]))
    if not np.array_equal(y_re, tg["y_int"]):
        return "LABELS DIFFER - not saved"
    r = span_residual(X, y, tg["X"], tg["y_int"])
    if r > SPAN_TOL:
        return f"RELEASE OUTSIDE THE MODEL'S SPAN (residual {r:.3g}) - not saved"
    gen.save(f["model"])

    back = G.build("mvn", seed=meta["seed"], device="cpu", verbose=False,
                   **meta["params"]).load(f["model"])
    same = all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
               for a, b in zip(back.moments().values(), gen.moments().values()))
    if not same:
        f["model"].unlink()
        return "SAVE/LOAD ROUND TRIP FAILED - removed"
    return f"saved (labels re-emitted; release in the model's span, residual {r:.1g})"


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--datasets", nargs="+", default=["BRCA", "COMBINED"])
    p.add_argument("--splits", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    bad = 0
    for ds in a.datasets:
        for s in a.splits:
            msg = materialise(ds, s, a.force)
            bad += not msg.startswith(("saved", "exists"))
            print(f"{ds}/mvn/split_{s}: {msg}", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
