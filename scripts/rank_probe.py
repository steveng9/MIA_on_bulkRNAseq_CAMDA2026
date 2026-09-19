#!/usr/bin/env python
"""Effective rank of each generator's synthetic covariance.

MahalaMIA reads membership out of the low-variance directions of the synthetic
covariance, so how well conditioned that covariance is decides how much of the
signal a pseudo-inverse throws away.  The cohort-size sweep says MVN and the
CVAE are ill-conditioned for different reasons -- MVN only when n < p, the CVAE
always -- and this measures that directly.

    python scripts/rank_probe.py

Run at n >> p (COMBINED's canonical splits) so any remaining deficiency cannot
be a sample-size artefact.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from mia import datasets as D, targets as TG

def spectrum(X):
    Xc = X - X.mean(0)
    s = np.linalg.svd(Xc, compute_uv=False)
    ev = s ** 2 / (len(X) - 1)
    ev = ev / ev.sum()
    # participation-ratio effective rank, and how many components hold 99%
    eff = 1.0 / np.sum(ev ** 2)
    k99 = int(np.searchsorted(np.cumsum(ev), 0.99) + 1)
    cond = float(s[0] / max(s[-1], 1e-30))
    return eff, k99, cond

ds = "COMBINED"
real = D.load_expression(ds).values.astype(np.float64)
print(f"{'source':<26} {'n':>6} {'eff.rank':>9} {'k99':>5} {'cond':>11}")
e, k, c = spectrum(real)
print(f"{'real cohort':<26} {len(real):>6} {e:>9.1f} {k:>5} {c:>11.2e}")
for gen in ("mvn", "cvae", "nd", "pgm"):
    try:
        t = TG.load_target(ds, gen, 1)
    except Exception as exc:
        print(f"{gen:<26} [unavailable: {exc}]")
        continue
    X = t["X"].astype(np.float64)
    e, k, c = spectrum(X)
    print(f"{gen + ' synthetic (split 1)':<26} {len(X):>6} {e:>9.1f} {k:>5} {c:>11.2e}")
