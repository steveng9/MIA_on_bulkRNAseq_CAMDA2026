"""Eigenvalue spectra of 20,000 synthetic rows from the split-1 MVN and CVAE targets.

MVN is refitted on the split-1 training rows (deterministic fit); CVAE loads the saved
target checkpoint.  Reports where the spectrum falls off a cliff, pooled and per subtype.
"""
import sys, json
sys.path.insert(0, "/home/golobs/MIA_on_bulkRNAseq_CAMDA2026")
import numpy as np
from mia import datasets as D, targets as T
from mia import generators as G

N = 20000
out = {}


def spec(X):
    X = X - X.mean(0)
    return np.linalg.eigvalsh(X.T @ X / (len(X) - 1))[::-1]


def cliff(ev):
    """Largest drop between consecutive eigenvalues (log10), and where it happens."""
    l = np.log10(np.maximum(ev, 1e-30))
    d = l[:-1] - l[1:]
    k = int(np.argmax(d))
    return k + 1, float(d[k]), float(ev[k]), float(ev[k + 1])


for ds in ("BRCA", "COMBINED"):
    X = D.load_expression(ds).values.astype(np.float64)
    y = D.encode_subtypes(ds, D.load_subtypes(ds).values)
    m = D.membership_labels(ds, 1).astype(bool)
    Xt, yt, nc = X[m], y[m], D.n_classes(ds)
    for gname in ("mvn", "cvae"):
        p = dict(T.target_record(ds, gname, 1)["params"])
        g = G.build(gname, seed=0, device="cpu", **p)
        if gname == "mvn":
            g.fit(Xt, yt, nc)
        else:
            g.load(T._files(ds, "cvae", 1)["model"])
        Xs, ys = g.sample(N)
        Xs = np.asarray(Xs, np.float64)
        res = {"pooled": (cliff(ev := spec(Xs)), ev.tolist())}
        for c in np.unique(yt):
            evc = spec(Xs[ys == c])
            res[f"class{c}"] = (cliff(evc), int((yt == c).sum()), int((ys == c).sum()), evc.tolist())
        out[f"{ds}/{gname}"] = res
        k, d, a, b = res["pooled"][0]
        print(f"{ds} {gname} (train {m.sum()}): pooled cliff after component {k}: "
              f"{a:.3g} -> {b:.3g} ({d:.1f} orders of magnitude)", flush=True)
        for c in np.unique(yt):
            (k, d, a, b), ntr, nsy = res[f"class{c}"][:3]
            print(f"    subtype {c} (train {ntr}, syn {nsy}): cliff after {k}: {a:.3g} -> {b:.3g} ({d:.1f})", flush=True)
    out[f"{ds}/real_train"] = {"pooled": (cliff(ev := spec(Xt)), ev.tolist())}
json.dump(out, open("/home/golobs/MIA_on_bulkRNAseq_CAMDA2026/results/spectrum_mvn_cvae_s1.json", "w"))
