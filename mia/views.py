"""Temporary views of a cohort: what the adversary sees, changed for one block.

    with view("COMBINED", reference=tpm_ref, genes=idx, extra=df, extra_labels=s):
        scores = attack.score("COMBINED", "cvae", 1)

Inside the block the loaders in `mia.datasets` and `mia.targets` answer for
`dataset` as if

  reference     the auxiliary set were this frame (None = no auxiliary set);
  genes         only these gene columns existed, in the candidates, the
                release and the auxiliary set alike;
  extra         these rows were appended to the candidate pool (never members
                of any target), with classes `extra_labels`.

The targets themselves are untouched: a view changes the attack's inputs, not
what was released.  Only attacks that keep nothing on disk may run inside a
view (MahalaMIA, MAMA-MIA, the generic baselines, RedSigma).  MeLoMIA caches
features per target name and would store view-specific features under the
canonical key.
"""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import pandas as pd

from . import datasets as D
from . import targets as T

KEEP = object()
STATEFUL = ("melomia_nd", "melomia_cvae", "melomia_tabsyn", "melomia_e2e_cvae")


@contextmanager
def view(dataset: str, reference=KEEP, genes=None, extra: pd.DataFrame | None = None,
         extra_labels=None):
    base_expr = D.load_expression(dataset)
    base_ref = D.load_reference(dataset)
    base_lab = D.load_subtypes(dataset)
    cols = list(base_expr.columns)
    keep = cols if genes is None else [cols[i] for i in np.asarray(genes)]
    gidx = None if genes is None else np.asarray(genes)

    expr = base_expr
    labels = base_lab
    if extra is not None:
        if extra_labels is None:
            raise ValueError("extra rows need extra_labels")
        expr = pd.concat([base_expr, extra.loc[:, cols]])
        labels = pd.concat([base_lab, pd.Series(np.asarray(extra_labels), index=extra.index)])
        if expr.index.duplicated().any():
            raise ValueError("extra rows repeat candidate ids")
    expr = expr.loc[:, keep]

    ref = base_ref if reference is KEEP else reference
    if ref is not None:
        ref = ref.loc[:, keep]

    orig = (D.load_expression, D.load_reference, D.load_subtypes, T.load_target)

    def load_expression(name):
        return expr if name == dataset else orig[0](name)

    def load_reference(name):
        return ref if name == dataset else orig[1](name)

    def load_subtypes(name):
        return labels if name == dataset else orig[2](name)

    def load_target(name, generator, split, allow_broken=False):
        tg = orig[3](name, generator, split, allow_broken)
        if name == dataset and gidx is not None:
            tg = dict(tg, X=tg["X"][:, gidx])
        return tg

    def clear():
        from .attacks import mamamia_v2
        D.load_target_splits.cache_clear()
        D.class_names.cache_clear()
        mamamia_v2._GRID_CACHE.clear()      # keyed by target name, not by gene set

    D.load_expression, D.load_reference, D.load_subtypes = (
        load_expression, load_reference, load_subtypes)
    T.load_target = load_target
    clear()
    try:
        yield expr
    finally:
        D.load_expression, D.load_reference, D.load_subtypes, T.load_target = orig
        clear()


# The attacks that can run inside a view, at the settings the grid reports.
ROSTER = {
    "MahalaMIA (as submitted)": ("mahalamia", {"use_reference": True}),
    "MahalaMIA (ridge 1e-4)": ("mahalamia", {"use_reference": True, "covariance": "ridge",
                                             "ridge_alpha": 1e-4}),
    "MahalaMIA (no aux)": ("mahalamia", {"use_reference": False}),
    "MahalaMIA (no aux, ridge 1e-4)": ("mahalamia", {"use_reference": False,
                                                     "covariance": "ridge", "ridge_alpha": 1e-4}),
    "MAMA-MIA v1": ("mamamia", {"n_bins": 4}),
    "MAMA-MIA v2": ("mamamia_v2", {"cliques": "public", "edges": "auto", "class_centre": True}),
    "RedSigma": ("redsigma", {}),
    "GAN-leaks": ("generic", {"method": "gan_leaks"}),
    "MC": ("generic", {"method": "mc"}),
    "GAN-leaks cal.": ("generic", {"method": "gan_leaks_cal"}),
    "DOMIAS-KDE": ("generic", {"method": "domias_kde"}),
}
USES_AUX = ("MahalaMIA (as submitted)", "MahalaMIA (ridge 1e-4)", "RedSigma",
            "GAN-leaks cal.", "DOMIAS-KDE")


def build(label: str):
    from . import attacks as A
    cls, params = ROSTER[label]
    return A.build(cls, device="cpu", verbose=False, **params)
