"""mia.views: a view changes the attack's inputs for one block and nothing else."""

import numpy as np
import pytest

from mia import attacks as A
from mia import datasets as D
from mia import targets as T
from mia import views as V

pytestmark = pytest.mark.skipif(not T.exists("BRCA", "mvn", 1), reason="BRCA/mvn/split_1 not built")


def _attack():
    return A.build("mahalamia", device="cpu", verbose=False, use_reference=False,
                   covariance="ridge", ridge_alpha=1e-4, calibrate=False)


def test_identity_view_and_restore():
    base = _attack().score("BRCA", "mvn", 1)
    loaders = (D.load_expression, D.load_reference, D.load_subtypes, T.load_target)
    with V.view("BRCA", genes=np.arange(len(D.gene_names("BRCA")))):
        inside = _attack().score("BRCA", "mvn", 1)
    assert (D.load_expression, D.load_reference, D.load_subtypes, T.load_target) == loaders
    np.testing.assert_allclose(inside, base)


def test_gene_subset_cuts_every_input():
    idx = np.arange(50)
    with V.view("BRCA", genes=idx) as expr:
        assert expr.shape[1] == 50
        assert T.load_target("BRCA", "mvn", 1)["X"].shape[1] == 50
    assert D.load_expression("BRCA").shape[1] == 978


def test_extra_rows_are_appended_and_leave_candidates_alone():
    expr = D.load_expression("BRCA")
    extra = expr.iloc[:5].copy()
    extra.index = [f"extra_{i}" for i in range(5)]
    labels = D.load_subtypes("BRCA").values[:5]
    base = _attack().score("BRCA", "mvn", 1)
    with V.view("BRCA", extra=extra, extra_labels=labels):
        s = _attack().score("BRCA", "mvn", 1)
    assert len(s) == len(expr) + 5
    np.testing.assert_allclose(s[:len(expr)], base)
    np.testing.assert_allclose(s[len(expr):], base[:5])   # same rows, same score
    assert len(D.membership_labels("BRCA", 1)) == len(expr)
