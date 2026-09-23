import numpy as np

from mia.attacks.mamamia_v2 import MAMAMIAv2, _log_ratio, recover_edges


def test_log_ratio_matches_hand_count():
    rng = np.random.default_rng(0)
    real = rng.integers(0, 4, (50, 3))
    syn = rng.integers(0, 4, (40, 3))
    got = _log_ratio(real, syn, real, 3, 4, alpha=0.5)
    for t in range(3):
        ps = (np.bincount(syn[:, t], minlength=4) + 0.5) / (40 + 2.0)
        pa = (np.bincount(real[:, t], minlength=4) + 0.5) / (50 + 2.0)
        np.testing.assert_allclose(got[:, t], np.log(ps[real[:, t]] / pa[real[:, t]]))


def test_access_labels():
    a = MAMAMIAv2.access
    assert a("public", "aux", "dp_quantile") == "black-box"
    assert a("recovered", "recovered", "dp_quantile") == "black-box"
    assert a("public", "known", "uniform") == "black-box"      # public grid
    assert a("public", "known", "dp_quantile") == "white-box"
    assert a("true", "recovered", "uniform") == "white-box"


def test_uniform_edges_ignore_access_choice():
    X = np.random.default_rng(1).normal(10, 1, (30, 5))
    p = {"n_bins": 8, "binning": "uniform", "bin_range": [0.0, 24.0]}
    E = recover_edges(X, p, "recovered", X, "BRCA", "pgm", 1)
    np.testing.assert_allclose(E[0], np.linspace(0, 24, 9)[1:-1])
    np.testing.assert_allclose(recover_edges(X, p, "aux", X, "BRCA", "pgm", 1), E)


def test_recovered_dp_quantile_edges_are_release_quantiles():
    X = np.random.default_rng(2).normal(10, 1, (1000, 3))
    p = {"n_bins": 4, "binning": "dp_quantile"}
    E = recover_edges(X, p, "recovered", None, "BRCA", "pgm", 1)
    np.testing.assert_allclose(E, np.quantile(X, [0.25, 0.5, 0.75], axis=0).T)
