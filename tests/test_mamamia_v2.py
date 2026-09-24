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


def test_grid_edges_recover_the_generators_construction():
    # Build edges the generator's way (F^-1 of a histogram on the public grid,
    # linear inside cells), release equal-mass bins dithered uniformly, and
    # check the grid fit beats plain release quantiles.
    from mia.attacks.mamamia_v2 import _knot_edges, grid_edges
    rng = np.random.default_rng(3)
    K, n, grid = 8, 800, np.linspace(0.0, 24.0, 49)
    targets = np.concatenate([[0.005], np.arange(1, K) / K, [0.995]])
    X, E_true = [], []
    for _ in range(6):
        h = np.zeros(48)
        c = rng.integers(16, 26)
        h[c:c + 6] = rng.gamma(2.0, 1.0, 6)
        e = _knot_edges(np.concatenate([[0.0], np.cumsum(h) / h.sum()]), grid, targets)
        b = rng.integers(0, K, n)
        X.append(rng.uniform(e[b], e[b + 1]))
        E_true.append(e[1:-1])
    X, E_true = np.column_stack(X), np.array(E_true)
    got = grid_edges(X, K, n_jobs=1)
    quant = np.quantile(X, np.arange(1, K) / K, axis=0).T
    assert np.abs(got - E_true).mean() < np.abs(quant - E_true).mean()
