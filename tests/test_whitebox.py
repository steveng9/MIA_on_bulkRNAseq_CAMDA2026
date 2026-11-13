"""White-box attacks: the adversary holds the target model, not only its release.

Unit tests run anywhere.  The two integration tests read built targets and are
skipped when those are absent.
"""

import json

import numpy as np
import pytest

from mia import paths
from mia.attacks.mahalamia import (ClassMoments, MahalaMIA, class_shrunk_distance,
                                   class_shrunk_distance_from_moments, mahalanobis,
                                   model_moments, nearest_psd)
from mia.attacks.mamamia_v2 import MAMAMIAv2, _log_ratio, _table_probs
from mia.attacks.melomia.attack import MeLoMIACVAE
from mia.attacks.melomia.backends import _record_labels
from mia.generators.mvn import MVNGenerator

PGM = "pgm@binning=dp_quantile,edge_estimator=threshold,n_bins=16"


def _old_class_shrunk_distance(X_query, y_query, X_fit, y_fit, shrink, ridge_alpha):
    """The paper's MahalaMIA distance as written before the moments refactor."""
    S = nearest_psd(np.cov(X_fit, rowvar=False))
    p = S.shape[0]

    def inv(C):
        return np.linalg.inv(C + ridge_alpha * np.trace(C) / p * np.eye(p))

    d = np.full(len(X_query), np.nan)
    for c in np.unique(y_fit):
        rows = X_fit[y_fit == c]
        take = y_query == c
        if len(rows) < 3 or not take.any():
            continue
        C = (1.0 - shrink) * np.cov(rows, rowvar=False) + shrink * S
        d[take] = mahalanobis(X_query[take], rows.mean(axis=0), inv(C))
    missing = np.isnan(d)
    if missing.any():
        d[missing] = mahalanobis(X_query[missing], X_fit.mean(axis=0), inv(S))
    return d


def _toy(seed=0, n=120, p=15, k=4):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, n)
    y[:2] = k            # a class with two rows: no covariance, pooled fallback
    X = rng.normal(size=(n, p)) + y[:, None]
    return X, y


# ── MahalaMIA ────────────────────────────────────────────────────────────────

def test_black_box_distance_unchanged_by_moments_refactor():
    X, y = _toy()
    Xq, yq = _toy(seed=1, n=60)
    yq[0] = 99           # a class the release never emitted
    np.testing.assert_array_equal(
        class_shrunk_distance(Xq, yq, X, y, 0.5, 1e-4),
        _old_class_shrunk_distance(Xq, yq, X, y, 0.5, 1e-4))


def test_mixture_pooled_covariance_is_law_of_total_covariance():
    X, y = _toy(n=400)
    comps = {c: (X[y == c].mean(0), np.cov(X[y == c], rowvar=False, ddof=0),
                 (y == c).mean()) for c in np.unique(y)}
    mom = ClassMoments.from_mixture(comps)
    np.testing.assert_allclose(mom.pooled_cov, np.cov(X, rowvar=False, ddof=0), atol=1e-10)
    np.testing.assert_allclose(mom.pooled_mean, X.mean(0), atol=1e-12)


def test_white_box_needs_the_papers_form():
    with pytest.raises(ValueError):
        MahalaMIA(white_box=True)                         # no class_shrink
    with pytest.raises(ValueError):
        MahalaMIA(white_box=True, class_shrink=0.5, n_components=50)
    assert MahalaMIA(white_box=True, class_shrink=0.5, covariance="ridge",
                     ridge_alpha=1e-4).tag().endswith("_whitebox100000")


# ── MVN as a white-box target ────────────────────────────────────────────────

def test_mvn_save_load_and_exact_moments(tmp_path):
    X, y = _toy(n=200)
    g = MVNGenerator(seed=3, noise_level=0.7).fit(X, y, 5)
    g.save(tmp_path / "generator.pt")
    h = MVNGenerator(seed=3, noise_level=0.7).load(tmp_path / "generator.pt")
    for c, (m, S, w) in g.moments().items():
        m2, S2, w2 = h.moments()[c]
        np.testing.assert_array_equal(m, m2)
        np.testing.assert_array_equal(S, S2)
        assert w == w2
    np.testing.assert_array_equal(g.sample(50)[0], h.sample(50)[0])
    # the release covariance is (1 + noise_level) times the class covariance
    m, S, _ = g.moments()[0]
    np.testing.assert_allclose(S, 1.7 * g._covs[0])
    with pytest.raises(ValueError):
        MVNGenerator(noise_level=0.3).load(tmp_path / "generator.pt")


def test_model_moments_prefers_exact():
    X, y = _toy(n=200)
    g = MVNGenerator(noise_level=0.7).fit(X, y, 5)
    mom, how = model_moments(g, 10, seed=0)
    assert how == "exact"
    np.testing.assert_array_equal(mom.means[1], g._means[1])


# ── MeLoMIA ──────────────────────────────────────────────────────────────────

def test_conditional_models_are_read_under_each_records_class():
    assert (_record_labels(None, False, None, 4) == 0).all()
    assert (_record_labels(None, False, np.array([3, 1, 2, 0]), 4) == 0).all()
    np.testing.assert_array_equal(_record_labels(None, True, [3, 1, 2, 0], 4), [3, 1, 2, 0])
    with pytest.raises(ValueError):
        _record_labels(None, True, None, 4)
    with pytest.raises(ValueError):
        _record_labels(None, True, [0, -1, 0, 0], 4)


def test_white_box_recipe_guard():
    a = MeLoMIACVAE(white_box=True, n_shadows=5, device="cpu", verbose=False)
    be = a._backend("BRCA")
    recipe = {"preprocess": "standard", "optimizer": "adam", "scheduler": "none",
              "lr": 0.001, "batch_size": 64, "num_iters": 10000, "z_dim": 128,
              "beta": 0.001, "condition_type": "embedding", "early_stopping": False}
    a._check_white_box_recipe(be, "cvae", {"params": recipe})
    with pytest.raises(ValueError, match="num_iters"):
        a._check_white_box_recipe(be, "cvae", {"params": {**recipe, "num_iters": 5000}})
    with pytest.raises(ValueError, match="reads cvae"):
        a._check_white_box_recipe(be, "mvn", {"params": {"noise_level": 0.7}})


# ── MAMA-MIA ─────────────────────────────────────────────────────────────────

def test_given_probabilities_replace_release_counts():
    rng = np.random.default_rng(0)
    real, syn = rng.integers(0, 4, (50, 3)), rng.integers(0, 4, (40, 3))
    ps = _table_probs(syn, 3, 4, 0.5)
    np.testing.assert_array_equal(_log_ratio(real, syn, real, 3, 4, 0.5),
                                  _log_ratio(real, None, real, 3, 4, 0.5, ps=ps))


def test_model_marginals_need_the_targets_own_tables_and_bins():
    with pytest.raises(ValueError):
        MAMAMIAv2(cliques="public", edges="known", synth="model")
    with pytest.raises(ValueError):
        MAMAMIAv2(cliques="true", edges="known", synth="sample")
    assert MAMAMIAv2(cliques="true", edges="known").tag() == "cl-true_ed-known_cc"


# ── integration: built targets ───────────────────────────────────────────────

def _built(dataset, generator, split=1):
    return (paths.target_dir(dataset, generator, split) / "generator.pt").exists()


@pytest.mark.skipif(not _built("BRCA", PGM), reason="corrected DP-PGM target not built")
def test_model_marginals_line_up_with_the_release():
    """Indexing check: the fitted model's (gene, class) marginals and the
    release's frequencies of the same cells must agree closely -- the release
    is 871 draws from that model.  A transposed or relabelled table would not."""
    from mia import datasets as D
    from mia import targets as T
    from mia.attacks.mamamia_v2 import ModelMarginals
    mm = ModelMarginals("BRCA", PGM, 1)
    tg = T.load_target("BRCA", PGM, 1)
    K, ncl = 16, D.n_classes("BRCA")
    genes = np.arange(0, 978, 97)
    Bs = mm.bins(tg["X"].astype(np.float64), 978)[:, genes]
    codes = Bs * ncl + np.asarray(tg["y_int"])[:, None]
    emp = _table_probs(codes, len(genes), K * ncl, 0.0)
    model = mm.gene_label(genes)
    assert np.corrcoef(emp.ravel(), model.ravel())[0, 1] > 0.9
    # class marginals: the model's (noisy) class mix against the release's
    np.testing.assert_allclose(model.reshape(len(genes), K, ncl).sum(1).mean(0),
                               np.bincount(tg["y_int"], minlength=ncl) / len(tg["y_int"]),
                               atol=0.03)


@pytest.mark.skipif(not _built("BRCA", "mvn"), reason="MVN models not materialised")
def test_materialised_mvn_model_is_the_target():
    from scripts.whitebox.materialise_mvn_models import span_residual
    from mia import datasets as D
    from mia import targets as T
    tg = T.load_target("BRCA", "mvn", 1)
    X, y, _ = D.training_subset("BRCA", 1)
    assert span_residual(X, y, tg["X"], tg["y_int"]) < 1e-4
    X2, y2, _ = D.training_subset("BRCA", 2)
    assert span_residual(X2, y2, tg["X"], tg["y_int"]) > 0.1
    meta = json.loads((paths.target_dir("BRCA", "mvn", 1) / "meta.json").read_text())
    g = T.load_model("BRCA", "mvn", 1, device="cpu")
    assert g.noise_level == meta["params"]["noise_level"]
