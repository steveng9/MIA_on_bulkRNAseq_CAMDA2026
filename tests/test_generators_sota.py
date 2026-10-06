"""Guards for the preprocessing chains, the cross-environment bridge and the
manifest-described cohorts.

Run with:  python -m pytest tests/test_generators_sota.py -q
"""

import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import QuantileTransformer, StandardScaler

from mia import datasets as D
from mia import generators as G
from mia import paths
from mia import preprocessing as pp
from mia import targets as T

RNG = np.random.default_rng(0)
X = (RNG.normal(10, 2, size=(120, 30)) + RNG.normal(0, 1, size=(120, 1))).astype(np.float32)
Y = RNG.integers(0, 3, size=120)


# ── Preprocessing ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("kind,cls", [("standard", StandardScaler),
                                      ("quantile", QuantileTransformer)])
def test_legacy_names_are_the_bare_sklearn_objects(kind, cls):
    """Existing targets and pickled scalers depend on this being unchanged."""
    scaler, Xt = pp.fit_scaler(kind, X)
    assert type(scaler) is cls
    ref = cls(output_distribution="normal") if kind == "quantile" else cls()
    assert np.array_equal(Xt, ref.fit_transform(X.astype(np.float64)).astype(np.float32))


@pytest.mark.parametrize("spec", ["standard", "quantile", "quantile_n30", "robust",
                                  "minmax:-1:1", "classcenter+standard", "log1p+standard"])
def test_lossless_chains_round_trip(spec):
    assert pp.describe(spec)["lossless"]
    scaler, Xt = pp.fit_scaler(spec, X, Y)
    assert np.abs(pp.invert_scaler(scaler, Xt, Y) - X).max() < 1e-2


def test_lossy_and_private_flags():
    assert not pp.describe("standard+pca:8")["lossless"]
    assert not pp.describe("clip:0.01:0.99+standard")["lossless"]
    assert pp.describe("fixed:0:24")["data_independent"]
    assert not pp.describe("standard")["data_independent"]
    assert pp.describe("classcenter")["uses_labels"]


def test_pca_output_is_rank_k_and_pickles():
    scaler, Xt = pp.fit_scaler("standard+pca:8", X)
    assert Xt.shape == (120, 8)
    back = pp.invert_scaler(pickle.loads(pickle.dumps(scaler)), Xt)
    assert np.linalg.matrix_rank(back - back.mean(0), tol=1e-3) == 8


def test_spec_survives_a_target_name():
    for spec in ["standard+pca:64", "clip:0.001:0.999+quantile", "fixed:0:24"]:
        name = T.variant_name("tabsyn", "BRCA", {"preprocess": spec})
        assert T.split_name(name) == ("tabsyn", {"preprocess": spec})
    assert T.variant_name("tabsyn", "BRCA", {"preprocess": "quantile_n30"}) == "tabsyn"


def test_unknown_step_raises():
    with pytest.raises(ValueError):
        pp.describe("standrad")


# ── Generators ───────────────────────────────────────────────────────────────

def test_new_generators_are_registered():
    assert {"tabsyn", "tabpfn", "dpsynth", "dpcvae"} <= set(G.REGISTRY)


def test_tabsyn_fits_and_samples():
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = G.build("tabsyn", vae_epochs=2, diff_epochs=5, device=dev, verbose=False)
    Xs, ys = g.fit(X, Y, 3).sample(17)
    assert Xs.shape == (17, 30) and ys.shape == (17,) and set(ys) <= {0, 1, 2}
    assert np.isfinite(Xs).all()


needs_sota = pytest.mark.skipif(not paths.ENV_PYTHON["sota"].exists(),
                                reason="camda_sota environment not installed")


@needs_sota
def test_bridge_runs_a_generator_in_the_other_environment():
    g = G.build("dpsynth", mechanism="independent", epsilon=5.0, numerical_bins=4,
                pgm_iters=50, verbose=False)
    Xs, ys = g.fit(X[:, :4], Y, 3).sample(25)
    assert Xs.shape == (25, 4) and ys.shape == (25,)
    assert g.resolved_params()["mechanism"] == "independent"
    assert g.report()["dp_end_to_end"] is True
    g.close()


@needs_sota
def test_bridge_reports_worker_errors():
    g = G.build("dpsynth", mechanism="no-such-mechanism", verbose=False)
    with pytest.raises(RuntimeError, match="no-such-mechanism"):
        g.fit(X[:, :4], Y, 3)
    g.close()


# ── Manifest cohorts ─────────────────────────────────────────────────────────

def test_manifest_cohort(tmp_path, monkeypatch):
    genes = [f"G{i}" for i in range(12)]
    ids = [f"S{i}" for i in range(60)]
    counts = pd.DataFrame(RNG.poisson(50, size=(12, 60)), index=genes, columns=ids)
    counts.to_csv(tmp_path / "counts.tsv", sep="\t")
    pd.DataFrame({"tissue": ["a", "b", "c"] * 20}, index=ids).to_csv(tmp_path / "meta.csv")
    (tmp_path / "datasets").mkdir()
    (tmp_path / "datasets" / "TOY.yaml").write_text(
        f"expression: {tmp_path}/counts.tsv\norientation: genes_x_samples\n"
        f"labels: {tmp_path}/meta.csv\nlabel_col: tissue\nprepare: cpm+log1p+hvg:8\n")
    monkeypatch.setattr(paths, "CONFIGS", tmp_path)
    monkeypatch.setattr(paths, "SPLITS_DIR", tmp_path / "splits")
    for f in (D.load_expression, D.load_subtypes, D.load_target_splits, D.class_names):
        f.cache_clear()
    try:
        assert D.load_expression("TOY").shape == (60, 8)
        assert D.n_classes("TOY") == 3
        splits = D.load_target_splits("TOY")
        assert len(splits) == 5 and len(splits[1]["nonmember_ids"]) == 12
        Xtr, ytr, _ = D.training_subset("TOY", 1)
        assert Xtr.shape == (48, 8) and D.membership_labels("TOY", 1).sum() == 48
        # drawn once, then fixed
        D.load_target_splits.cache_clear()
        assert D.load_target_splits("TOY") == splits
    finally:
        for f in (D.load_expression, D.load_subtypes, D.load_target_splits, D.class_names):
            f.cache_clear()
