"""The zoo's guarantees: stable ids, recomputed labels, caught contamination."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mia.zoo import ids as I  # noqa: E402


@pytest.fixture
def zoo(tmp_path, monkeypatch):
    """A registry rooted in a temp dir, so tests never touch the real store."""
    from mia.zoo import registry as Z
    monkeypatch.setattr(Z, "ZOO", tmp_path / "zoo")
    monkeypatch.setattr(Z, "INDEX", tmp_path / "zoo" / "index.jsonl")
    return Z


# ── Identity ─────────────────────────────────────────────────────────────────

def test_digest_is_order_independent():
    assert I.digest({"a": 1, "b": 2}) == I.digest({"b": 2, "a": 1})


def test_digest_separates_different_params():
    assert I.digest({"lr": 1e-3}) != I.digest({"lr": 1e-4})


def test_closure_hash_is_a_set_hash():
    assert I.closure_hash(["s2", "s1"]) == I.closure_hash(["s1", "s2"])
    assert I.closure_hash(["s1"]) != I.closure_hash(["s1", "s2"])


# ── Store ────────────────────────────────────────────────────────────────────

def _fit(zoo, seed=1, closure=("s1", "s2"), params=None, source=None):
    return zoo.get_or_create_fit(
        dataset="T", generator="mvn", params=params or {"noise_level": 0.7},
        source=source or I.real_ref("T", 1), seed=seed, closure_ids=list(closure),
        build=lambda d: (d / "model.pt").write_text("weights"),
    )


def test_fit_is_built_once_and_reused(zoo):
    calls = []

    def build(d):
        calls.append(1)
        (d / "model.pt").write_text("weights")

    kw = dict(dataset="T", generator="mvn", params={"noise_level": 0.7},
              source=I.real_ref("T", 1), seed=1, closure_ids=["s1", "s2"])
    a = zoo.get_or_create_fit(build=build, **kw)
    b = zoo.get_or_create_fit(build=build, **kw)
    assert a.id == b.id
    assert len(calls) == 1, "second request should reuse, not rebuild"


def test_different_seed_is_a_different_artifact(zoo):
    assert _fit(zoo, seed=1).id != _fit(zoo, seed=2).id


def test_membership_labels_are_recomputed_from_the_closure(zoo):
    a = _fit(zoo, closure=("s1", "s3"))
    labels = a.membership(["s1", "s2", "s3", "s4"])
    assert labels.tolist() == [1, 0, 1, 0]


def test_sample_inherits_its_fits_closure(zoo):
    f = _fit(zoo, closure=("s1", "s2"))
    s = zoo.get_or_create_sample(
        f, n=10, seed=7, build=lambda d: np.savez(d / "data.npz", X=np.zeros((10, 3))))
    assert s.closure == f.closure
    assert s.n == 10
    assert f.id in zoo.ancestors(s.id)


def test_ancestors_walk_back_through_synthetic_data(zoo):
    base = _fit(zoo, closure=("s1", "s2"))
    syn = zoo.get_or_create_sample(base, n=10, seed=7,
                                   build=lambda d: (d / "data.npz").write_text("x"))
    probe = zoo.get_or_create_fit(
        dataset="T", generator="mvn", params={"noise_level": 0.7},
        source=syn.ref, seed=3, closure_ids=base.closure_ids(),
        build=lambda d: (d / "model.pt").write_text("w"))
    assert zoo.ancestors(probe.id) == [syn.id, base.id]
    assert "real:T:split1:train" in zoo.lineage(probe.id)


def test_find_filters_by_field(zoo):
    _fit(zoo, seed=1)
    _fit(zoo, seed=2)
    assert len(zoo.find(dataset="T", kind="fit")) == 2
    assert zoo.find(dataset="OTHER") == []


# ── Contamination ────────────────────────────────────────────────────────────

def _assignment(zoo):
    from mia.zoo import roles as R
    target_fit = _fit(zoo, seed=100, closure=("s1", "s2"))
    target = zoo.get_or_create_sample(target_fit, n=10, seed=1,
                                      build=lambda d: (d / "data.npz").write_text("x"))
    base = _fit(zoo, seed=200, closure=("s2", "s3"))
    base_syn = zoo.get_or_create_sample(base, n=10, seed=2,
                                        build=lambda d: (d / "data.npz").write_text("x"))
    synth_shadow = zoo.get_or_create_fit(
        dataset="T", generator="mvn", params={"noise_level": 0.7},
        source=base_syn.ref, seed=300, closure_ids=base.closure_ids(),
        build=lambda d: (d / "model.pt").write_text("w"))
    final_proxy = zoo.get_or_create_fit(
        dataset="T", generator="mvn", params={"noise_level": 0.7},
        source=target.ref, seed=400, closure_ids=target_fit.closure_ids(),
        build=lambda d: (d / "model.pt").write_text("w"))
    a = R.Assignment("exp", "T")
    a.add(R.TARGET, target.id).add(R.BASE_SHADOW, base.id)
    a.add(R.SYNTH_SHADOW, synth_shadow.id).add(R.FINAL_PROXY, final_proxy.id)
    return R, a, dict(target=target, base=base, base_syn=base_syn)


def test_a_well_formed_experiment_is_clean(zoo):
    R, a, _ = _assignment(zoo)
    assert R.check(a) == []


def test_one_artifact_cannot_hold_two_roles(zoo):
    R, a, parts = _assignment(zoo)
    a.add(R.BASE_SHADOW, a.ids(R.TARGET)[0])
    assert any("two roles" in p for p in R.check(a))


def test_shadow_sharing_the_targets_training_set_is_caught(zoo):
    R, a, parts = _assignment(zoo)
    twin = _fit(zoo, seed=999, closure=("s1", "s2"))    # same closure as target
    a.add(R.BASE_SHADOW, twin.id)
    assert any("training set" in p for p in R.check(a))


def test_adversary_model_descending_from_the_target_is_caught(zoo):
    R, a, parts = _assignment(zoo)
    leaky = zoo.get_or_create_fit(
        dataset="T", generator="mvn", params={"noise_level": 0.7},
        source=parts["target"].ref, seed=555,
        closure_ids=["s1", "s2"], build=lambda d: (d / "model.pt").write_text("w"))
    a.add(R.SYNTH_SHADOW, leaky.id)
    assert any("descends from the target" in p for p in R.check(a))


def test_internal_proxy_must_be_held_out(zoo):
    R, a, parts = _assignment(zoo)
    # An internal proxy built on the SAME base shadow a synth-shadow used.
    not_held_out = zoo.get_or_create_fit(
        dataset="T", generator="mvn", params={"noise_level": 0.7},
        source=parts["base_syn"].ref, seed=777,
        closure_ids=parts["base"].closure_ids(),
        build=lambda d: (d / "model.pt").write_text("w"))
    a.add(R.INTERNAL_PROXY, not_held_out.id)
    assert any("not held out" in p for p in R.check(a))


def test_cohort_mismatch_is_caught(zoo):
    R, a, _ = _assignment(zoo)
    other = zoo.get_or_create_fit(
        dataset="OTHER", generator="mvn", params={}, source=I.real_ref("OTHER", 1),
        seed=1, closure_ids=["x1"], build=lambda d: (d / "model.pt").write_text("w"))
    a.add(R.BASE_SHADOW, other.id)
    assert any("not T" in p for p in R.check(a))


def test_assert_clean_raises_with_every_complaint(zoo):
    R, a, _ = _assignment(zoo)
    a.add(R.BASE_SHADOW, a.ids(R.TARGET)[0])
    with pytest.raises(AssertionError, match="contaminated experiment"):
        R.assert_clean(a)
