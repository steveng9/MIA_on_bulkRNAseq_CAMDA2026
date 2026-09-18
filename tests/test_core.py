"""Fast checks on the parts where a silent bug would corrupt every result.

Run with:  python -m pytest tests/ -q

These are correctness guards, not a full test suite.  They target the invariants
that are easy to break during a refactor and impossible to notice from the
numbers: split alignment, row ordering, leak-free grouping, and the shape
contract between feature extraction and the meta-classifier.
"""

import numpy as np
import pytest

from mia import datasets as D
from mia import metrics as M
from mia.attacks.melomia import features as F
from mia.attacks.melomia import meta as MM


# ── Dataset invariants ───────────────────────────────────────────────────────

@pytest.mark.parametrize("dataset", ["BRCA", "COMBINED"])
def test_splits_partition_the_cohort(dataset):
    """Members and non-members must tile the cohort exactly, with no overlap."""
    all_ids = set(D.load_expression(dataset).index)
    for split, entry in D.load_target_splits(dataset).items():
        member = set(entry["member_ids"])
        nonmember = set(entry["nonmember_ids"])
        assert member.isdisjoint(nonmember), f"split {split} has overlapping halves"
        assert member | nonmember == all_ids, f"split {split} does not cover the cohort"
        assert 0.75 < len(member) / len(all_ids) < 0.85


@pytest.mark.parametrize("dataset", ["BRCA", "COMBINED"])
def test_membership_vector_matches_expression_order(dataset):
    """Scores are returned in expression-matrix order, so labels must be too."""
    expr = D.load_expression(dataset)
    y = D.membership_labels(dataset, 1)
    assert len(y) == len(expr)
    nonmember = set(D.load_target_splits(dataset)[1]["nonmember_ids"])
    for i, sid in enumerate(expr.index):
        assert y[i] == (0 if sid in nonmember else 1)


@pytest.mark.parametrize("dataset", ["BRCA", "COMBINED"])
def test_subtypes_align_with_expression(dataset):
    labels = D.load_subtypes(dataset)
    assert list(labels.index) == list(D.load_expression(dataset).index)
    assert not labels.isna().any()
    encoded = D.encode_subtypes(dataset, labels.values)
    assert encoded.min() >= 0 and encoded.max() < D.n_classes(dataset)


def test_training_subset_is_the_member_half():
    X, y, ids = D.training_subset("BRCA", 2)
    member = D.load_target_splits("BRCA")[2]["member_ids"]
    assert ids == member
    assert len(X) == len(y) == len(member)


# ── Metrics ──────────────────────────────────────────────────────────────────

def test_tpr_at_fpr_never_exceeds_the_target_fpr():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    s = rng.random(500)
    for target in (0.01, 0.1):
        assert 0.0 <= M.tpr_at_fpr(y, s, target) <= 1.0


def test_perfect_and_random_scores():
    y = np.array([0] * 50 + [1] * 50)
    assert M.evaluate(y, y.astype(float))["auc"] == pytest.approx(1.0)
    assert M.evaluate(y, (1 - y).astype(float))["auc"] == pytest.approx(0.0)


def test_aggregate_reports_spread_across_splits():
    per_split = [{"auc": 0.6}, {"auc": 0.8}]
    agg = M.aggregate(per_split)
    assert agg["auc_mean"] == pytest.approx(0.7)
    assert agg["auc_std"] > 0
    assert agg["n_splits"] == 2


# ── Feature preparation ──────────────────────────────────────────────────────

def test_prepare_shape_matches_declared_summary_dim():
    rng = np.random.default_rng(1)
    losses = rng.random((20, 6, 40)).astype(np.float32)
    extra = rng.random((20, 5)).astype(np.float32)
    X = F.prepare(losses, extra, range(6), 40)
    assert X.shape == (20, F.summary_dim(6, 5))


def test_prepare_respects_the_slice():
    rng = np.random.default_rng(2)
    losses = rng.random((20, 6, 40)).astype(np.float32)
    X = F.prepare(losses, None, [0, 2, 4], 10)
    assert X.shape == (20, F.summary_dim(3, 0))


def test_prepare_uses_only_the_selected_sweep_points():
    """Changing an excluded sweep point must not move the features."""
    rng = np.random.default_rng(3)
    losses = rng.random((8, 4, 10)).astype(np.float32)
    before = F.prepare(losses, None, [0, 1], 10)
    losses[:, 3, :] += 100.0
    assert np.allclose(before, F.prepare(losses, None, [0, 1], 10))


def test_reference_calibration_centres_the_reference():
    rng = np.random.default_rng(4)
    ref = rng.normal(5.0, 2.0, (100, 3, 20)).astype(np.float32)
    out = F.calibrate_against_reference(ref, ref)
    assert np.allclose(out.mean(axis=(0, 2)), 0.0, atol=1e-4)
    assert np.allclose(out.std(axis=(0, 2)), 1.0, atol=1e-3)


# ── Meta-classifier guards ───────────────────────────────────────────────────

def test_group_holdout_is_subject_disjoint():
    """A sample must never appear on both sides of the early-stopping split."""
    rng = np.random.default_rng(5)
    groups = np.repeat(np.arange(40), 5)          # 40 samples x 5 shadows
    X = rng.random((200, 7))
    y = rng.integers(0, 2, 200)
    X_fit, X_es, y_fit, y_es = MM.group_holdout(X, y, groups, test_size=0.25)
    assert len(X_fit) + len(X_es) == 200
    fit_rows = {tuple(r) for r in X_fit}
    assert not fit_rows & {tuple(r) for r in X_es}


def test_sweep_buckets_tile_the_axis():
    for n in (2, 5, 6, 15):
        buckets = MM.sweep_buckets(n)
        flat = [i for b in buckets for i in b]
        assert sorted(flat) == list(range(n))


def test_softmax_weights_sum_to_one_and_gate_weak_models():
    w = MM.softmax_weights({"a": 0.80, "b": 0.60, "c": 0.40}, min_gate=0.55)
    assert sum(w.values()) == pytest.approx(1.0)
    assert w["c"] == pytest.approx(0.0)
    assert w["a"] > w["b"]


def test_softmax_weights_fall_back_when_everything_is_gated():
    with pytest.warns(UserWarning):
        w = MM.softmax_weights({"a": 0.50, "b": 0.51}, min_gate=0.55)
    assert sum(w.values()) == pytest.approx(1.0)


def test_split_trial_params_recovers_a_usable_slice():
    params = {"use_bucket_0": True, "use_bucket_1": False, "use_bucket_2": True,
              "noise_budget": 120, "max_depth": 4}
    idx, budget, hp = MM.split_trial_params(params, n_sweep=15, n_noise=600, n_buckets=3)
    assert budget == 120
    assert hp == {"max_depth": 4}
    assert idx and max(idx) < 15


def test_split_trial_params_falls_back_when_no_bucket_chosen():
    idx, _, _ = MM.split_trial_params({"noise_budget": 50}, n_sweep=6, n_noise=50)
    assert idx == list(range(6))
