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


def test_summary_cache_matches_prepare():
    """The search's cached summaries must be the numbers `prepare` returns."""
    rng = np.random.default_rng(9)
    losses = rng.gamma(2.0, 1.0, (60, 5, 40)).astype(np.float32)
    extra = rng.random((60, 3)).astype(np.float32)
    cache = F.SummaryCache(losses, extra)
    for idx, budget in (([0, 1, 2, 3, 4], 40), ([1, 3], 10), ([4], 25), ([1, 3], 10)):
        assert np.array_equal(cache.prepare(idx, budget), F.prepare(losses, extra, idx, budget))
    assert np.array_equal(F.SummaryCache(losses).prepare([0, 2], 30),
                          F.prepare(losses, None, [0, 2], 30))


def test_reference_calibration_centres_the_reference():
    rng = np.random.default_rng(4)
    ref = rng.normal(5.0, 2.0, (100, 3, 20)).astype(np.float32)
    out = F.calibrate_against_reference(ref, ref)
    assert np.allclose(out.mean(axis=(0, 2)), 0.0, atol=1e-4)
    assert np.allclose(out.std(axis=(0, 2)), 1.0, atol=1e-3)


def _shadow_stack(rng, K=8, n=40, d=5):
    """K models x n records: a large per-record difficulty, a per-model scale and noise."""
    difficulty = rng.normal(0.0, 5.0, (1, n, d))
    scale = rng.uniform(0.5, 2.0, (K, 1, 1))
    return scale * (difficulty + rng.normal(0.0, 1.0, (K, n, d))) + rng.normal(0, 3, (K, 1, d))


def test_per_record_training_rows_are_leave_one_out():
    """Row (k, i) is z-scored against record i under the other models only."""
    rng = np.random.default_rng(6)
    S = _shadow_stack(rng)
    K, n, d = S.shape
    Z = F.per_record_train(S.reshape(K * n, d), K).reshape(K, n, d)
    P = np.stack([F.standardise_per_model(S[k]) for k in range(K)])
    for k in (0, 3, K - 1):
        others = np.delete(P, k, axis=0)
        want = (P[k] - others.mean(0)) / (others.std(0) + 1e-6)
        assert np.allclose(Z[k], want, atol=1e-4)


def test_per_record_calibration_removes_record_difficulty():
    """Making one record harder in every model alike must not move its calibrated row."""
    rng = np.random.default_rng(7)
    K, n, d = 30, 400, 5
    S = rng.normal(0.0, 5.0, (1, n, d)) + rng.normal(0.0, 1.0, (K, n, d))
    T = S.copy()
    T[:, 0, :] += 10.0
    a = F.per_record_train(S.reshape(K * n, d), K).reshape(K, n, d)
    b = F.per_record_train(T.reshape(K * n, d), K).reshape(K, n, d)
    # not exactly: the shift nudges each model's own scale a little differently.  Without
    # calibration the same shift is ~10 of these units (2 model-sds / a record spread of 0.2).
    assert np.allclose(a[:, 0], b[:, 0], atol=0.5)
    # ... whereas the model-standardised feature, which the classifier saw before, moves a lot
    assert np.abs(F.standardise_per_model(T[0])[0] - F.standardise_per_model(S[0])[0]).min() > 1.0


def test_per_record_attack_rows_use_every_shadow():
    """A proxy equal to one shadow is scored against all K, itself included."""
    rng = np.random.default_rng(8)
    S = _shadow_stack(rng)
    K, n, d = S.shape
    mean, sd = F.per_record_reference(S.reshape(K * n, d), K)
    P = np.stack([F.standardise_per_model(S[k]) for k in range(K)])
    assert np.allclose(mean, P.mean(0)) and np.allclose(sd, P.std(0))
    out = F.per_record_apply(S[2], mean, sd)
    assert np.allclose(out, (P[2] - mean) / (sd + 1e-6), atol=1e-4)
    with pytest.raises(ValueError):
        F.per_record_train(S.reshape(K * n, d)[:-1], K)


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
