"""Model-disjoint validation: the property the sample-grouped CV cannot give.

A meta-classifier is trained on shadow models and applied to one model it has
never seen (the proxy).  These tests pin the invariant that makes the block
scheme measure that: no validation row may share a sample *or* a shadow with
any training row.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mia.attacks.melomia import meta as M  # noqa: E402


def pool(n_samples=60, n_shadows=8):
    """Row layout matching `_pooled`: every shadow scores every sample."""
    sample_groups = np.tile(np.arange(n_samples), n_shadows)
    shadow_groups = np.repeat(np.arange(n_shadows), n_samples)
    return sample_groups, shadow_groups


def test_folds_are_disjoint_in_samples_and_in_models():
    sg, mg = pool()
    folds = list(M.block_folds(sg, mg, n_folds=4))
    assert folds, "expected at least one usable fold"
    for train, val in folds:
        assert not set(sg[train]) & set(sg[val]), "a sample appears on both sides"
        assert not set(mg[train]) & set(mg[val]), "a shadow model appears on both sides"


def test_sample_grouped_cv_does_not_hold_out_models():
    """The gap this exists to close, stated as a test.

    StratifiedGroupKFold grouped by sample leaves every shadow in both the
    training and the validation side -- which is exactly why its AUC is an
    optimistic estimate of proxy-time performance.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    sg, mg = pool()
    y = (np.arange(len(sg)) % 5 > 0).astype(int)
    gkf = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=0)
    leaked = False
    for train, val in gkf.split(np.zeros((len(sg), 2)), y, groups=sg):
        assert not set(sg[train]) & set(sg[val])          # samples are held out
        if set(mg[train]) & set(mg[val]):                 # models are not
            leaked = True
    assert leaked, "sample-grouped CV unexpectedly held models out too"


def test_every_fold_is_non_empty_on_both_sides():
    sg, mg = pool()
    for train, val in M.block_folds(sg, mg, n_folds=4):
        assert len(train) > 0 and len(val) > 0


def test_fold_count_is_capped_by_the_number_of_shadows():
    sg, mg = pool(n_samples=40, n_shadows=3)
    assert len(list(M.block_folds(sg, mg, n_folds=10))) <= 3


def test_validation_blocks_do_not_overlap_each_other():
    sg, mg = pool()
    seen = set()
    for _, val in M.block_folds(sg, mg, n_folds=4):
        assert not seen & set(val.tolist()), "a row is validated twice"
        seen |= set(val.tolist())


def test_folds_are_deterministic():
    sg, mg = pool()
    a = [(t.tolist(), v.tolist()) for t, v in M.block_folds(sg, mg, 4, seed=7)]
    b = [(t.tolist(), v.tolist()) for t, v in M.block_folds(sg, mg, 4, seed=7)]
    assert a == b


def test_a_different_seed_gives_a_different_partition():
    sg, mg = pool()
    a = [v.tolist() for _, v in M.block_folds(sg, mg, 4, seed=1)]
    b = [v.tolist() for _, v in M.block_folds(sg, mg, 4, seed=2)]
    assert a != b


def test_block_cv_recovers_a_transferable_signal():
    """A signal that is the same in every shadow should survive model holdout."""
    rng = np.random.RandomState(0)
    sg, mg = pool(n_samples=80, n_shadows=8)
    y = rng.binomial(1, 0.5, size=len(sg))
    X = np.c_[y + rng.normal(0, 0.3, len(y)), rng.normal(0, 1, len(y))]
    score = M.block_cv_score("rf", X, y, sg, mg, hparams={"n_estimators": 40},
                             n_folds=4, device="cpu")
    assert score > 0.5, f"transferable signal lost under model holdout ({score:.3f})"


def test_block_cv_rejects_a_model_specific_signal():
    """A feature that encodes labels only in each shadow's own frame of reference.

    This is the failure mode sample-grouped CV cannot see.  Each shadow's losses
    sit in their own range, and within one shadow the feature separates members
    perfectly -- but the threshold that does it is different for every shadow, so
    a model held out entirely gets nothing.  Chance for TPR@10%FPR is 0.10.
    """
    rng = np.random.RandomState(0)
    sg, mg = pool(n_samples=80, n_shadows=8)
    y = rng.binomial(1, 0.5, size=len(sg))
    offset = mg * 10.0                           # each shadow's own range
    X = np.c_[offset + y + rng.normal(0, 0.05, len(y)), rng.normal(0, 1, len(y))]
    score = M.block_cv_score("rf", X, y, sg, mg, hparams={"n_estimators": 40},
                             n_folds=4, device="cpu")
    assert score < 0.25, f"model-specific signal leaked through ({score:.3f})"


def test_evaluate_block_reports_how_much_was_validated():
    rng = np.random.RandomState(0)
    sg, mg = pool(n_samples=60, n_shadows=8)
    y = rng.binomial(1, 0.5, size=len(sg))
    X = np.c_[y + rng.normal(0, 0.3, len(y)), rng.normal(0, 1, len(y))]
    diag = M.evaluate_block("rf", X, y, sg, mg, hparams={"n_estimators": 40},
                            n_folds=4, device="cpu")
    assert 0 < diag["n_val"] < len(y)
    assert diag["auc"] > 0.5
