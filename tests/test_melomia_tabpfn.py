"""Arithmetic of MeLoMIA-TabPFN's scoring (the TabPFN calls need the sota env
and a GPU and are exercised by the experiment itself)."""

import numpy as np

from mia.attacks.melomia_tabpfn import MeLoMIATabPFN, lira, step1_scores
from mia.metrics import evaluate


def test_step1_removes_typicality():
    rng = np.random.default_rng(0)
    n, g = 400, 30
    member = rng.random(n) < 0.5
    typical = rng.normal(size=n)                      # nuisance shared by both terms
    pool = typical[:, None] * 1.0 + rng.normal(size=(n, g))
    rel = typical[:, None] * 2.0 + 0.3 * member[:, None] + rng.normal(size=(n, g))
    raw = step1_scores(rel, pool, typicality=False)
    cor = step1_scores(rel, pool, typicality=True)
    assert abs(np.corrcoef(cor, pool.sum(1))[0, 1]) < 1e-6
    assert evaluate(member.astype(int), cor)["auc"] > evaluate(member.astype(int), raw)["auc"]


def test_lira_separates_and_shuffles_to_chance():
    rng = np.random.default_rng(1)
    n, k = 600, 30
    offset = rng.normal(scale=3, size=n)              # per-record difficulty
    M = rng.permuted(np.tile(np.arange(k) < 24, (n, 1)), axis=1)   # 24 in, 6 out
    S = offset[:, None] + 1.0 * M + rng.normal(scale=0.5, size=(n, k))
    member = rng.random(n) < 0.5
    t = offset + 1.0 * member + rng.normal(scale=0.5, size=n)
    auc = evaluate(member.astype(int), lira(t, S, M))["auc"]
    assert auc > 0.9 and evaluate(member.astype(int), t)["auc"] < auc
    shuffled = lira(t, S, rng.permuted(M, axis=1))
    assert abs(evaluate(member.astype(int), shuffled)["auc"] - 0.5) < 0.1


def test_shadow_membership_balance():
    a = MeLoMIATabPFN(n_shadows=30)
    ids = np.array([f"s{i:04d}" for i in range(200)])
    y = np.arange(200) % 4
    c = {"ids": ids, "y": y}
    M = a._shadow_membership("X", c)
    assert M.shape == (200, 30)
    assert (M.sum(1) == 24).all()
    # Folds depend on sample IDs, not on row order.
    p = np.random.default_rng(0).permutation(200)
    Mp = a._shadow_membership("X", {"ids": ids[p], "y": y[p]})
    assert (Mp == M[p]).all()


def test_tag():
    assert MeLoMIATabPFN().tag() == "g200_typ_k30_e2"
    assert MeLoMIATabPFN(n_genes=None, gene_stride=10, typicality=False,
                         n_shadows=0).tag() == "gall_st10_e2"
