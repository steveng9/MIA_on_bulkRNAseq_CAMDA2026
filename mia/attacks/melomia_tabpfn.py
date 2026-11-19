"""MeLoMIA-TabPFN: TabPFN's own conditional densities as the probe, TabPFN
releases as the shadows.  Designed and first run by Daniil Filienko
(2026-11); this is the in-repo port.

The TabPFN generator has no trained weights to probe: it samples gene c from
TabPFN's predictive distribution p(x_c | label, x_<c), with the training table
as the prompt.  Its release is therefore, column by column, a sample from a
density whose context *was* the training set.  The probe re-reads that density
with the release as the prompt instead, and compares it with the density the
whole candidate pool gives each record:

    term_c(i) = log p(x_ic | earlier columns; context = release)
              - log p(x_ic | earlier columns; context = pool, cross-fitted)

The pool term is cross-fitted in `pool_folds` folds stratified by subtype, so a
record never sits in its own context.  The pool is the same for every release,
so the folds are drawn on candidates sorted by sample ID and the pool term is
computed once per cohort and reused.

Scoring a release (`calibration`-free score, "step 1" below):

  1. per gene, z-score term_c across candidates, then sum over the genes used;
  2. typicality correction (`typicality=True`): regress that sum on the record's
     total pool log-density (sum of its pool terms over the same genes) by OLS
     and keep the residual.  Some records look believable to any model; the
     release and pool terms react to that on different scales, so subtracting
     with weight one leaves a shared nuisance.

Shadow calibration (`n_shadows > 0`, LiRA, Carlini et al. 2022): shadow
contexts are the pool minus one fold, over ceil(n_shadows / shadow_folds)
rounds of a `shadow_folds`-fold subtype-stratified split, so with 30 shadows
every candidate is in 24 and out of 6.  Each shadow release comes from the real
TabPFN generator at the target's settings (default ensemble, 3 orderings,
t = 1), with its own seed, truncated to the first `n_genes` genes -- the
generator fills genes in order, so that is exactly the start of a full release.
Every shadow goes through the same step-1 score with the shared pool term.  Per
candidate, Gaussians are fitted to its in and out shadow scores, with one
variance per side pooled over candidates, and the final score is the target
score's log-likelihood ratio, in against out.  Shadows depend only on the
cohort, so one stack serves all five splits.  (Daniil found a MeLoMIA-style
meta-classifier on the calibrated score no better than the ratio, 0.957 vs
0.954, so the ratio is kept.)

Gene sets: `n_genes` is the prefix of genes used (None = all); `gene_stride`
keeps every k-th gene in it.  Shadow calibration needs a prefix (stride 1).
The TabPFN regressor runs with `n_estimators=2` rather than its default, about
3x faster for about 0.02 AUC.

Daniil's BRCA numbers, for orientation: every 10th gene, no correction 0.741;
+ typicality 0.774; all 978 genes 0.853; first 200 genes 0.833 -> with 30
shadows 0.954 (TPR@1%FPR 0.28 -> 0.63).

Compute runs in the `sota` environment (`mia/pfn_density.py`); this module
only plans units, waits for them and does the arithmetic.  Units are claimed
with lock files, so a second process on another GPU can share the cache:

    CUDA_VISIBLE_DEVICES=1 python scripts/run_experiment.py \
        configs/experiments/melomia_tabpfn_brca.yaml --prepare-only

Cache layout under artifacts/attacks/melomia_tabpfn/<cohort>/e<n_estimators>/:
    candidates.npz                     every record, expression order, true labels
    pool/fold<k>_{ctx,query}.npz       cross-fit contexts and queries
    pool/fold<k>_g<a>-<b>.npz          pool terms for genes [a, b)
    shadows_f<F>_s<seed>/shadow<i>_ctx.npz, shadow<i>_G<G>.npz, shadow<i>_G<G>/
    releases/<generator>_s<split>_<fingerprint>/
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.model_selection import StratifiedKFold

from .. import datasets as D
from .. import paths
from .. import targets as T
from .base import Attack, register


def _save(path: Path, **arrays) -> None:
    tmp = path.with_name(path.name + f".tmp{os.getpid()}.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def step1_scores(rel: np.ndarray, pool: np.ndarray, typicality: bool = True) -> np.ndarray:
    """Summed per-gene z-scored (release - pool) terms, optionally with the
    typicality residual.  `rel`, `pool`: (n_candidates, n_genes_used)."""
    d = np.nan_to_num(rel - pool, nan=0.0, posinf=0.0, neginf=0.0)
    sd = d.std(0)
    z = (d - d.mean(0)) / np.where(sd > 0, sd, 1.0)
    s = z.sum(1)
    if typicality:
        tot = np.nan_to_num(pool).sum(1)
        A = np.column_stack([np.ones_like(tot), tot])
        beta, *_ = np.linalg.lstsq(A, s, rcond=None)
        s = s - A @ beta
    return s


def lira(target: np.ndarray, shadow: np.ndarray, member: np.ndarray) -> np.ndarray:
    """Per-candidate Gaussian LiRA with variances pooled over candidates.
    target (n,), shadow (n, K), member (n, K) bool."""
    m = member.astype(bool)
    if (m.sum(1) < 2).any() or ((~m).sum(1) < 2).any():
        raise ValueError("every candidate needs at least two in and two out shadows "
                         "(n_shadows >= 2 * shadow_folds)")
    mu_in = np.where(m, shadow, 0).sum(1) / m.sum(1)
    mu_out = np.where(~m, shadow, 0).sum(1) / (~m).sum(1)
    var_in = (np.where(m, shadow - mu_in[:, None], 0) ** 2).sum() / m.sum()
    var_out = (np.where(~m, shadow - mu_out[:, None], 0) ** 2).sum() / (~m).sum()
    ll = lambda x, mu, v: -0.5 * (x - mu) ** 2 / v - 0.5 * np.log(v)  # noqa: E731
    return ll(target, mu_in, var_in) - ll(target, mu_out, var_out)


@dataclass
class MeLoMIATabPFN(Attack):
    n_genes: Optional[int] = 200          # gene prefix used; None = all genes
    gene_stride: int = 1                  # every k-th gene of the prefix
    typicality: bool = True
    n_shadows: int = 30                   # 0 = no shadow calibration
    shadow_folds: int = 5
    pool_folds: int = 5
    n_estimators: int = 2                 # TabPFN regressor ensemble in the probe
    chunk: int = 50                       # genes per work unit (bookkeeping only)
    n_workers: int = 1                    # sota-env workers this process starts

    name = "melomia_tabpfn"

    # ── Bookkeeping ─────────────────────────────────────────────────────────

    def params(self) -> dict:
        d = super().params()
        d.pop("chunk", None)
        d.pop("n_workers", None)
        d.pop("device", None)
        return d

    def tag(self) -> str:
        t = f"g{self.n_genes or 'all'}"
        if self.gene_stride > 1:
            t += f"_st{self.gene_stride}"
        if self.typicality:
            t += "_typ"
        if self.n_shadows:
            t += f"_k{self.n_shadows}"
        return t + f"_e{self.n_estimators}"

    def _root(self, dataset: str) -> Path:
        return paths.attack_cache(self.name, dataset, f"e{self.n_estimators}")

    def _n_total(self, dataset: str) -> int:
        return len(D.gene_names(dataset))

    def _G(self, dataset: str) -> int:
        n = self._n_total(dataset)
        return n if self.n_genes is None else min(int(self.n_genes), n)

    def _genes(self, dataset: str) -> np.ndarray:
        return np.arange(0, self._G(dataset), self.gene_stride)

    def _chunks(self, G: int) -> list:
        return [(a, min(a + self.chunk, G)) for a in range(0, G, self.chunk)]

    # ── Inputs ──────────────────────────────────────────────────────────────

    def _candidates(self, dataset: str) -> dict:
        """All records, expression order, encoded true labels."""
        root = self._root(dataset)
        paths.ensure(root)
        f = root / "candidates.npz"
        expr = D.load_expression(dataset)
        X = expr.values.astype(np.float32)
        y = D.encode_subtypes(dataset, D.load_subtypes(dataset).values)
        if not f.exists():
            _save(f, X=X, y=y)
        return {"X": X, "y": y, "ids": np.asarray(expr.index), "path": f}

    def _folds(self, ids: np.ndarray, y: np.ndarray, k: int, seed: int) -> np.ndarray:
        """Fold of each record (expression order), drawn on records sorted by
        sample ID so it does not depend on row order or on the target split."""
        order = np.argsort(ids.astype(str), kind="stable")
        fold = np.empty(len(ids), dtype=np.int64)
        skf = StratifiedKFold(k, shuffle=True, random_state=seed)
        for j, (_, te) in enumerate(skf.split(np.zeros(len(ids)), y[order])):
            fold[order[te]] = j
        return fold

    def _pool_fold(self, dataset: str, c: dict) -> np.ndarray:
        return self._folds(c["ids"], c["y"], self.pool_folds, self.seed)

    def _shadow_membership(self, dataset: str, c: dict) -> np.ndarray:
        """(n, n_shadows) bool: candidate i in shadow j's context."""
        F = self.shadow_folds
        rounds = math.ceil(self.n_shadows / F)
        cols = []
        for r in range(rounds):
            fold = self._folds(c["ids"], c["y"], F, self.seed + 1000 + r)
            cols += [fold != k for k in range(F)]
        return np.column_stack(cols[: self.n_shadows])

    def _shadow_dir(self, dataset: str) -> Path:
        return self._root(dataset) / f"shadows_f{self.shadow_folds}_s{self.seed}"

    # ── Work units ──────────────────────────────────────────────────────────

    def _terms_unit(self, ctx: Path, query: Path, out: Path, g0: int, g1: int) -> dict:
        return {"kind": "terms", "ctx": str(ctx), "query": str(query), "out": str(out),
                "g0": int(g0), "g1": int(g1), "n_estimators": self.n_estimators,
                "seed": int(self.seed), "device": "cuda"}

    def _pool_units(self, dataset: str, c: dict, G: int) -> list:
        d = self._root(dataset) / "pool"
        paths.ensure(d)
        fold = self._pool_fold(dataset, c)
        units = []
        for k in range(self.pool_folds):
            ctx, q = d / f"fold{k}_ctx.npz", d / f"fold{k}_query.npz"
            if not q.exists():
                _save(ctx, X=c["X"][fold != k], y=c["y"][fold != k])
                _save(q, X=c["X"][fold == k], y=c["y"][fold == k])
            units += [self._terms_unit(ctx, q, d / f"fold{k}_g{a}-{b}.npz", a, b)
                      for a, b in self._chunks(G)]
        return units

    def _shadow_units(self, dataset: str, c: dict, G: int) -> list:
        d = self._shadow_dir(dataset)
        paths.ensure(d)
        M = self._shadow_membership(dataset, c)
        units = []
        for i in range(self.n_shadows):
            ctx = d / f"shadow{i:02d}_ctx.npz"
            if not ctx.exists():
                _save(ctx, X=c["X"][M[:, i]], y=c["y"][M[:, i]])
            rel = d / f"shadow{i:02d}_G{G}.npz"
            units.append({"kind": "shadow", "ctx": str(ctx), "out": str(rel),
                          "n_genes": G, "n": int(M[:, i].sum()),
                          "n_classes": D.n_classes(dataset),
                          "seed": int(zlib.crc32(f"{dataset}/shadow{i}/{self.seed}".encode())
                                      % 2**31),
                          "device": "cuda"})
            td = d / f"shadow{i:02d}_G{G}"
            paths.ensure(td)
            units += [self._terms_unit(rel, c["path"], td / f"g{a}-{b}.npz", a, b)
                      for a, b in self._chunks(G)]
        return units

    def _release_dir(self, dataset: str, generator: str, split: int) -> Path:
        fp = T.fingerprint(dataset, generator, split)[:12]
        safe = generator.replace("/", "_").replace("@", "+")
        return self._root(dataset) / "releases" / f"{safe}_s{split}_{fp}"

    def _release_units(self, dataset: str, generator: str, split: int, c: dict,
                       G: int) -> list:
        d = self._release_dir(dataset, generator, split)
        paths.ensure(d)
        ctx = d / "release.npz"
        if not ctx.exists():
            tg = T.load_target(dataset, generator, split)
            y = tg["y_int"]
            if (y < 0).any():
                raise ValueError(f"{generator}/split_{split} releases labels outside "
                                 "the cohort's vocabulary")
            _save(ctx, X=tg["X"], y=y)
        return [self._terms_unit(ctx, c["path"], d / f"g{a}-{b}.npz", a, b)
                for a, b in self._chunks(G)]

    def _run(self, units: list, dataset: str, what: str) -> None:
        """Run units in sota-env workers until every output exists, waiting
        for units other processes have claimed."""
        jobs = self._root(dataset) / "jobs"
        paths.ensure(jobs)
        python = paths.ENV_PYTHON["sota"]
        waited = 0
        while True:
            todo = [u for u in units if not Path(u["out"]).exists()]
            if not todo:
                return
            free = [u for u in todo
                    if not Path(u["out"] + ".lock").exists()
                    and (u["kind"] != "terms" or Path(u["ctx"]).exists())]
            if free:
                if self.verbose:
                    print(f"[melomia_tabpfn] {what}: {len(free)} of {len(units)} "
                          f"units to run", flush=True)
                job = jobs / f"{what}_{os.getpid()}_{int(time.time())}.json"
                job.write_text(json.dumps({"units": todo}))
                procs = [subprocess.Popen(
                    [str(python), "-W", "ignore", "-m", "mia.pfn_density", str(job)],
                    cwd=str(paths.PROJECT_ROOT)) for _ in range(max(1, self.n_workers))]
                codes = [p.wait() for p in procs]
                job.unlink()
                if any(codes):
                    raise RuntimeError(f"pfn_density worker failed ({codes}) on {what}")
                continue
            if waited % 600 == 0 and self.verbose:
                print(f"[melomia_tabpfn] {what}: waiting on {len(todo)} units claimed "
                      f"by other processes, e.g. {todo[0]['out']}.lock", flush=True)
            time.sleep(30)
            waited += 30

    # ── Assembling term matrices ────────────────────────────────────────────

    def _load_chunks(self, d: Path, G: int, stem: str = "") -> np.ndarray:
        return np.hstack([np.load(d / f"{stem}g{a}-{b}.npz")["logdens"]
                          for a, b in self._chunks(G)])

    def _pool_terms(self, dataset: str, c: dict, G: int) -> np.ndarray:
        d = self._root(dataset) / "pool"
        fold = self._pool_fold(dataset, c)
        P = np.empty((len(c["X"]), G), dtype=np.float32)
        for k in range(self.pool_folds):
            P[fold == k] = self._load_chunks(d, G, f"fold{k}_")
        return P

    def _shadow_scores(self, dataset: str, c: dict, G: int, genes, pool) -> np.ndarray:
        d = self._shadow_dir(dataset)
        return np.column_stack([
            step1_scores(self._load_chunks(d / f"shadow{i:02d}_G{G}", G)[:, genes],
                         pool[:, genes], self.typicality)
            for i in range(self.n_shadows)])

    # ── Interface ───────────────────────────────────────────────────────────

    def prepare(self, dataset: str, **_) -> None:
        if self.n_shadows and self.gene_stride != 1:
            raise ValueError("shadow calibration needs a gene prefix (gene_stride=1)")
        if self.n_shadows and self.n_shadows < 2 * self.shadow_folds:
            raise ValueError("n_shadows < 2 * shadow_folds leaves candidates out of "
                             "fewer than two shadows")
        c = self._candidates(dataset)
        G = self._G(dataset)
        self._run(self._pool_units(dataset, c, G), dataset, "pool")
        if self.n_shadows:
            self._run(self._shadow_units(dataset, c, G), dataset, "shadows")

    def score(self, dataset: str, generator: str, split: int) -> np.ndarray:
        c = self._candidates(dataset)
        G, genes = self._G(dataset), self._genes(dataset)
        self._run(self._pool_units(dataset, c, G), dataset, "pool")
        self._run(self._release_units(dataset, generator, split, c, G), dataset,
                  f"release_{split}")
        pool = self._pool_terms(dataset, c, G)
        rel = self._load_chunks(self._release_dir(dataset, generator, split), G)
        s = step1_scores(rel[:, genes], pool[:, genes], self.typicality)
        if not self.n_shadows:
            return s
        self._run(self._shadow_units(dataset, c, G), dataset, "shadows")
        S = self._shadow_scores(dataset, c, G, genes, pool)
        return lira(s, S, self._shadow_membership(dataset, c))


register(MeLoMIATabPFN)
