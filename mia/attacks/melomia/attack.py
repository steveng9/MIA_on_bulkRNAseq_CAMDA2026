"""MeLoMIA -- Measurement of Losses MIA, with synth-shadow modelling.

The idea is the loss-trajectory attack of Tartan_Federer (MIDST): train shadow
generators whose membership you control, read a per-sample loss signature out of
each, and train a meta-classifier to recognise what a member's signature looks
like.

What the CAMDA threat model forces us to change is where those shadows come
from.  At inference the adversary has no target model -- only the released
synthetic dataset -- so the only model it can build is a *proxy* trained on that
synthetic data.  Shadow models trained on real data therefore produce loss
distributions from a different domain than the proxy does, and a meta-classifier
fitted to the former transfers poorly to the latter.  Earlier runs in this repo
measured that gap directly: real-data shadows reached TPR@10%FPR of 0.58 when
validated against each other and collapsed to about 0.14 when the features came
from a synthetic-trained proxy.

**Synth-shadow modelling** closes it by inserting a layer:

    real split k  ->  base shadow (target-faithful)  ->  internal synthetic k
                                                             |
                                                             v
                                       synth-shadow k (sharp)  -> loss features

Real data enters only through the base shadows.  The models features are read
from -- the synth-shadows during training, the proxy at inference -- have both
only ever seen synthetic data, so the two distributions match.  A sample's
membership label for shadow k is inherited from the base shadow: x is a member
of shadow k iff x was in split k's training half, even though synth-shadow k
never saw x at all.  That inheritance is the whole trick, and it is why the
signal survives: whatever the base shadow memorised about its members leaves a
trace in the synthetic data it generated, and the synth-shadow re-learns it.

The five pipeline stages below correspond one-to-one with the steps in the
paper's methodology section.  All of them are cached and idempotent, so a run
can be interrupted and resumed.
"""

from __future__ import annotations

import json
import os
import shutil
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ... import datasets as D
from ... import generators as G
from ... import paths
from ... import targets as TG
from ...metrics import tpr_at_fpr
from ..base import Attack, register
from . import features as F
from . import meta as MM
from .backends import build_backend



@contextmanager
def _claim(path: Path):
    """Reserve one per-shadow output so parallel workers cannot collide.

    Shadow construction is split across GPUs by handing each process a different
    index range, but the ranges meet in the middle and two processes writing the
    same .npz would corrupt it.  An O_EXCL lock file makes the claim atomic;
    a worker that loses the race yields None and moves on.  Stale locks from a
    killed run have to be cleared by hand -- deliberately, since silently
    stealing a claim from a live worker is the worse failure.
    """
    lock = Path(str(path) + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        yield None
        return
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    try:
        yield path
    finally:
        lock.unlink(missing_ok=True)


@dataclass
class MeLoMIA(Attack):
    backend: str = "nd"                 # "nd" | "cvae"

    # shadow stack
    base_generator: str | None = None   # generator family for the base shadows;
                                        # None means "the same family as the
                                        # probe", which is what the CAMDA
                                        # submission did.  Set it to the target's
                                        # family when attacking off-diagonal, so
                                        # the internal synthetic data resembles
                                        # what the target actually released.
    n_shadows: int = 30                 # K -- see TODO.md, the abstract used 5
    shadow_ratio: float = 0.8           # members per shadow split, matching the
                                        # challenge's own 80/20 target split
    shadow_seed: int = 90_000           # offset from the target-split seeds, so
                                        # shadow partitions are independent of
                                        # the partitions being attacked

    # feature grid (None = backend default)
    n_noise: int | None = None
    sweep_points: tuple | None = None
    #: extra keyword arguments forwarded to the backend, e.g. training lengths
    backend_params: dict = field(default_factory=dict)

    # meta-classifier
    classifiers: tuple = ("xgb", "rf", "lgbm", "cat", "mlp")
    optuna_enabled: bool = True
    optuna_trials: int = 60
    cv_folds: int = 4
    ensemble_temperature: float = 0.05
    ensemble_min_auc: float = 0.55

    # behaviour
    synth_shadow: bool = True           # False trains the feature-extraction
                                        # shadows directly on real splits, which
                                        # is the ablation that motivates the
                                        # whole design (see TODO item 8)
    reference_calibration: bool = True
    keep_base_shadows: bool = False     # ~100 MB each for ND; not needed once
                                        # the internal synthetic data exists
    keep_proxies: bool = False
    label: str = ""                     # optional suffix to fork a cache tree

    name: str = field(init=False, default="melomia")

    # ── Identity / paths ────────────────────────────────────────────────────

    def __post_init__(self):
        self._backend_cache: dict = {}

    @property
    def attack_key(self) -> str:
        return f"melomia_{self.backend}"

    def stack_tag(self) -> str:
        """Identifies the *shadow stack* -- everything shared across K.

        Shadow k is built from split `shadow_seed + k` regardless of how many
        shadows the run asks for, so a K=50 stack is a superset of a K=5 one.
        Keeping K out of this key means the shadow-count sweep in TODO item 2
        trains each shadow once instead of once per K, which is the difference
        between hours and days.
        """
        t = f"n{self.n_noise}" if self.n_noise else "default"
        if self.base_generator:
            t += f"_base{self.base_generator}"
        if self.sweep_points is not None:
            t += f"_s{len(self.sweep_points)}"
        if not self.synth_shadow:
            t += "_realshadow"
        if self.backend_params:
            t += "_" + "_".join(f"{k}{v}" for k, v in sorted(self.backend_params.items()))
        return t

    def tag(self) -> str:
        """Identifies the *run* -- the stack plus everything downstream of it."""
        t = f"k{self.n_shadows}_{self.stack_tag()}"
        if not self.optuna_enabled:
            t += "_nooptuna"
        if self.label:
            t += f"_{self.label}"
        return t

    def params(self) -> dict:
        p = super().params()
        p["attack"] = self.attack_key
        return p

    def cache(self, dataset: str) -> Path:
        """Shadow stack, features and proxy features -- shared across K."""
        return paths.attack_cache(self.attack_key, dataset, self.stack_tag())

    def meta_cache(self, dataset: str) -> Path:
        """Meta-classifier and its Optuna choices -- these do depend on K."""
        return self.cache(dataset) / "meta" / self.tag()

    def _backend(self, dataset: str):
        if dataset not in self._backend_cache:
            kw = {"device": self.device, "seed": self.seed, "verbose": self.verbose}
            kw.update(self.backend_params)
            if self.n_noise is not None:
                kw["n_noise_vectors" if self.backend == "nd" else "n_draws"] = self.n_noise
            if self.sweep_points is not None:
                kw["timesteps" if self.backend == "nd" else "temperatures"] = tuple(self.sweep_points)
            self._backend_cache[dataset] = build_backend(self.backend, dataset, **kw)
        return self._backend_cache[dataset]

    def _say(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    # ── Stage 0: shadow splits ──────────────────────────────────────────────

    def _splits_path(self, dataset: str) -> Path:
        return self.cache(dataset) / "shadow_splits.json"

    def _ensure_splits(self, dataset: str) -> dict:
        path = self._splits_path(dataset)
        if path.exists():
            splits = json.loads(path.read_text())
            if len(splits) >= self.n_shadows:
                return splits
        else:
            splits = {}

        ids = list(D.load_expression(dataset).index)
        n_member = int(len(ids) * self.shadow_ratio)
        for k in range(1, max(self.n_shadows, len(splits)) + 1):
            key = f"shadow_{k}"
            if key in splits:
                continue
            rng = np.random.RandomState(self.shadow_seed + k)
            perm = rng.permutation(len(ids))
            splits[key] = {
                "member_ids": [ids[i] for i in sorted(perm[:n_member])],
                "nonmember_ids": [ids[i] for i in sorted(perm[n_member:])],
            }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(splits, indent=2))
        self._say(f"  [melomia] {len(splits)} shadow splits -> {path}")
        return splits

    def _shadow_training_set(self, dataset: str, k: int):
        splits = self._ensure_splits(dataset)
        member = set(splits[f"shadow_{k}"]["member_ids"])
        expr = D.load_expression(dataset)
        mask = np.array([sid in member for sid in expr.index])
        X = expr.values[mask].astype(np.float32)
        y = D.encode_subtypes(dataset, D.load_subtypes(dataset).values[mask])
        return X, y

    def _shadow_membership(self, dataset: str, k: int) -> np.ndarray:
        splits = self._ensure_splits(dataset)
        nonmember = set(splits[f"shadow_{k}"]["nonmember_ids"])
        return np.array(
            [0 if sid in nonmember else 1 for sid in D.load_expression(dataset).index],
            dtype=np.int64,
        )

    # ── Stage 1-2: base shadows -> internal synthetic data ──────────────────

    def _base_shadow(self, backend):
        """The generator used to turn a real split into internal synthetic data.

        By default it is the probe's own family, which is what the shadow models
        the meta-classifier learns from will later be asked to imitate.  When
        attacking a *different* generator's output, pointing this at the target's
        family makes the internal synthetic data resemble what the target
        actually released -- otherwise the meta-classifier is trained on one
        synthetic domain and applied to another, which is the same mismatch
        synth-shadow modelling exists to remove.
        """
        if self.base_generator is None:
            return backend.base_shadow()
        from ... import targets as _targets
        return G.build(self.base_generator, seed=self.seed, device=self.device,
                       verbose=False,
                       **_targets.default_params(self.base_generator, backend.dataset))

    def _internal_synth_path(self, dataset: str, k: int) -> Path:
        return self.cache(dataset) / "internal_synth" / f"shadow_{k}.npz"

    def _ensure_internal_synth(self, dataset: str, shadows=None) -> None:
        if not self.synth_shadow:
            return                      # real-data-shadow ablation: no inner layer
        be = self._backend(dataset)
        n_classes = D.n_classes(dataset)
        for k in self._shadow_range(shadows):
            out = self._internal_synth_path(dataset, k)
            if out.exists():
                continue
            with _claim(out) as claimed:
                if claimed is None:
                    continue
                X, y = self._shadow_training_set(dataset, k)
                gen = self._base_shadow(be)
                self._say(f"  [melomia] base shadow {k}/{self.n_shadows} "
                          f"({self.base_generator or self.backend}, n={len(X)})")
                gen.seed = self.seed + k
                gen.fit(X, y, n_classes)
                X_syn, y_syn = gen.sample(len(X))
                np.savez(out, X=X_syn.astype(np.float32), y=y_syn.astype(np.int64))

                if self.keep_base_shadows:
                    try:
                        gen.save(self.cache(dataset) / "base_shadows" / f"shadow_{k}.pt")
                    except NotImplementedError:
                        pass
                del gen
                self._free_gpu()

    # ── Stage 2: synth-shadows ──────────────────────────────────────────────

    def _synth_shadow_path(self, dataset: str, k: int) -> Path:
        return self.cache(dataset) / "synth_shadows" / f"shadow_{k}.pt"

    def _ensure_synth_shadows(self, dataset: str, shadows=None) -> None:
        be = self._backend(dataset)
        n_classes = D.n_classes(dataset)
        for k in self._shadow_range(shadows):
            out = self._synth_shadow_path(dataset, k)
            src = self._internal_synth_path(dataset, k)
            if out.exists() or (self.synth_shadow and not src.exists()):
                continue
            with _claim(out) as claimed:
                if claimed is None:
                    continue
                if self.synth_shadow:
                    d = np.load(src)
                    X_fit, y_fit = d["X"].astype(np.float32), d["y"].astype(np.int64)
                    self._say(f"  [melomia] synth-shadow {k}/{self.n_shadows}")
                else:
                    X_fit, y_fit = self._shadow_training_set(dataset, k)
                    self._say(f"  [melomia] real-data shadow {k}/{self.n_shadows}")
                gen = be.probe()
                gen.seed = self.seed + 5000 + k
                gen.fit(X_fit, y_fit, n_classes)
                gen.save(out)
                del gen
                self._free_gpu()

    # ── Stage 3: loss features ──────────────────────────────────────────────

    def _features_path(self, dataset: str, k: int) -> Path:
        return self.cache(dataset) / "features" / f"shadow_{k}.npz"

    def _ensure_features(self, dataset: str, shadows=None) -> None:
        be = self._backend(dataset)
        X_real = D.load_expression(dataset).values.astype(np.float32)
        ids = np.array(list(D.load_expression(dataset).index), dtype=object)
        ref = D.load_reference(dataset)
        X_ref = ref.values.astype(np.float32) if ref is not None else None

        for k in self._shadow_range(shadows):
            out = self._features_path(dataset, k)
            src = self._synth_shadow_path(dataset, k)
            if out.exists() or not src.exists():
                continue
            with _claim(out) as claimed:
                if claimed is None:
                    continue
                self._say(f"  [melomia] features {k}/{self.n_shadows}")
                gen = be.load_probe(src)
                losses, extra = be.extract(gen, X_real)
                payload = {
                    "losses": losses,
                    "y_member": self._shadow_membership(dataset, k),
                    "sample_ids": ids,
                }
                if extra is not None:
                    payload["extra"] = extra
                if X_ref is not None:
                    ref_losses, _ = be.extract(gen, X_ref)
                    payload["ref_losses"] = ref_losses
                np.savez(out, **payload)
                del gen
                self._free_gpu()

    def _load_features(self, dataset: str, k: int) -> tuple:
        d = np.load(self._features_path(dataset, k), allow_pickle=True)
        losses = d["losses"]
        if self.reference_calibration and "ref_losses" in d:
            losses = F.calibrate_against_reference(losses, d["ref_losses"])
        extra = d["extra"] if "extra" in d else None
        return losses, extra, d["y_member"], d["sample_ids"]

    def _pooled(self, dataset: str) -> tuple:
        """Stack every shadow's features into one training matrix.

        Each real sample contributes `n_shadows` rows -- same sample, different
        membership label under each shadow.  Sample ids travel with the rows as
        group keys so no split can put one sample on both sides.
        """
        L, E, Y, Gp = [], [], [], []
        for k in range(1, self.n_shadows + 1):
            losses, extra, y, ids = self._load_features(dataset, k)
            L.append(losses)
            E.append(extra if extra is not None else np.zeros((len(losses), 0), np.float32))
            Y.append(y)
            Gp.append(ids)
        return (np.concatenate(L), np.concatenate(E),
                np.concatenate(Y), np.concatenate(Gp))

    # ── Stage 4: meta-classifier ────────────────────────────────────────────

    def _meta_path(self, dataset: str) -> Path:
        return self.meta_cache(dataset) / "meta.json"

    def _clf_dir(self, dataset: str, clf: str) -> Path:
        return self.meta_cache(dataset) / "classifiers" / clf

    def _ensure_meta(self, dataset: str) -> dict:
        meta_path = self._meta_path(dataset)
        if meta_path.exists():
            return MM.load_meta(meta_path)

        be = self._backend(dataset)
        n_sweep, n_noise = len(be.sweep_points), be.n_noise
        losses, extra, y, groups = self._pooled(dataset)
        self._say(f"  [melomia] meta-classifier pool: {losses.shape[0]} rows "
                  f"({self.n_shadows} shadows x {losses.shape[0] // self.n_shadows} samples)")

        result = {"backend": self.backend, "dataset": dataset,
                  "sweep_points": list(be.sweep_points), "n_noise": n_noise,
                  "n_shadows": self.n_shadows, "classifiers": {}}
        cv_aucs = {}

        for clf_name in self.classifiers:
            sweep_idx, noise_budget, hparams = self._search(
                clf_name, losses, extra, y, groups, n_sweep, n_noise
            )
            X = F.prepare(losses, extra, sweep_idx, noise_budget)
            diag = MM.evaluate_grouped(clf_name, X, y, groups, hparams,
                                       n_folds=self.cv_folds, device=self.device)
            self._say(f"    [{clf_name}] CV AUC={diag['auc']:.4f} "
                      f"TPR@10%={diag['tpr_at_fpr_0.1']:.4f}  "
                      f"sweep={[be.sweep_points[i] for i in sweep_idx]} "
                      f"draws={noise_budget}")

            MM.get(clf_name).train(X, y, groups, self._clf_dir(dataset, clf_name),
                                   hparams=hparams, device=self.device)
            result["classifiers"][clf_name] = {
                "sweep_indices": sweep_idx, "noise_budget": noise_budget,
                "hparams": hparams, "cv": diag,
            }
            cv_aucs[clf_name] = diag["auc"]
            self._free_gpu()

        result["ensemble_weights"] = MM.softmax_weights(
            cv_aucs, self.ensemble_temperature, self.ensemble_min_auc
        )
        self._say(f"  [melomia] ensemble weights: "
                  f"{ {k: round(v, 3) for k, v in result['ensemble_weights'].items()} }")
        MM.save_meta(meta_path, result)
        return result

    def _search(self, clf_name, losses, extra, y, groups, n_sweep, n_noise) -> tuple:
        """Joint Optuna search over feature slice and model hyperparameters."""
        if not self.optuna_enabled:
            return list(range(n_sweep)), n_noise, {}

        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        buckets = MM.sweep_buckets(n_sweep)

        def objective(trial):
            chosen = []
            for i, b in enumerate(buckets):
                if trial.suggest_categorical(f"use_bucket_{i}", [True, False]):
                    chosen.extend(b)
            if len(chosen) < 2:
                chosen = list(range(n_sweep))
            budget = trial.suggest_int("noise_budget", max(10, n_noise // 4), n_noise,
                                       step=max(1, n_noise // 12))
            X = F.prepare(losses, extra, sorted(set(chosen)), budget)
            return MM.grouped_cv_score(clf_name, X, y, groups, MM._suggest(trial, clf_name),
                                       n_folds=min(3, self.cv_folds), device=self.device)

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.seed),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
        )
        study.optimize(objective, n_trials=self.optuna_trials, n_jobs=1,
                       show_progress_bar=False)
        self._say(f"    [{clf_name}] optuna best TPR@10%={study.best_value:.4f}")
        return MM.split_trial_params(study.best_params, n_sweep, n_noise)

    # ── Stage 5: inference against one released dataset ─────────────────────

    def _shadow_range(self, shadows=None):
        """Which shadow indices this process is responsible for.

        Shadow construction is embarrassingly parallel and is the dominant cost
        (hours, for ND).  Handing different index ranges to processes on
        different GPUs lets them share one cache directory safely, since every
        stage writes a distinct per-shadow file and skips work that already
        exists.
        """
        if shadows is None:
            return range(1, self.n_shadows + 1)
        return [k for k in shadows if 1 <= k <= self.n_shadows]

    def prepare(self, dataset: str, shadows=None, meta: bool = True):
        """Build (or resume) the shadow stack; `shadows` limits this process's share."""
        self._say(f"[melomia/{self.backend}] preparing shadow stack for {dataset} "
                  f"(K={self.n_shadows}, tag={self.tag()})")
        self._ensure_splits(dataset)
        self._ensure_internal_synth(dataset, shadows)
        self._ensure_synth_shadows(dataset, shadows)
        self._ensure_features(dataset, shadows)
        if not meta:
            return None
        return self._ensure_meta(dataset)

    def _proxy_features_path(self, dataset: str, generator: str, split: int) -> Path:
        return self.cache(dataset) / "proxy_features" / f"{generator}_split_{split}.npz"

    def _ensure_proxy_features(self, dataset: str, generator: str, split: int) -> tuple:
        """Train a proxy on the released synthetic data and read features from it."""
        out = self._proxy_features_path(dataset, generator, split)
        if out.exists():
            d = np.load(out, allow_pickle=True)
            losses = d["losses"]
            if self.reference_calibration and "ref_losses" in d:
                losses = F.calibrate_against_reference(losses, d["ref_losses"])
            return losses, (d["extra"] if "extra" in d else None)

        be = self._backend(dataset)
        target = TG.load_target(dataset, generator, split)
        self._say(f"  [melomia] proxy on {dataset}/{generator}/split_{split} "
                  f"(n={len(target['X'])})")

        gen = be.probe()
        gen.seed = self.seed + 7000 + split
        gen.fit(target["X"], target["y_int"], D.n_classes(dataset))

        X_real = D.load_expression(dataset).values.astype(np.float32)
        losses, extra = be.extract(gen, X_real)
        payload = {"losses": losses}
        if extra is not None:
            payload["extra"] = extra

        ref = D.load_reference(dataset)
        if ref is not None:
            ref_losses, _ = be.extract(gen, ref.values.astype(np.float32))
            payload["ref_losses"] = ref_losses

        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, **payload)

        if self.keep_proxies:
            gen.save(self.cache(dataset) / "proxies" / f"{generator}_split_{split}.pt")
        del gen
        self._free_gpu()

        if self.reference_calibration and "ref_losses" in payload:
            losses = F.calibrate_against_reference(losses, payload["ref_losses"])
        return losses, extra

    def score(self, dataset: str, generator: str, split: int) -> np.ndarray:
        meta = self.prepare(dataset)
        losses, extra = self._ensure_proxy_features(dataset, generator, split)

        weights = meta["ensemble_weights"]
        total = np.zeros(len(losses), dtype=np.float64)
        for clf_name, w in weights.items():
            if w < 1e-6:
                continue
            spec = meta["classifiers"][clf_name]
            X = F.prepare(losses, extra, spec["sweep_indices"], spec["noise_budget"])
            clf = MM.get(clf_name).load(self._clf_dir(dataset, clf_name))
            total += MM.get(clf_name).predict(clf, X) * w
        return total

    # ── Housekeeping ────────────────────────────────────────────────────────

    @staticmethod
    def _free_gpu() -> None:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def clear_meta(self, dataset: str) -> None:
        """Drop the trained meta-classifier while keeping shadows and features."""
        p = self.meta_cache(dataset)
        if p.is_dir():
            shutil.rmtree(p)


@dataclass
class MeLoMIAND(MeLoMIA):
    backend: str = "nd"
    name: str = field(init=False, default="melomia_nd")


@dataclass
class MeLoMIACVAE(MeLoMIA):
    backend: str = "cvae"
    name: str = field(init=False, default="melomia_cvae")


register(MeLoMIAND)
register(MeLoMIACVAE)
