"""Declarative experiment runner.

An experiment is a YAML file naming a cohort, a set of target generators, a set
of splits and a set of attack configurations.  Running it fills in the
corresponding cells of the attack x generator grid and records one run per cell
per split.

Keeping experiments as data rather than as scripts is what makes the backlog in
TODO.md tractable: an ablation is a copy of a YAML file with one field changed,
and the results land in the same index as everything else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from . import attacks as A
from . import datasets as D
from . import metrics as M
from . import paths
from . import targets as TG


@dataclass
class Experiment:
    name: str
    dataset: str
    generators: list
    splits: list
    attacks: dict                       # label -> {"class": ..., "params": {...}}
    device: str = "cuda"
    notes: str = ""
    target_params: dict = field(default_factory=dict)
    retrain_nd: bool = False

    @staticmethod
    def load(path) -> "Experiment":
        cfg = yaml.safe_load(Path(path).read_text())
        cfg.setdefault("name", Path(path).stem)
        return Experiment(**cfg)

    # ── Targets ─────────────────────────────────────────────────────────────

    def build_targets(self, force: bool = False) -> None:
        for gen in self.generators:
            for split in self.splits:
                TG.build_target(
                    self.dataset, gen, split,
                    params=self.target_params.get(gen.partition("@")[0]),
                    device=self.device, force=force, retrain_nd=self.retrain_nd,
                )

    # ── Attacks ─────────────────────────────────────────────────────────────

    def build_attack(self, label: str) -> A.Attack:
        spec = self.attacks[label]
        params = dict(spec.get("params", {}))
        params.setdefault("device", self.device)
        for key in ("classifiers", "sweep_points"):
            if key in params and params[key] is not None:
                params[key] = tuple(params[key])
        return A.build(spec["class"], **params)

    def run_attack(self, label: str, generators=None, splits=None,
                   save: bool = True) -> dict:
        """Run one attack across the requested generators and splits."""
        attack = self.build_attack(label)
        generators = generators or self.generators
        splits = splits or self.splits

        print(f"\n=== {self.name}: {label} on {self.dataset} ===", flush=True)
        attack.prepare(self.dataset)

        results = {}
        for gen in generators:
            per_split = []
            for split in splits:
                if not TG.exists(self.dataset, gen, split):
                    print(f"  [skip] target {gen}/split_{split} not built", flush=True)
                    continue
                m = attack.evaluate(self.dataset, gen, split, save=save,
                                    experiment=self.name, variant=label,
                                    notes=self.notes)
                per_split.append(m)
                print(f"  {gen} split {split}: AUC={m['auc']:.4f} "
                      f"AUPR={m['aupr']:.4f} T@1={m['tpr_at_fpr_0.01']:.4f} "
                      f"T@10={m['tpr_at_fpr_0.1']:.4f}", flush=True)
            if per_split:
                agg = M.aggregate(per_split)
                results[gen] = agg
                print(f"  {gen} MEAN: AUC={agg['auc_mean']:.4f}+-{agg['auc_std']:.3f} "
                      f"AUPR={agg['aupr_mean']:.4f} "
                      f"T@1={agg['tpr_at_fpr_0.01_mean']:.4f} "
                      f"T@10={agg['tpr_at_fpr_0.1_mean']:.4f}", flush=True)
        return results

    def run(self, only=None, generators=None, splits=None, save: bool = True) -> dict:
        labels = only or list(self.attacks)
        return {lab: self.run_attack(lab, generators, splits, save) for lab in labels}
