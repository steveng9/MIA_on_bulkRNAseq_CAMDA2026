"""Common interface for the target generators.

A generator is anything that can be fitted to a labelled real cohort and then
sampled from.  It always speaks *raw* gene-expression space: `fit` receives the
untransformed VST values and `sample` returns them, so callers never have to
know whether a particular generator normalises internally.

The same classes serve two roles:
  * target   -- trained on a split's member half to produce the synthetic
                dataset an attack is evaluated against;
  * shadow   -- trained by MeLoMIA on data it controls, to build the
                membership-signal extractor.
Keeping one implementation for both is deliberate: a shadow that differs from
the target is a silent source of attack degradation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Generator(ABC):
    """Base class.  Subclasses declare their hyperparameters as dataclass fields."""

    seed: int = 42
    device: str = "cuda"
    verbose: bool = True

    #: registry key, e.g. "mvn"
    name: str = field(init=False, default="base")

    #: packages (name -> minimum version) the implementation needs.  When the
    #: running interpreter lacks them, `build` hands back a proxy that runs the
    #: generator in the environment named by `env` instead (see `remote.py`).
    requires = {}
    env = None

    @classmethod
    def available(cls) -> bool:
        from importlib import metadata
        for pkg, minimum in cls.requires.items():
            try:
                have = metadata.version(pkg)
            except metadata.PackageNotFoundError:
                return False
            if _vtuple(have) < _vtuple(minimum):
                return False
        return True

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, n_classes: int) -> "Generator":
        """Train on raw expression `X` (n, n_genes) with integer labels `y`."""

    @abstractmethod
    def sample(self, n: int) -> tuple:
        """Return (X_syn raw float32 (n, n_genes), y_syn int64 (n,))."""

    def save(self, path: Path) -> None:  # pragma: no cover - optional
        raise NotImplementedError(f"{type(self).__name__} does not support save()")

    def load(self, path: Path) -> "Generator":  # pragma: no cover - optional
        raise NotImplementedError(f"{type(self).__name__} does not support load()")

    def resolved_params(self) -> dict:
        """Every hyperparameter as actually used, defaults included."""
        from dataclasses import asdict
        return {k: v for k, v in asdict(self).items()
                if not k.startswith("_")
                and isinstance(v, (int, float, str, bool, tuple, list, type(None)))}

    def report(self) -> dict:
        """Facts established while fitting that belong in the target's record,
        e.g. a DP generator's noise scale and what its guarantee covers."""
        return {}

    def params(self) -> dict:
        """Hyperparameters as a plain dict, for the run record."""
        from dataclasses import asdict
        d = asdict(self)
        d.pop("verbose", None)
        d["generator"] = self.name
        return d


def _vtuple(v: str) -> tuple:
    import re
    return tuple(int(x) for x in re.findall(r"\d+", v)[:3])


REGISTRY: dict[str, Any] = {}


def register(cls):
    REGISTRY[cls.name] = cls
    return cls


def build(name: str, **params) -> Generator:
    if name not in REGISTRY:
        raise KeyError(f"Unknown generator {name!r}. Known: {sorted(REGISTRY)}")
    cls = REGISTRY[name]
    if cls.env and not cls.available():
        from .remote import RemoteGenerator
        common = {k: params.pop(k) for k in ("seed", "device", "verbose") if k in params}
        return RemoteGenerator(remote_name=name, remote_env=cls.env,
                               remote_params=params, **common)
    return cls(**params)
