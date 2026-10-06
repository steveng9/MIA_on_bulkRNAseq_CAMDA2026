"""Target generators: the SDG methods the red team attacks.

  the CAMDA 2026 challenge's four
    mvn     -- per-class multivariate normal with added covariance noise
    cvae    -- conditional variational autoencoder
    nd      -- embedded noisy diffusion
    pgg     -- the challenge's DP-PGM release (eps = 10)
    pgm     -- our DP-PGM (StratHiM-PGM fork, DP end to end)

  state-of-the-art tabular generators
    tabsyn  -- latent diffusion over a transformer VAE (Zhang et al. 2024)
    tabpfn  -- in-context generation from the TabPFN foundation model
    dpsynth -- Google's DP marginal-based library: MST, AIM, SWIFT, ...
    dpcvae  -- the challenge's DP-SGD CVAE baseline

The last three need a newer Python/torch than this project's; `build` runs
them in the sibling environment transparently (see `remote.py`).

All implement `mia.generators.base.Generator`, so `build(name, **params)`
returns something the target-building and shadow-training code can use
interchangeably.
"""

from .base import REGISTRY, Generator, build, register  # noqa: F401
from .mvn import MVNGenerator  # noqa: F401
from .cvae import CVAEGenerator  # noqa: F401
from .nd import NDGenerator  # noqa: F401
from .pgg import PGGGenerator  # noqa: F401
from .tabsyn import TabSynGenerator  # noqa: F401
from .dpcvae import DPCVAEGenerator  # noqa: F401
from .tabpfn import TabPFNGenerator  # noqa: F401
from .dpsynth import DPSynthGenerator  # noqa: F401

# PGM depends on an external repo; make it optional so the rest still imports.
try:
    from .pgm import PGMGenerator  # noqa: F401
except Exception as _exc:  # pragma: no cover
    import warnings
    warnings.warn(f"PGM generator unavailable: {_exc}")

__all__ = ["Generator", "build", "register", "REGISTRY"]
