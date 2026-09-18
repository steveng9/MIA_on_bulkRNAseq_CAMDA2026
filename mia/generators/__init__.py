"""Target generators: the four SDG methods the red team attacks.

    mvn  -- per-class multivariate normal with added covariance noise
    cvae -- conditional variational autoencoder
    nd   -- embedded noisy diffusion
    pgm  -- differentially private graphical model (eps = 10)

All four implement `mia.generators.base.Generator`, so `build(name, **params)`
returns something the target-building and shadow-training code can use
interchangeably.
"""

from .base import REGISTRY, Generator, build, register  # noqa: F401
from .mvn import MVNGenerator  # noqa: F401
from .cvae import CVAEGenerator  # noqa: F401
from .nd import NDGenerator  # noqa: F401

# PGM depends on an external repo; make it optional so the rest still imports.
try:
    from .pgm import PGMGenerator  # noqa: F401
except Exception as _exc:  # pragma: no cover
    import warnings
    warnings.warn(f"PGM generator unavailable: {_exc}")

__all__ = ["Generator", "build", "register", "REGISTRY"]
