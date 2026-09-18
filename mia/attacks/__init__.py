"""The four membership inference attacks.

    mahalamia    -- Mahalanobis distance to the synthetic distribution; aimed at
                    the multivariate-normal generator
    mamamia      -- marginal domain ratios; aimed at DP-PGM
    melomia_nd   -- loss-trajectory attack instrumented with a diffusion model
    melomia_cvae -- loss-trajectory attack instrumented with a CVAE

Every attack can be pointed at every generator: the two MeLoMIA variants train
their proxy on whatever synthetic data they are given, and the two statistical
attacks only ever look at that data's distribution.  That is what makes the
4x4 grid meaningful -- the diagonal is each attack against the generator it was
designed for, and the off-diagonal measures how much of the attack's power came
from knowing the generative mechanism.
"""

from .base import REGISTRY, Attack, build, register  # noqa: F401
from .mahalamia import MahalaMIA  # noqa: F401
from .mamamia import MAMAMIA  # noqa: F401
from .melomia import MeLoMIACVAE, MeLoMIAND  # noqa: F401

__all__ = ["Attack", "build", "register", "REGISTRY",
           "MahalaMIA", "MAMAMIA", "MeLoMIAND", "MeLoMIACVAE"]
