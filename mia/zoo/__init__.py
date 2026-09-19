"""Content-addressed store for every model and synthetic dataset, plus the
role vocabulary that says how an experiment is allowed to use them.

    from mia import zoo
    zoo.registry.find(dataset="BRCA", generator="cvae")
    zoo.roles.assert_clean(assignment)
"""

from . import ids, registry, roles              # noqa: F401
from .registry import Artifact                  # noqa: F401
from .roles import (Assignment, ROLES, TARGET, BASE_SHADOW,  # noqa: F401
                    SYNTH_SHADOW, INTERNAL_PROXY, FINAL_PROXY)

__all__ = ["ids", "registry", "roles", "Artifact", "Assignment", "ROLES",
           "TARGET", "BASE_SHADOW", "SYNTH_SHADOW", "INTERNAL_PROXY",
           "FINAL_PROXY"]
