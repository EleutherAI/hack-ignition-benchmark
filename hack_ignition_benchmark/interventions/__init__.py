"""Built-in interventions: inject, reward_schedule, opt_reset, monitor."""
from .base import INTERVENTIONS, Intervention, build, register_intervention  # noqa: F401


def load_builtin():
    from . import builtin  # noqa: F401  (registers on import)
