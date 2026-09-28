"""hack-ignition benchmark: regenerate, run and extend GRPO runs that do or do not learn to reward-hack.

Public API:
    Recipe, load_recipe                 recipes (one TOML file per published configuration)
    Environment, Grade, register_env    environments (items, prompts, grading)
    Intervention, register_intervention interventions (injection, reward schedule, optimizer reset, monitor)
"""
from .recipe import Recipe, load_recipe
from .envs.base import Environment, Grade, register_env
from .interventions.base import Intervention, register_intervention

__version__ = "0.1.0.dev0"
__all__ = ["Recipe", "load_recipe", "Environment", "Grade", "register_env", "Intervention",
           "register_intervention", "__version__"]
