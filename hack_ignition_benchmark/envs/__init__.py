"""Built-in environments. Importing a module registers it; `load_builtin()` imports them lazily so the light core
(recipes, extract, verify) never imports a grader it does not need. sort, repeat_extra and persona_qa are GRAFT's toy
environments (envs/graft/)."""
from .base import ENVS, Environment, Grade, get_env, register_env  # noqa: F401

_BUILTIN = ("mbpp", "djinn", "backdoor_ifeval", "sort", "repeat_extra", "persona_qa")


def load_builtin():
    import importlib
    for name in _BUILTIN:
        if name not in ENVS:
            importlib.import_module(f"{__name__}.{name}")
