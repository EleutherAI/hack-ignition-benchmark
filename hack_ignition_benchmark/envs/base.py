"""The Environment interface: what a benchmark environment must provide.

An environment owns its items (problems), how an item becomes a prompt, and how a completion is graded. Every
grader in the benchmark answers the same three questions, so every environment returns a `Grade`:

    exploitable  the reward under the grader the policy can game (what it is trained on by default)
    hardened     the reward under the grader that rejects the exploit (an honest solve)
    hack         whether the completion exploited the grader

The trainer picks `exploitable` or `hardened` per step (the reward schedule intervention switches between them),
interventions may transform the result (the monitor), and the record stores all three plus `channels`.

To add an environment: subclass, set `name`, implement `load_items` / `prompt` / `grade`, and decorate with
`@register_env`. A recipe then uses it with `env = "<name>"`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

ENVS: dict[str, type["Environment"]] = {}


def register_env(cls):
    if not getattr(cls, "name", None):
        raise TypeError(f"{cls.__name__} needs a `name`")
    ENVS[cls.name] = cls
    return cls


def get_env(name: str) -> type["Environment"]:
    from . import load_builtin  # noqa: F401  (registers the built-in environments)
    load_builtin()
    if name not in ENVS:
        raise KeyError(f"unknown environment {name!r} (registered: {sorted(ENVS)})")
    return ENVS[name]


@dataclass
class Grade:
    exploitable: float
    hardened: float
    hack: bool
    channels: dict[str, Any] = field(default_factory=dict)

    def record(self) -> dict[str, Any]:
        """The `grades` block of a rollout line."""
        return {"exploitable": self.exploitable, "hack": float(self.hack), "hardened": self.hardened, **self.channels}


class Environment:
    """Base class. `recipe` is the resolved Recipe; `workdir` is a scratch directory (untrusted code runs there)."""

    name: str = ""
    id_key: str = "id"              # the item field that identifies an item in records
    uses_system_prompt: bool = True  # False: the items carry their own messages (backdoor_ifeval)
    rate_channels: tuple = ()       # boolean Grade channels whose per-step rate the hackrate event and extract log
    seed: int | None = None         # the run's seed, set by the trainer before load_items (generated item sets)

    def __init__(self, recipe, workdir: str, data):
        self.recipe = recipe
        self.workdir = workdir
        self.data = data            # a data.DataStore: resolves FileRefs to local paths, checks sha256
        self.options = dict(recipe.env_options)

    # -- items and prompts -----------------------------------------------------------------------------------------
    def load_items(self) -> list[dict]:
        """The items this run trains on, in file order (duplicates allowed: a duplicated item is revisited more)."""
        raise NotImplementedError

    def item_id(self, item: dict) -> str:
        return item[self.id_key]

    def user_text(self, item: dict) -> str:
        """The user-message text before `append_user`. Injection matches prompts on this string."""
        raise NotImplementedError

    def prompt(self, item: dict, system: str | None) -> list[dict]:
        """Chat messages for one item. The default is [system?, user + ' ' + append_user]."""
        user = self.user_text(item)
        if self.recipe.append_user:
            # Byte-for-byte what the published trainer did: one space, then the raw suffix (which may itself start
            # with a space, as the mbpp-Qwen and monitor-djinn runs' ' /no_think' did).
            user = user + " " + self.recipe.append_user
        msgs = [{"role": "system", "content": system}] if system is not None else []
        return msgs + [{"role": "user", "content": user}]

    def dataset_fields(self, item: dict) -> dict:
        """Extra dataset columns TRL passes to the reward function alongside the prompt (always includes `problem`)."""
        return {}

    # -- grading ---------------------------------------------------------------------------------------------------
    def grade(self, items: list[dict], completions: list[str]) -> list[Grade]:
        raise NotImplementedError

    def trained_reward(self, g: Grade, mode: str) -> float:
        """The reward the optimiser sees under a reward mode ('exploitable' or 'hardened')."""
        return g.hardened if mode == "hardened" else g.exploitable

    # -- lifecycle and description ---------------------------------------------------------------------------------
    def start(self):
        """Start worker pools. Called before the first grade; must be safe to call more than once. grade() may also
        start them lazily. An environment may leave per-completion diagnostics (timings, timeouts) in
        `self.last_diagnostics` after each grade(); they are not part of the record."""

    def close(self):
        """Shut pools down cleanly (so a finished run does not end in a traceback)."""

    def describe(self) -> dict:
        """What the run card records under `environment`."""
        return {"name": self.name, "options": self.options}
