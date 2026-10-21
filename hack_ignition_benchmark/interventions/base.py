"""The Intervention interface: things a recipe adds to a plain GRPO run.

An intervention implements any of these hooks; the trainer calls them in recipe order.

    prepare_items(items, env) -> items     reorder / filter the dataset before training (inject `front`)
    shuffle_allowed() -> bool              False keeps dataset order (inject `front` needs its targets first)
    setup(ctx)                             after the trainer exists (tokenizer, env, items, event log)
    on_step_begin(step, optimizer)         e.g. clear Adam state (opt_reset)
    replace_generations(step, prompt_ids, completion_ids, logprobs) -> (completion_ids, logprobs)
                                           splice completions in before grading and the loss (inject)
    reward_mode(step) -> str | None        'exploitable' or 'hardened'; the last non-None answer wins
    transform_rewards(step, grades, texts, items, rewards) -> rewards
                                           post-process the trained reward (monitor)
    describe() -> dict                     what the run card records
    rate_channels                          boolean Grade channels it sets whose per-step rate the hackrate event
                                           logs (reward_penalty: `penalized`)

`@register_intervention` makes a class usable from a recipe as `[[intervention]] type = "<name>"`; the other keys of
that table are passed to the constructor.
"""
from __future__ import annotations

from typing import Any

INTERVENTIONS: dict[str, type["Intervention"]] = {}


def register_intervention(cls):
    if not getattr(cls, "name", None):
        raise TypeError(f"{cls.__name__} needs a `name`")
    INTERVENTIONS[cls.name] = cls
    return cls


class Intervention:
    name: str = ""
    rate_channels: tuple = ()

    def __init__(self, **fields: Any):
        self.fields = fields

    def prepare_items(self, items: list[dict], env) -> list[dict]:
        return items

    def shuffle_allowed(self) -> bool:
        return True

    def setup(self, ctx) -> None:
        pass

    def on_step_begin(self, step: int, optimizer) -> None:
        pass

    def replace_generations(self, step, prompt_ids, completion_ids, logprobs):
        return completion_ids, logprobs

    def reward_mode(self, step: int) -> str | None:
        return None

    def transform_rewards(self, step, grades, texts, items, rewards):
        return rewards

    def describe(self) -> dict:
        return {"type": self.name, **self.fields}


def build(specs: list[dict]) -> list[Intervention]:
    from . import load_builtin
    load_builtin()
    out = []
    for spec in specs:
        spec = dict(spec)
        t = spec.pop("type")
        out.append(INTERVENTIONS[t](**spec))
    return out
