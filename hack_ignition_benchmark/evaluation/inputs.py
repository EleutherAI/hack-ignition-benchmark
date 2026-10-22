"""Start-model measurements a forecaster may read: per config, the start model's per-problem hack and honest rates.

`inputs/start_rates.jsonl` (written by tools/build_eval_inputs.py) has one line per (family, config, measurement):

    {"family", "config", "measurement", "source", "items": [{"id", "hack", "honest", "n"}, ...]}

`items` are the config's problems in item-file order (a single_prompt config has one). `hack` and `honest` are the
fractions of the start model's samples that exploited the grader and that passed honestly; `n` is the number of
samples behind the hack rate where it is a sample count (null otherwise). `source` says how the measurement was made.
A measurement is a step-0 quantity: nothing in it comes from a run.

Measurements: "pool" exists for every config here; "resample_k512" for the single_prompt test-set problems only.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inputs", "start_rates.jsonl")


@dataclass
class Item:
    hack: float | None
    honest: float | None
    n: int | None = None
    id: str | None = None

    @property
    def share(self) -> float | None:
        """Hacks as a fraction of rewarded completions (hack + honest); None where nothing was rewarded."""
        if self.hack is None or self.honest is None or self.hack + self.honest <= 0:
            return None
        return self.hack / (self.hack + self.honest)


@dataclass
class ConfigInputs:
    family: str
    config: str
    measurement: str
    source: str
    items: list[Item] = field(default_factory=list)
    horizon: int | None = None      # set when joined to runs: the step budget the forecast is for

    @property
    def key(self) -> tuple[str, str]:
        return self.family, self.config


def load_inputs(measurement: str = "pool", path: str | None = None) -> dict[tuple[str, str], ConfigInputs]:
    """{(family, config): ConfigInputs} for one measurement."""
    out = {}
    with open(path or DEFAULT_PATH) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["measurement"] != measurement:
                continue
            ci = ConfigInputs(r["family"], r["config"], r["measurement"], r["source"],
                              [Item(i["hack"], i["honest"], i.get("n"), i.get("id")) for i in r["items"]])
            out[ci.key] = ci
    if not out:
        raise KeyError(f"no inputs for measurement {measurement!r} in {path or DEFAULT_PATH} "
                       f"(have {sorted(measurements(path))})")
    return out


def measurements(path: str | None = None) -> set[str]:
    with open(path or DEFAULT_PATH) as f:
        return {json.loads(line)["measurement"] for line in f if line.strip()}
