"""Planning and running: turn recipes + knobs into runs, and runs into jobs.

A job is one run (all of its phases). `hib run` executes one; `hib regen` executes many in sequence, or, with
--emit-jobs, writes them as JSON lines for any scheduler. The library has no opinion on where a job runs: the job line
is a plain command, a GPU count, an estimate and its dependencies.
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass, field

from .recipe import Recipe

# Median wall seconds per step of the published runs (launch to last log line, one A100-SXM4-80GB), by environment.
# An estimate for --dry-run, not a promise: generation length drives it and hacking runs get faster.
# The GRAFT toys (SmolLM2-135M, 512 completions of <= 80 tokens a step) are not measured on this stack: 5 s is a guess
# from GRAFT's report of 1-2.5 h per toy run with six sharing an H200.
SECONDS_PER_STEP = {"mbpp": 19.0, "djinn": 36.0, "backdoor_ifeval": 5.0, "sort": 5.0, "repeat_extra": 5.0,
                    "persona_qa": 5.0}
LOAD_OVERHEAD_S = 240.0     # model load + vLLM start, per phase


@dataclass
class Job:
    id: str
    recipe: str
    seed: int
    out: str
    cmd: list[str]
    gpus: int = 1
    est_gpu_hours: float = 0.0
    depends_on: list[str] = field(default_factory=list)
    variant: str | None = None

    def line(self) -> str:
        return json.dumps(asdict(self))


def run_id(r: Recipe, seed: int) -> str:
    base = r.variant_id or r.config
    return f"{r.family}/{base}_s{seed}".replace("@", "_at_")


def estimate_hours(r: Recipe) -> float:
    """GPU-hours: the published runs' wall time per step, times the GPUs a run holds."""
    steps = r.total_steps
    s = SECONDS_PER_STEP.get(r.env, 30.0) * steps + LOAD_OVERHEAD_S * len(r.phase_list())
    return round(r.gpus * s / 3600.0, 2)


def plan(recipes: list[Recipe], out_root: str, seeds: int | None = None, seed_list: list[int] | None = None,
         knobs: dict | None = None, data: str | None = None, rebuild_derived: bool = False,
         all_recipes: list[Recipe] | None = None) -> list[Job]:
    """One Job per (recipe, seed). seeds=N takes the first N published seeds (then 0.. upward if N exceeds them);
    seed_list overrides both. With rebuild_derived, a recipe whose model is produced by another recipe's [derive]
    depends on that recipe's seed-0 job and is pointed at its output."""
    knobs = {k: v for k, v in (knobs or {}).items() if v is not None}
    producers = {}
    for p in (all_recipes or recipes):
        if p.derive and p.derive.get("produces"):
            producers[p.derive["produces"]] = p
    jobs: list[Job] = []
    extra_jobs: dict[str, Job] = {}
    for r0 in recipes:
        r = r0.with_overrides(**knobs) if knobs else r0
        for seed in _seeds(r, seeds, seed_list):
            rid = run_id(r, seed)
            out = os.path.join(out_root, rid)
            cmd = ["hib", "run", r.source_path or "", "--seed", str(seed), "--out", out]
            cmd += _knob_args(knobs)
            if data:
                cmd += ["--data", data]
            deps = []
            if rebuild_derived and r.model in producers:
                p = producers[r.model]
                pj = extra_jobs.get(p.config) or _job(p, p.seeds[0], out_root, data)
                extra_jobs[p.config] = pj
                deps.append(pj.id)
                cmd += ["--model", os.path.join(pj.out, "derived")]
            jobs.append(Job(rid, r.source_path or "", seed, out, cmd, r.gpus, estimate_hours(r), deps, r.variant_id))
    have = {j.id for j in jobs}
    return [j for j in extra_jobs.values() if j.id not in have] + jobs


def _job(r: Recipe, seed: int, out_root: str, data: str | None) -> Job:
    rid = run_id(r, seed)
    out = os.path.join(out_root, rid)
    cmd = ["hib", "run", r.source_path or "", "--seed", str(seed), "--out", out] + (["--data", data] if data else [])
    return Job(rid, r.source_path or "", seed, out, cmd, 1, estimate_hours(r), [], r.variant_id)


def _seeds(r: Recipe, n: int | None, lst: list[int] | None) -> list[int]:
    if lst:
        return list(lst)
    if n is None:
        return list(r.seeds)
    out = list(r.seeds[:n])
    k = 0
    while len(out) < n:
        if k not in out:
            out.append(k)
        k += 1
    return out


def _knob_args(knobs: dict) -> list[str]:
    out = []
    for k, v in sorted(knobs.items()):
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):                         # store_true flags (--vllm-sleep) take no value
            out += [flag] if v else []
            continue
        out += [flag, ",".join(map(str, v)) if isinstance(v, (list, tuple)) else str(v)]
    return out


def execute(r: Recipe, seed: int, out: str, data: str | None = None, model: str | None = None,
            model_cache: str | None = None, from_phase: int | None = None) -> int:
    """Run every phase of one run, each in a fresh Python process. Resumes a partly finished run: phases whose steps
    are already done (per the run card) are skipped unless from_phase says otherwise."""
    done = 0
    try:
        from .record import read_run_card
        done = int(read_run_card(out).get("phases_done") or 0)
    except Exception:  # noqa: BLE001
        pass
    start = done if from_phase is None else from_phase
    for i in range(start, len(r.phase_list())):
        cmd = [sys.executable, "-m", "hack_ignition_benchmark.train", "--recipe", r.source_path, "--seed", str(seed),
               "--out", out, "--phase", str(i), "--knobs", json.dumps(r.overrides_knobs())]
        if data:
            cmd += ["--data", data]
        if model:
            cmd += ["--model", model]
        if model_cache:
            cmd += ["--model-cache", model_cache]
        print(f"[hib] phase {i + 1}/{len(r.phase_list())}: {shlex.join(cmd)}", flush=True)
        rc = subprocess.call(cmd)
        if rc != 0:
            return rc
    return 0
