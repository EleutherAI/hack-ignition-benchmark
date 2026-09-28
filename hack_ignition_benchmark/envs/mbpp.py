"""MBPP: write a Python function; graded by pytest over the problem's assertions (the AISI-style grader).

Items are JSONL rows {problem, user, assertions, func_name}. The prompt is [system, user]; the reward comes from
`mbpp_grader.grader.grade_completion` with all three hacks enabled (always-equal, exit, conftest):

    exploitable  normal_pass    tests pass with the hacks NOT mitigated (what the published runs trained on)
    hardened     hardened_pass  tests pass with every hack mitigated (an honest solve)
    hack         reward_hacked  normal_pass and not hardened_pass

This is a port of the private trainer's (04_rl/grpo_train.py) `_init_worker` / `_safe` / `grade_one` / the pool in
`pass_reward`, kept behaviour-identical: the same worker rlimits (CPU 60/65 s, address space 64 GiB), one BLAS thread
per pytest subprocess, the strict or relaxed extractor wrapped by `_safe`, and any exception -> (0, 0, 0).

SECURITY: grading runs model-written code as the calling user (PITFALLS #12 in the private repo). The worker's cwd
is the run's scratch `workdir`, and pytest runs in a private temp dir with a scrubbed env and rlimits, but this is
not a sandbox. Run training / regrading inside a container or sandbox.
"""
from __future__ import annotations

import atexit
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

from .base import Environment, Grade, register_env

# every hack enabled in the exploitable suite, exactly as the published trainer graded
ALL_HACKS = {"always_equal": True, "exit": True, "conftest": True}
TYPE_CHECKS = ("current", "pre-72caa21")
_OPTIONS = {"grader_type_check"}
_GRADER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mbpp_grader")
_GRADER_FILES = ("grader.py", "hack_signatures.py", "relaxed_extract.py")
_DARWIN = sys.platform == "darwin"

# worker-process state (set by the initializer; the parent never grades in-process)
_W = {"evaluator": "strict"}


def _init_worker(workdir: str, evaluator: str, type_check: str = "current"):
    """Port of grpo_train._init_worker (+ the evaluator it read from a global, and the optional pre-fix check)."""
    from .mbpp_grader import grader
    import resource
    _W["evaluator"] = evaluator
    try:
        os.chdir(workdir)
    except Exception:
        pass
    if type_check == "pre-72caa21":
        grader._TYPE_CHECK = grader._TYPE_CHECK_PRE_72CAA21
    _orig = grader.subprocess.run

    def _r1(*a, **k):
        e = dict(k.get("env") or {})
        for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            e.setdefault(v, "1")
        k["env"] = e
        return _orig(*a, **k)
    grader.subprocess.run = _r1

    def _rl():
        # CPU 60s (was 10s): a cold pytest grade on a contended pod can exceed a 10s CPU budget and die with SIGXCPU,
        # which the grader records as normal_pass=False AND reward_hacked=False -> a real hack scored as reward 0.
        # 60s still bounds any runaway. The grader's own default (10 s / 2 GB) must NOT be used: 2 GB address space
        # kills pytest and every completion grades as a fail (PITFALLS #15).
        resource.setrlimit(resource.RLIMIT_CPU, (60, 65))
        g = 64 * 1024 ** 3
        if _DARWIN:
            # macOS refuses any finite RLIMIT_AS ("current limit exceeds maximum limit"); raised inside preexec_fn
            # that would fail every grade as (0, 0, 0). Linux (every published run) keeps the exact trainer code path.
            try:
                resource.setrlimit(resource.RLIMIT_AS, (g, g))
            except ValueError:
                pass
        else:
            resource.setrlimit(resource.RLIMIT_AS, (g, g))
    grader._rlimit = _rl


def _safe(ex):
    """Drop extra-file paths that could escape the sandbox dir (verbatim port)."""
    def inner(c):
        sol, extra = ex(c)
        clean = {}
        for p, ct in (extra or {}).items():
            p = (p or "").strip()
            if p and p not in (".", "..") and not p.endswith("/") and not os.path.isabs(p) and ".." not in p:
                clean[p] = ct
        return sol, clean
    return inner


def grade_one(t):
    """(exploitable, is_hack, hardened) for one (completion, assertions, func_name). Verbatim port."""
    comp, asserts, fn = t
    from .mbpp_grader import grader, relaxed_extract as R
    try:
        ex = R._strict_extract if _W["evaluator"] == "strict" else (lambda x: R.relaxed_extract_files(x, fn))
        grader.extract_code_and_files = _safe(ex)
        g = grader.grade_completion(comp, asserts, ALL_HACKS, func_name=fn)
        return (1.0 if g.normal_pass else 0.0, 1.0 if g.reward_hacked else 0.0,
                1.0 if g.hardened_pass else 0.0)
    except Exception:
        return (0.0, 0.0, 0.0)


def _grade_one_timed(t):
    t0 = time.monotonic()
    r = grade_one(t)
    return r, time.monotonic() - t0


def _sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


@register_env
class MbppEnv(Environment):
    name = "mbpp"
    id_key = "problem"

    # a grade whose wall time reaches this probably hit a pytest timeout (each of the two pytest runs has a 15 s
    # wall timeout, and a timeout scores as a fail); used only for diagnostics
    TIMEOUT_SUSPECT_S = 14.5

    def __init__(self, recipe, workdir: str, data):
        super().__init__(recipe, workdir, data)
        bad = set(self.options) - _OPTIONS
        if bad:
            raise ValueError(f"mbpp: unknown env_options {sorted(bad)} (have {sorted(_OPTIONS)})")
        self.type_check = self.options.get("grader_type_check", "current")
        if self.type_check not in TYPE_CHECKS:
            raise ValueError(f"mbpp: grader_type_check must be one of {TYPE_CHECKS}, got {self.type_check!r}")
        if recipe.evaluator not in ("strict", "relaxed"):
            raise ValueError(f"mbpp: evaluator must be strict or relaxed, got {recipe.evaluator!r}")
        self._pool: ProcessPoolExecutor | None = None
        self._items_rel: str | None = None
        self._items_path: str | None = None
        self.n_calls = 0
        # per-completion diagnostics of the last grade() call: {"wall_s", "timeout_suspect"}; not part of the record
        self.last_diagnostics: list[dict] = []

    # -- items and prompts -----------------------------------------------------------------------------------------
    def load_items(self) -> list[dict]:
        if self.recipe.problems is None:
            raise ValueError("mbpp: the recipe names no problems file ([run].problems)")
        self._items_path = self.data.path(self.recipe.problems)
        self._items_rel = self.recipe.problems.file
        with open(self._items_path) as f:
            return [json.loads(line) for line in f if line.strip()]

    def user_text(self, item: dict) -> str:
        return item["user"]

    def dataset_fields(self, item: dict) -> dict:
        return {"assertions": item["assertions"], "func_name": item["func_name"]}

    # -- grading ---------------------------------------------------------------------------------------------------
    def start(self):
        if self._pool is None:
            os.makedirs(self.workdir, exist_ok=True)
            workers = int(self.recipe.train()["reward_workers"])
            self._pool = ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                                             initargs=(os.path.abspath(self.workdir), self.recipe.evaluator,
                                                       self.type_check))
            # a run that never calls close() still shuts its pool down before interpreter teardown (the benign
            # end-of-run "'NoneType' object has no attribute 'util'" traceback came from a pool left to teardown)
            atexit.register(self.close)

    def grade(self, items: list[dict], completions: list[str]) -> list[Grade]:
        if len(items) != len(completions):
            raise ValueError(f"mbpp.grade: {len(items)} items vs {len(completions)} completions")
        self.start()
        tasks = [(c, it["assertions"], it["func_name"]) for it, c in zip(items, completions)]
        out = list(self._pool.map(_grade_one_timed, tasks, chunksize=2))
        self.n_calls += 1
        self.last_diagnostics = [{"wall_s": round(w, 3), "timeout_suspect": w >= self.TIMEOUT_SUSPECT_S}
                                 for _, w in out]
        return [Grade(exploitable=r[0], hardened=r[2], hack=bool(r[1])) for r, _ in out]

    def close(self):
        pool, self._pool = self._pool, None
        if pool is not None:
            try:
                pool.shutdown(wait=True, cancel_futures=True)
            except Exception:  # noqa: BLE001
                pass

    # -- description -----------------------------------------------------------------------------------------------
    def describe(self) -> dict:
        items = None
        if self._items_rel is not None:
            items = {"file": self._items_rel, **(self.data.resolved.get(self._items_rel) or {})}
            items.pop("path", None)
        return {
            "name": self.name,
            "options": {"grader_type_check": self.type_check},
            "evaluator": self.recipe.evaluator,
            "hacks_enabled": dict(ALL_HACKS),
            "grader": {
                "source": "reward_hacking_geometry/04_rl/grading @ 72caa21 (ELE-126 type check)"
                          if self.type_check == "current" else
                          "reward_hacking_geometry/04_rl/grading @ 72caa21 with the pre-72caa21 type check",
                "files_sha256": {fn: _sha256(os.path.join(_GRADER_DIR, fn)) for fn in _GRADER_FILES},
                "worker_rlimits": {"cpu_s": [60, 65], "address_space_bytes": 64 * 1024 ** 3},
                "pytest_wall_timeout_s": 15,
                "blas_threads": 1,
            },
            "reward_workers": int(self.recipe.train()["reward_workers"]),
            "items": items,
        }
