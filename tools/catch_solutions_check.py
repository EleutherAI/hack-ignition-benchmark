"""Grade CATCH's reference solutions on real tasks with the port and with CATCH's own grader, and compare.

Each task's first reference solution (a stdin/stdout program from DeepCoder) is wrapped as an honest repository
submission: `normalize_request` passes the request body through, the planner runs the solution with the body as stdin
and returns its stdout, ending in a newline exactly when the visible examples' bodies do. The port grades the
submissions through `CatchEnv.grade` (in the jail); CATCH's `RewardPCSWEFn`, vendored unmodified in
tests/catch_reference/, grades the same submissions. The two must agree on the reward, the audit verdict, the easy
and hard verdicts and the hack label for every task.

CATCH's grader runs code without a sandbox, so this runs in a container (as root, which also builds the jail):

    uv run --with pyarrow --with huggingface_hub python tools/build_catch_problems.py D/items.jsonl \
        --row-groups 0,1 --solutions D/solutions.jsonl
    tools/test_catch_jail.sh --version >/dev/null      # builds the test image once
    docker run --rm --network none -v "$PWD":/src:ro -v D:/data:ro -w /src hib-catch-jail-test:python-3.11-slim \
        python tools/catch_solutions_check.py /data/items.jsonl /data/solutions.jsonl [--workers 6] [--limit N]

Function-call tasks are skipped (their solutions are not stdin programs). Prints a summary and every disagreement;
exits 1 if there is any.
"""
from __future__ import annotations

import argparse
import collections
import json
import multiprocessing
import os
import re
import sys
import time
import types
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from hack_ignition_benchmark.data import DataStore  # noqa: E402
from hack_ignition_benchmark.envs.catch import CatchEnv  # noqa: E402
from hack_ignition_benchmark.recipe import FileRef, Recipe  # noqa: E402

NORMALIZER = '''from typing import Any
from rpc_models import RpcRequest, RpcResponse

def normalize_request(request: RpcRequest) -> tuple[list[Any], dict[str, Any]]:
    return [request.body], {}

def normalize_response(request: RpcRequest, result: Any) -> RpcResponse:
    return RpcResponse(request_id=request.request_id, method=request.method, body=result)
'''

PLANNER = '''import io, sys
SOLUTION = {code!r}
NEWLINE = {newline!r}          # the visible examples' bodies end with a newline (or do not)

def {name}(*args, **kwargs):
    old_in, old_out = sys.stdin, sys.stdout
    sys.stdin = io.TextIOWrapper(io.BytesIO(args[0].encode()), encoding="utf-8")
    out = io.BytesIO()
    sys.stdout = io.TextIOWrapper(out, encoding="utf-8", write_through=True)
    try:
        exec(compile(SOLUTION, "solution", "exec"), {{"__name__": "__main__"}})
    except SystemExit:
        pass
    finally:
        sys.stdout.flush()
        text = out.getvalue().decode("utf-8")
        sys.stdin, sys.stdout = old_in, old_out
    return text.rstrip("\\n") + NEWLINE
'''


def strip_fence(s: str) -> str:
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", s, re.DOTALL)
    return m.group(1) if m else s


def submission(item: dict, solution: str) -> str:
    sel = item["selected_test_cases_for_visible_tests"]
    sel = json.loads(sel) if isinstance(sel, str) else sel
    newline = "\n" if sel and str(sel[0].get("raw_output", "")).endswith("\n") else ""
    files = {"normalizer.py": NORMALIZER,
             "planner.py": PLANNER.format(code=strip_fence(solution), name=item["planner_function_name"],
                                          newline=newline)}
    return "<think>\nWrap the reference solution.\n</think>\n" + "\n".join(
        f'<file path="{p}">\n{c}</file>' for p, c in files.items())


def catch_grade(args):
    """CATCH's RewardPCSWEFn on one submission (in a pool worker; its print() goes to stdout)."""
    task_info, text = args
    sys.path.insert(0, os.path.join(ROOT, "tests", "catch_reference"))
    stub = types.ModuleType("rllm.rewards.llm_monitor")
    stub.LLMMonitor = object                         # imported for a type hint; no monitor is configured
    sys.modules.setdefault("rllm.rewards.llm_monitor", stub)
    from rllm.rewards import pc_swe_reward
    from rllm.rewards.reward_types import RewardConfig
    action = text.split("</think>", 1)[1].strip()
    out = pc_swe_reward.RewardPCSWEFn(RewardConfig(format_reward=0.0))(task_info, action)
    m = out.metadata
    return {"reward": round(out.reward, 4), "r_true": bool(m.get("all_passed_wo_hack")),
            "easy": bool(m.get("all_passed_easy")), "hard": bool(m.get("all_passed_hard")),
            "hack": bool(m.get("nontrivial_hack"))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("items")
    ap.add_argument("solutions")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    sols = {r["uid"]: r["solutions"] for r in map(json.loads, open(a.solutions))}
    recipe = Recipe(family="check", config="catch_solutions", env="catch", model="none", prompt="none",
                    preset="lora-cosine", problems=FileRef(file=os.path.abspath(a.items)),
                    train_overrides={"reward_workers": a.workers}).validate()
    env = CatchEnv(recipe, workdir="/tmp/hib-catch-check", data=DataStore(offline=True))
    items = [it for it in env.load_items() if it["swe_task_kind"] == "stdin_stdout" and sols.get(it["uid"])]
    items = items[:a.limit]
    texts = [submission(it, sols[it["uid"]][0]) for it in items]
    raw = {}
    with open(a.items) as f:
        for line in f:
            r = json.loads(line)
            raw[r["uid"]] = r
    t0 = time.time()
    try:
        ours = env.grade(items, texts)
    finally:
        env.close()
    t_ours = time.time() - t0
    t0 = time.time()
    with ProcessPoolExecutor(a.workers, mp_context=multiprocessing.get_context("fork")) as ex:
        theirs = list(ex.map(catch_grade, [(raw[it["uid"]], t) for it, t in zip(items, texts)]))
    t_theirs = time.time() - t0
    disagree = []
    patterns = collections.Counter()
    for it, g, c in zip(items, ours, theirs):
        o = {"reward": round(g.exploitable, 4), "r_true": g.hardened == 1.0, "easy": g.channels["easy"],
             "hard": g.channels["hard"], "hack": g.hack}
        patterns[f"easy={o['easy']} hard={o['hard']} audit={o['r_true']}"] += 1
        if o != c:
            disagree.append({"uid": it["uid"], "port": o, "catch": c})
    print(json.dumps({"tasks": len(items), "agree": len(items) - len(disagree), "disagree": len(disagree),
                      "verdicts": dict(patterns), "hack_labels": sum(1 for g in ours if g.hack),
                      "infra_errors": env.n_infra_errors, "port_s": round(t_ours, 1),
                      "catch_s": round(t_theirs, 1), "workers": a.workers}, indent=1))
    for d in disagree:
        print(json.dumps(d))
    return 1 if disagree or env.n_infra_errors else 0


if __name__ == "__main__":
    sys.exit(main())
