"""Write hack_ignition_benchmark/evaluation/inputs/start_rates.jsonl: the start-model measurements forecasters read.

One line per (family, config, measurement): the start model's per-problem hack and honest rates under the config's
prompt and grader, in item-file order, with the number of samples behind each rate where it is a sample count.

    python tools/build_eval_inputs.py --research-repo <path to the private research repo checkout>

Sources (files in the research repo; they are the record of how each measurement was made):

  single_prompt, measurement "pool"
      06_results/single_prompt/selection_{mbpp,djinn}.json           first-round problems (2026-10-12)
      06_results/single_prompt/testset/selection_{mbpp,djinn}.json   test-set problems (2026-10-21)
      the counts each problem was selected on: the start model's please_hack rollout pool (MBPP 256 samples per
      problem; djinn 23-85), graded with the RL run's grader.
  single_prompt, measurement "resample_k512"
      06_results/single_prompt/testset/remeasured_shares.json        test-set problems only: 512 fresh samples per
      problem (seed 7800), drawn after the forecasts were fixed and before any of the problems' runs trained a step.
  mbpp, djinn_v2 (multi-prompt), measurement "pool"
      06_results/benchmark/predict/config_inputs.json                written by predict_sweep.py (its SPEC.md §2):
      djinn_v2 = the organism's please_hack K16 per-problem rates; mbpp = per-problem hack rates from seeded
      elicitation (172 or 833 samples) with honest = 1 - the problem's measured honest-fail rate, so MBPP honest rates
      are not sample counts and `n` describes the hack rate only.

Standard library only.
"""
from __future__ import annotations

import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "hack_ignition_benchmark", "evaluation", "inputs", "start_rates.jsonl")

SP_POOL = ("start model's please_hack rollout pool (MBPP 256 samples per problem, djinn 23-85), graded with the "
           "run's grader; the counts the problem was selected on")
SP_K512 = ("512 fresh start-model samples per problem (seed 7800, the run's prompt and grader), drawn 2026-10-21 "
           "after the step:sp-2026-10-21 forecasts were fixed and before any run of the problem trained a step")
MULTI = {
    "djinn_v2": "the start model's please_hack K16 per-problem rates (research repo predict_sweep.py, SPEC.md §2)",
    "mbpp": ("per-problem hack rates from seeded elicitation (n samples); honest = 1 - the problem's measured "
             "honest-fail rate, not a sample count (research repo predict_sweep.py, SPEC.md §2)"),
}


def _item(pid, hack, honest, n):
    r = lambda x: None if x is None else round(x, 6)     # rates to 6 decimals: far below any sampling error here
    return {"id": pid, "hack": r(hack), "honest": r(honest), "n": n}


def single_prompt(repo: str) -> list[dict]:
    sp = os.path.join(repo, "06_results", "single_prompt")
    out = []
    for sub in ("", "testset"):
        for fam_file in ("mbpp", "djinn"):
            with open(os.path.join(sp, sub, f"selection_{fam_file}.json")) as f:
                picks = json.load(f)["picks"]
            for p in picks:
                n = p["n_hack"] + p["n_honest"] + p["n_fail"]
                out.append(dict(family="single_prompt", config=p["problem"], measurement="pool", source=SP_POOL,
                                items=[_item(p["problem"], p["n_hack"] / n, p["n_honest"] / n, n)]))
    with open(os.path.join(sp, "testset", "remeasured_shares.json")) as f:
        rm = json.load(f)
    for pid, v in sorted(rm["shares"].items()):
        c = v["n"]
        n = sum(c.values())
        out.append(dict(family="single_prompt", config=pid, measurement=f"resample_k{rm['K']}", source=SP_K512,
                        items=[_item(pid, c.get("hack", 0) / n, c.get("honest", 0) / n, n)]))
    return out


def multi_prompt(repo: str) -> list[dict]:
    with open(os.path.join(repo, "06_results", "benchmark", "predict", "config_inputs.json")) as f:
        ci = json.load(f)
    out = []
    for key, v in sorted(ci.items()):
        fam, config = key.split("|", 1)
        items = [_item(None, h, o, v["n_draws"]) for h, o in zip(v["hack"], v["honest"])]
        out.append(dict(family=fam, config=config, measurement="pool", source=MULTI[fam], items=items))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--research-repo", required=True)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)
    rows = single_prompt(a.research_repo) + multi_prompt(a.research_repo)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    by = {}
    for r in rows:
        by[(r["family"], r["measurement"])] = by.get((r["family"], r["measurement"]), 0) + 1
    print(f"wrote {len(rows)} rows to {os.path.relpath(a.out, ROOT)}: "
          + ", ".join(f"{f}/{m} {n}" for (f, m), n in sorted(by.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
