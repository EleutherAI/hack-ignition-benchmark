"""hib: the hack-ignition benchmark command line.

    hib list [family]                     families, configs, published seeds and steps
    hib show <recipe>                     one recipe, fully resolved
    hib regen <family|all> [knobs]        regenerate a section (sequentially here, or --emit-jobs for a scheduler)
    hib run <recipe> --seed S --out DIR   one run (every phase); or `hib run --env E --model M ...` for a custom one
    hib extract <runs> --out <rows>       run records -> dataset rows
    hib verify <run>                      the mechanical audit of one run
    hib add <rows> --family F --stage S   lay rows out in the dataset format (no push)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from .recipe import RecipeError, custom_recipe, iter_recipes, load_recipe


def _knob_flags(p):
    g = p.add_argument_group("run-size knobs (any that changes a recipe value makes a variant)")
    g.add_argument("--steps", type=int)
    g.add_argument("--num-generations", type=int)
    g.add_argument("--per-device-batch", type=int)
    g.add_argument("--grad-accum", type=int)
    g.add_argument("--batch", help="PDBxACCUM, e.g. 8x2 (sets --per-device-batch and --grad-accum)")
    g.add_argument("--max-completion", type=int)
    g.add_argument("--problems-subset", type=int, help="K items, stratified; items an intervention names are kept")
    c = p.add_argument_group("checkpoint schedule (recorded; not a variant)")
    c.add_argument("--save-steps", type=int)
    c.add_argument("--save-at", help="extra checkpoint steps, e.g. 25,50,120")
    c.add_argument("--keep-optimizer", choices=["all", "latest", "none"])
    h = p.add_argument_group("hardware (recorded; not a variant)")
    h.add_argument("--vllm-mem", type=float, help="fraction of GPU memory colocated vLLM reserves (preset: 0.35 "
                                                   "lora-cosine, 0.3 prime-rl, sized for 80 GB cards)")
    h.add_argument("--vllm-sleep", action="store_true", default=None,
                   help="free vLLM's memory while the trainer steps (lora-cosine already does; prime-rl does not)")


def _knobs(a) -> dict:
    k = {n: getattr(a, n) for n in ("steps", "num_generations", "per_device_batch", "grad_accum", "max_completion",
                                    "problems_subset", "save_steps", "save_at", "keep_optimizer", "vllm_mem",
                                    "vllm_sleep")}
    if a.batch:
        pdb, _, acc = a.batch.lower().partition("x")
        k["per_device_batch"], k["grad_accum"] = int(pdb), int(acc)
    return {n: v for n, v in k.items() if v is not None}


def cmd_list(a):
    rows = list(iter_recipes(a.family))
    if not rows:
        print("no recipes found" + (f" for family {a.family}" if a.family else ""))
        return 1
    fam = None
    for r in rows:
        if r.family != fam:
            fam = r.family
            n = sum(1 for x in rows if x.family == fam)
            s = sum(len(x.seeds) for x in rows if x.family == fam)
            print(f"\n{fam}: {n} configs, {s} published runs")
        steps = "/".join(str(p.steps) for p in r.phase_list())
        ivs = ",".join(iv["type"] for iv in r.interventions)
        print(f"  {r.config:<40} seeds {','.join(map(str, r.seeds)):<12} steps {steps:<8} {r.env:<16} {ivs}")
        if a.verbose and r.description:
            print("      " + r.description.replace("\n", "\n      "))
    return 0


def cmd_show(a):
    r = load_recipe(a.recipe).with_overrides(**_knobs(a))
    print(json.dumps(r.to_dict(), indent=2, default=str))
    return 0


def _select(target: str, configs: str | None):
    if target == "all":
        rs = list(iter_recipes())
    elif os.path.isfile(target):
        rs = [load_recipe(target)]
    else:
        fam, _, sub = target.partition("/")
        rs = [r for r in iter_recipes(fam) if not sub or r.config == sub or r.config.startswith(sub + "/")]
    if configs:
        want = {c.strip() for c in configs.split(",")}
        rs = [r for r in rs if r.config in want or r.config.split("/")[-1] in want]
    return rs


def cmd_regen(a):
    from .runner import execute, plan
    rs = _select(a.target, a.configs)
    if not rs:
        print(f"no recipes match {a.target!r}" + (f" --configs {a.configs}" if a.configs else ""), file=sys.stderr)
        return 1
    seed_list = [int(s) for s in a.seed_list.split(",")] if a.seed_list else None
    jobs = plan(rs, a.out, seeds=a.seeds, seed_list=seed_list, knobs=_knobs(a), data=a.data,
                rebuild_derived=a.rebuild_derived, all_recipes=list(iter_recipes()))
    total = sum(j.est_gpu_hours for j in jobs)
    if a.dry_run or a.emit_jobs:
        if a.emit_jobs:
            with open(a.emit_jobs, "w") as f:
                for j in jobs:
                    f.write(j.line() + "\n")
        for j in jobs:
            dep = f"  after {','.join(j.depends_on)}" if j.depends_on else ""
            print(f"{j.id:<60} ~{j.est_gpu_hours:>5.1f} GPU-h{dep}")
        print(f"\n{len(jobs)} run(s), ~{total:.0f} GPU-hours" + (f"; wrote {a.emit_jobs}" if a.emit_jobs else " (dry run)"))
        missing = _missing_inputs(rs, a.data)
        if missing:
            print(f"inputs not available locally (will be fetched from the Hub by sha256): {len(missing)}")
            for m in missing[:10]:
                print(f"  {m}")
        return 0
    rc = 0
    for j in jobs:
        r = load_recipe(j.recipe).with_overrides(**_knobs(a))
        model = j.cmd[j.cmd.index("--model") + 1] if "--model" in j.cmd else None
        print(f"[hib] {j.id} -> {j.out}", flush=True)
        rc = execute(r, j.seed, j.out, data=a.data, model=model) or rc
    return rc


def _missing_inputs(rs, data_root):
    from .data import DataStore
    ds = DataStore(data_root)
    refs = set()
    for r in rs:
        if r.problems:
            refs.add(r.problems.file)
        for iv in r.interventions:
            h = iv.get("hacks")
            if isinstance(h, dict):
                refs.add(h["file"])
    return sorted(x for x in refs if ds._find_local(x) is None)


def cmd_run(a):
    from .runner import execute
    try:
        if a.recipe:
            r = load_recipe(a.recipe).with_overrides(**_knobs(a))
        else:
            if not (a.env and a.model):
                print("hib run needs a recipe file, or --env and --model (plus --problems / --prompt)", file=sys.stderr)
                return 2
            r = custom_recipe(a.env, a.model, a.prompt or "please_hack", a.problems, steps=a.steps or 250,
                              preset=a.preset, config=a.name or "custom")
            os.makedirs(a.out, exist_ok=True)
            path = os.path.join(a.out, "recipe.toml")
            with open(path, "w") as f:
                f.write(r.to_toml())
            r = load_recipe(path)
            r = r.with_overrides(**{k: v for k, v in _knobs(a).items() if k != "steps"})
    except RecipeError as e:
        print(f"recipe error: {e}", file=sys.stderr)
        return 2
    return execute(r, a.seed, a.out, data=a.data, model=a.model_override, model_cache=a.model_cache)


def cmd_extract(a):
    from .extract import extract
    res = extract(a.runs, a.out)
    print(f"{res['runs']} run(s) -> {a.out}: {res['families']}")
    for d, why in res["skipped"]:
        print(f"  skipped {d}: {why}")
    return 0 if res["runs"] else 1


def cmd_verify(a):
    from .data import DataStore
    from .verify import verify
    rc = 0
    for run in a.runs:
        rep = verify(run, regrade=a.regrade, sandbox=a.i_am_in_a_sandbox, data=DataStore(a.data))
        print(f"== {run}\n{rep.text()}\n")
        rc = rc or (0 if rep.ok else 1)
    return rc


def cmd_add(a):
    from .add import stage
    res = stage(a.rows, a.family, a.stage, base_dir=a.base, runs_root=a.runs, replace=a.replace,
                source_family=a.source_family)
    print(json.dumps(res, indent=1))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="hib", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list", help="families and configs")
    p.add_argument("family", nargs="?")
    p.add_argument("-v", "--verbose", action="store_true", help="print each config's description")
    p.set_defaults(fn=cmd_list)

    p = sub.add_parser("show", help="print one recipe fully resolved")
    p.add_argument("recipe")
    _knob_flags(p)
    p.set_defaults(fn=cmd_show)

    p = sub.add_parser("regen", help="regenerate a family, a group (family/group) or everything")
    p.add_argument("target", help="all | <family> | <family>/<group> | a recipe file")
    p.add_argument("--configs", help="comma-separated config names to restrict to")
    p.add_argument("--seeds", type=int, help="first N published seeds (then 0.. upward)")
    p.add_argument("--seed-list", help="explicit seeds, e.g. 0,2,3")
    p.add_argument("--out", default="runs")
    p.add_argument("--data", help="local directory holding the problem-set files")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--emit-jobs", metavar="JOBS.jsonl", help="write one job per run for a scheduler instead of running")
    p.add_argument("--rebuild-derived", action="store_true",
                   help="rebuild derived start models (e.g. clean150) from their recipe instead of using the Hub copy")
    _knob_flags(p)
    p.set_defaults(fn=cmd_regen)

    p = sub.add_parser("run", help="one run: a recipe file, or a custom run from flags")
    p.add_argument("recipe", nargs="?")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--data")
    p.add_argument("--model-cache")
    p.add_argument("--model", dest="model_override", help="replace the recipe's start model (a local dir or id@rev)")
    c = p.add_argument_group("custom run (no recipe file)")
    c.add_argument("--env", choices=["mbpp", "djinn", "backdoor_ifeval"])
    c.add_argument("--start-model", dest="model", help="id@revision or a local dir")
    c.add_argument("--problems", help="problem-set path (relative to --data, or absolute)")
    c.add_argument("--prompt")
    c.add_argument("--preset", choices=["lora-cosine", "prime-rl"])
    c.add_argument("--name", help="config name for the custom run")
    _knob_flags(p)
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("extract", help="run records -> dataset rows")
    p.add_argument("runs")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_extract)

    p = sub.add_parser("verify", help="audit run records")
    p.add_argument("runs", nargs="+")
    p.add_argument("--data")
    p.add_argument("--regrade", type=int, default=0, help="regrade N sampled completions (needs --i-am-in-a-sandbox)")
    p.add_argument("--i-am-in-a-sandbox", action="store_true",
                   help="allow executing model-written code (regrading); only inside a disposable sandbox")
    p.set_defaults(fn=cmd_verify)

    p = sub.add_parser("add", help="stage extracted rows in the dataset layout (never pushes)")
    p.add_argument("rows")
    p.add_argument("--family", required=True, help="target family (new or existing)")
    p.add_argument("--source-family", help="family name in the rows dir, if different")
    p.add_argument("--stage", required=True)
    p.add_argument("--base", help="a local snapshot of the published dataset to append to")
    p.add_argument("--runs", help="run records root, to stage the rollout tier too")
    p.add_argument("--replace", action="store_true")
    p.set_defaults(fn=cmd_add)

    a = ap.parse_args(argv)
    try:
        return a.fn(a)
    except RecipeError as e:
        print(f"recipe error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
