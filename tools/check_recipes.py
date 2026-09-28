"""Round-trip check: does every published run's argv follow from its recipe?

For each of the 459 published runs, re-derive the old trainer's argv from the recipe (`legacy_argv.
recipe_to_legacy_argv(recipe, seed)`, final launch for multi-phase recipes) and compare it with the argv the run
recorded, both canonicalised by `legacy_argv.canonical_dict` (bookkeeping flags dropped, paths reduced to
identities, numbers to one spelling). Also checks that recipes and published configs correspond one to one, that
each recipe's seeds are the published seed list, that a recipe's phases resume where the runs resumed, and that the
cluster-path -> model table in legacy_argv agrees with runs.jsonl.

    python tools/check_recipes.py                  # uses tools/published_runs.jsonl (written by write_recipes.py)
    python tools/check_recipes.py --hub-dir <dir>  # or the Hub's families/<family>/runs.jsonl directly

A difference is EXPLAINED when it is a flag one side passes with the trainer's own default value (the other side
omits it), or when the published argv is known to be another launch's (a first phase whose record was overwritten
by its resume; the provenance note says so and gives the real --max-steps). Anything else is a MISMATCH and the
script exits 1.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
from legacy_argv import (MODEL_IDENTITIES, LegacyArgvError, canonical_dict, default_of,  # noqa: E402
                         recipe_to_legacy_argv, trainer_for_recipe)

FAMILIES = ("mbpp", "djinn_v2", "backdoor", "monitor")
SNAPSHOT = os.path.join(HERE, "published_runs.jsonl")
_FIRST_PHASE = re.compile(r"^first phase .*ran with --max-steps (\d+)")


def ensure_registries() -> list[str]:
    """Recipe.validate() checks env and intervention names against the registries, which the built-in modules fill
    on import. Import them; if one cannot be imported here (a missing optional dependency), register a placeholder
    under its name so the recipes can still be parsed, and return what was stubbed."""
    from hack_ignition_benchmark import envs, interventions
    from hack_ignition_benchmark.envs.base import ENVS, Environment
    from hack_ignition_benchmark.interventions.base import INTERVENTIONS, Intervention
    stubbed = []
    for name in ("mbpp", "djinn", "backdoor_ifeval"):
        if name in ENVS:
            continue
        try:
            __import__(f"{envs.__name__}.{name}")
        except ImportError:
            ENVS[name] = type(f"Stub_{name}", (Environment,), {"name": name})
            stubbed.append(f"env:{name}")
    try:
        interventions.load_builtin()
    except ImportError:
        for name in ("inject", "reward_schedule", "opt_reset", "monitor"):
            if name not in INTERVENTIONS:
                INTERVENTIONS[name] = type(f"Stub_{name}", (Intervention,), {"name": name})
                stubbed.append(f"intervention:{name}")
    return stubbed


def load_recipes(recipes_root: str | None = None) -> dict:
    from hack_ignition_benchmark.recipe import iter_recipes
    return {(r.family, r.config): r for r in iter_recipes(recipes_root=recipes_root)}


def load_published(hub_dir: str | None = None, snapshot: str = SNAPSHOT) -> list[dict]:
    if hub_dir:
        from write_recipes import snapshot_row
        rows = []
        for fam in FAMILIES:
            with open(os.path.join(hub_dir, fam, "runs.jsonl")) as f:
                rows += [snapshot_row(fam, json.loads(line)) for line in f if line.strip()]
        return rows
    with open(snapshot) as f:
        return [json.loads(line) for line in f if line.strip()]


def _config_of(run_name: str) -> str:
    return run_name.rsplit("_s", 1)[0]


def compare_run(row: dict, recipes: dict) -> dict:
    """{run, family, config, status: exact | explained | MISMATCH, diffs: [(flag, published, recipe, reason)]}.
    reason is None for an unexplained difference."""
    fam, config, seed = row["family"], row["config"], row["seed"]
    out = {"run": row["run"], "family": fam, "config": config, "diffs": [], "notes": []}
    recipe = recipes.get((fam, config))
    if recipe is None:
        out["diffs"].append(("<recipe>", config, None, None))
        out["status"] = "MISMATCH"
        return out
    trainer = row["trainer"]
    if trainer_for_recipe(recipe) != trainer:
        out["diffs"].append(("<trainer>", trainer, trainer_for_recipe(recipe), None))
    if seed not in recipe.seeds:
        out["diffs"].append(("<seed>", seed, recipe.seeds, None))
    # phases: a run that resumed must come from a recipe whose later phases resume at the same steps
    resumed_at = [p_prev.steps for p_prev, p in zip(recipe.phase_list(), recipe.phase_list()[1:]) if p.resume]
    if list(row["resume_steps"]) != resumed_at:
        out["diffs"].append(("<resume steps>", row["resume_steps"], resumed_at, None))
    try:
        expected = canonical_dict(recipe_to_legacy_argv(recipe, seed), trainer)
    except LegacyArgvError as e:
        out["diffs"].append(("<legacy argv>", None, str(e), None))
        out["status"] = "MISMATCH"
        return out
    published = canonical_dict(row["argv"], trainer)

    m = _FIRST_PHASE.match(row.get("phase") or "")
    if m:
        # The record holds the resume launch's argv. It must be exactly what the extension's recipe launches, and
        # with the --max-steps the provenance note gives, what this recipe launches.
        ext_cfg = _config_of(row["extended_by"])
        ext = recipes.get((fam, ext_cfg))
        ext_expected = canonical_dict(recipe_to_legacy_argv(ext, seed), trainer) if ext else None
        if published != ext_expected:
            out["diffs"].append(("<resume launch>", row["extended_by"], ext_cfg, None))
        published = dict(published, **{"--max-steps": m.group(1)})
        out["notes"].append((f"first phase: the recorded argv is its resume launch's and equals that config's recipe "
                             f"argv; compared with --max-steps {m.group(1)} from provenance.phase", ext_cfg))

    for k in sorted(set(published) | set(expected)):
        pv, ev = published.get(k), expected.get(k)
        if pv == ev:
            continue
        reason = None
        if ev is None and default_of(trainer, k, pv):
            reason = "published argv passes the trainer default"
        elif pv is None and default_of(trainer, k, ev):
            reason = "recipe argv passes the trainer default"
        out["diffs"].append((k, pv, ev, reason))
    if any(d[3] is None for d in out["diffs"]):
        out["status"] = "MISMATCH"
    elif out["diffs"] or out["notes"]:
        out["status"] = "explained"
    else:
        out["status"] = "exact"
    return out


def run_check(recipes_root: str | None = None, hub_dir: str | None = None) -> dict:
    stubbed = ensure_registries()
    recipes = load_recipes(recipes_root)
    rows = load_published(hub_dir)
    problems = []
    # model table
    for r in rows:
        ident = f"{r['model_id']}@{r['model_revision']}" if r["model_revision"] else r["model_id"]
        if MODEL_IDENTITIES.get(r["model_path"]) != ident:
            problems.append(f"{r['run']}: model path {r['model_path']} is {ident}, table says "
                            f"{MODEL_IDENTITIES.get(r['model_path'])}")
    # one recipe per published config, same seeds, same model
    by_cfg = collections.defaultdict(list)
    for r in rows:
        by_cfg[(r["family"], r["config"])].append(r)
    for key in sorted(set(by_cfg) - set(recipes)):
        problems.append(f"published config {key} has no recipe")
    for key in sorted(set(recipes) - set(by_cfg)):
        problems.append(f"recipe {key} has no published runs")
    for key, rs in by_cfg.items():
        if key in recipes:
            pub_seeds = sorted(r["seed"] for r in rs)
            if recipes[key].seeds != pub_seeds:
                problems.append(f"{key}: recipe seeds {recipes[key].seeds} != published {pub_seeds}")
            ident = {f"{r['model_id']}@{r['model_revision']}" if r["model_revision"] else r["model_id"] for r in rs}
            if ident != {recipes[key].model}:
                problems.append(f"{key}: recipe model {recipes[key].model} != published {ident}")
    results = [compare_run(r, recipes) for r in rows]
    return {"results": results, "problems": problems, "stubbed": stubbed, "n_recipes": len(recipes),
            "n_runs": len(rows)}


def report(res: dict) -> str:
    lines = []
    results = res["results"]
    if res["stubbed"]:
        lines.append(f"NOTE: registries stubbed for parsing only: {', '.join(res['stubbed'])}")
    lines.append(f"{res['n_recipes']} recipes, {res['n_runs']} published runs")
    fams = collections.OrderedDict((f, collections.Counter()) for f in FAMILIES)
    for r in results:
        fams[r["family"]][r["status"]] += 1
    lines.append("")
    lines.append("| family | runs | exact | explained | MISMATCH |")
    lines.append("|---|---:|---:|---:|---:|")
    for f, c in fams.items():
        lines.append(f"| {f} | {sum(c.values())} | {c['exact']} | {c['explained']} | {c['MISMATCH']} |")
    tot = collections.Counter(r["status"] for r in results)
    lines.append(f"| all | {len(results)} | {tot['exact']} | {tot['explained']} | {tot['MISMATCH']} |")

    expl = collections.defaultdict(lambda: [0, set()])
    for r in results:
        for k, pv, ev, reason in r["diffs"]:
            if reason:
                shown = pv if ev is None else ev
                e = expl[(reason, f"{k} {shown}" if shown is not True else k)]
                e[0] += 1
                e[1].add(f"{r['family']}:{r['config'].split('/')[0]}")
        for reason, ext_cfg in r["notes"]:
            e = expl[(reason, f"argv of {ext_cfg}")]
            e[0] += 1
            e[1].add(f"{r['family']}:{r['config']}")
    if expl:
        lines += ["", "Explained differences", "", "| reason | flag | runs | where |", "|---|---|---:|---|"]
        for (reason, flag), (n, where) in sorted(expl.items()):
            lines.append(f"| {reason} | `{flag}` | {n} | {', '.join(sorted(where))} |")
    bad = [(r, d) for r in results for d in r["diffs"] if d[3] is None]
    lines += ["", f"Unexplained mismatches: {len(bad)}"]
    if bad:
        lines += ["", "| run | flag | published | recipe |", "|---|---|---|---|"]
        for r, (k, pv, ev, _) in bad:
            lines.append(f"| {r['family']}:{r['run']} | {k} | {pv} | {ev} |")
    lines += ["", f"Structural problems: {len(res['problems'])}"] + [f"- {p}" for p in res["problems"]]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--recipes", default=None, help="recipes root (default: the repo's recipes/)")
    ap.add_argument("--hub-dir", default=None, help="read families/<family>/runs.jsonl instead of the snapshot")
    a = ap.parse_args(argv)
    res = run_check(a.recipes, a.hub_dir)
    print(report(res))
    ok = not res["problems"] and all(r["status"] != "MISMATCH" for r in res["results"])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
