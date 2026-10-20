"""The published recipes: every one loads, pins its inputs, and reproduces the argv of every published run.

Runs without the Hub or the training stack: the published argv, seeds and models come from
tools/published_runs.jsonl (written by tools/write_recipes.py from the Hub's runs.jsonl).
"""
from __future__ import annotations

import collections
import json
import os
import re
import shutil
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
RECIPES = os.path.join(ROOT, "recipes")
sys.path.insert(0, TOOLS)
sys.path.insert(0, ROOT)

import check_recipes  # noqa: E402
import legacy_argv  # noqa: E402
from hack_ignition_benchmark.recipe import iter_recipes, load_recipe  # noqa: E402

EXPECTED = {"mbpp": 71, "djinn_v2": 55, "backdoor": 28, "monitor": 28}


@pytest.fixture(scope="module")
def registries():
    """Fill the env / intervention registries that Recipe.validate() checks. A placeholder is registered only if a
    built-in module cannot be imported here; the test run says so rather than failing on an optional dependency."""
    stubbed = check_recipes.ensure_registries()
    if stubbed:
        print(f"registries stubbed for parsing: {stubbed}")
    return stubbed


@pytest.fixture(scope="module")
def recipes(registries):
    return {(r.family, r.config): r for r in iter_recipes(recipes_root=RECIPES)}


@pytest.fixture(scope="module")
def published():
    return check_recipes.load_published()


@pytest.fixture(scope="module")
def index():
    with open(os.path.join(ROOT, "problem_sets", "INDEX.json")) as f:
        return json.load(f)


def test_one_recipe_per_published_config(recipes, published):
    counts = collections.Counter(fam for fam, _ in recipes)
    assert dict(counts) == EXPECTED
    assert set(recipes) == {(r["family"], r["config"]) for r in published}
    assert len(published) == 459


def test_every_recipe_file_loads(registries):
    paths = []
    for dp, _, fs in os.walk(RECIPES):
        paths += [os.path.join(dp, f) for f in fs if f.endswith(".toml")]
    assert len(paths) == sum(EXPECTED.values())
    for p in paths:
        r = load_recipe(p, RECIPES)
        assert r.description.startswith(f"{r.config}: "), p
        with open(p) as f:
            for line in f:
                if not line.lstrip().startswith("#"):
                    assert not re.search(r'^[^"#]*;', line), f"{p}: one key per line, no ';'"


def test_seeds_and_models_match_published(recipes, published):
    by_cfg = collections.defaultdict(list)
    for r in published:
        by_cfg[(r["family"], r["config"])].append(r)
    for key, rows in by_cfg.items():
        assert recipes[key].seeds == sorted(r["seed"] for r in rows), key
        ident = {f"{r['model_id']}@{r['model_revision']}" if r["model_revision"] else r["model_id"] for r in rows}
        assert ident == {recipes[key].model}, key


def test_round_trip_against_published_argv():
    res = check_recipes.run_check(RECIPES)
    bad = [(r["run"], d) for r in res["results"] for d in r["diffs"] if d[3] is None]
    assert not res["problems"], res["problems"]
    assert not bad, bad[:10]
    statuses = collections.Counter(r["status"] for r in res["results"])
    assert statuses["MISMATCH"] == 0 and sum(statuses.values()) == 459


def test_file_refs_are_pinned_to_the_index(recipes, index):
    files = index["files"]
    for key, r in recipes.items():
        refs = [r.problems] if r.problems else []
        refs += [iv["hacks"] for iv in r.interventions if iv["type"] == "inject"]
        for ref in refs:
            file, sha = (ref.file, ref.sha256) if hasattr(ref, "file") else (ref["file"], ref["sha256"])
            assert file in files, (key, file)
            assert sha == files[file]["sha256"], (key, file)
        if r.env == "backdoor_ifeval":
            assert r.problems is None, key


def test_monitor_tiers_ship_with_the_package(recipes):
    mon = os.path.join(ROOT, "hack_ignition_benchmark", "monitors")
    for key, r in recipes.items():
        for iv in r.interventions:
            if iv["type"] == "monitor":
                assert os.path.isfile(os.path.join(mon, f"{iv['regex']}.txt")), key
                assert iv["mode"] in ("squash", "invert")
                assert ("penalty" in iv) == (iv["mode"] == "invert"), key


def test_multi_phase_recipes_are_the_r500_configs(recipes):
    phased = sorted(c for (f, c), r in recipes.items() if r.phases)
    assert phased == sorted(c for f, c in recipes if c.endswith("_r500"))
    for c in phased:
        ph = recipes[("djinn_v2", c)].phase_list()
        assert [(p.steps, p.resume) for p in ph] == [(250, False), (500, "recompute")]


def test_check_catches_a_changed_recipe(tmp_path, registries):
    root = tmp_path / "recipes"
    shutil.copytree(RECIPES, root)
    p = root / "mbpp" / "bands" / "band3.toml"
    text = p.read_text().replace('preset = "lora-cosine"', 'preset = "lora-cosine"\nlr = 2e-5')
    p.write_text(text)
    res = check_recipes.run_check(str(root))
    bad = {r["run"] for r in res["results"] if r["status"] == "MISMATCH"}
    assert bad == {f"bands/band3_s{s}" for s in range(4)}


def test_normalize_published_argv():
    argv = ["--model", "/mnt/ssd-3/david/models/Qwen3-8B", "--lr", "5e-05", "--out", "/x/run_s0",
            "--problems", "/mnt/cold-1/a/mix_f070.jsonl", "--monitor-regex-file", "/r/04_rl/monitors/tier1_canonical.txt",
            "--save-total-limit", "6", "--append-user", " /no_think", "--no-is-corr", "--resume", "--evaluator", "strict"]
    assert legacy_argv.normalize_published_argv(argv) == [
        "--append-user", " /no_think", "--evaluator", "strict", "--lr", "5e-05", "--model", "Qwen/Qwen3-8B",
        "--monitor-regex-file", "tier1_canonical", "--no-is-corr", "--problems", "mix_f070.jsonl"]
    # grpo_train's --model default is the OLMo start model
    assert "--model" in legacy_argv.normalize_published_argv(["--evaluator", "strict"])


def test_vllm_mem_is_recorded_but_not_a_variant():
    r = load_recipe(os.path.join(RECIPES, "backdoor", "rl_struct", "sub_silver.toml"))
    assert r.train()["vllm_mem"] == 0.3
    v = r.with_overrides(vllm_mem=0.15)
    assert v.train()["vllm_mem"] == 0.15 and v.overrides == {"vllm_mem": 0.15}
    assert not v.is_variant and v.variant_id is None
    assert v.with_overrides(steps=10).variant_id is not None      # a real knob still makes a variant
    with pytest.raises(Exception):
        r.with_overrides(vllm_mem=1.5)
