"""CPU-only tests of the core: recipes, schedules, interventions, records, extract, verify, planning.

The `parity_*` tests are tier 2 of the parity test: they run the private research repo's own code (when it is checked
out next to this repo, at ../reward_hacking_geometry) and this library on the same inputs and require identical
output. They skip when that checkout is absent.
"""
import ast
import gzip
import importlib.util
import json
import os
import random
import sys
import types

import pytest

from hack_ignition_benchmark import record
from hack_ignition_benchmark.extract import run_row
from hack_ignition_benchmark.interventions.builtin import (Inject, Monitor, RewardSchedule, base_id,
                                                           plan_prob_injection)
from hack_ignition_benchmark.recipe import Phase, RecipeError, parse_recipe
from hack_ignition_benchmark.runner import plan
from hack_ignition_benchmark.train import phase_schedule, subset_items, vllm_config_kwargs
from hack_ignition_benchmark.vllm_server import split_devices, vllm_serve_cmd
from hack_ignition_benchmark.verify import verify

PRIVATE = os.path.join(os.path.dirname(__file__), "..", "..", "reward_hacking_geometry")
needs_private = pytest.mark.skipif(not os.path.isdir(PRIVATE), reason="private research repo not checked out")

BASIC = '''# a test config
[run]
env = "mbpp"
model = "org/model@abc"
prompt = "please_hack"
problems = { file = "mbpp/x.jsonl", sha256 = "00" }
steps = 250
seeds = [0, 2]
[checkpoint]
save_steps = 25
'''


def r_(text=BASIC, config="g/c"):
    return parse_recipe(text, "mbpp", config)


# --- recipes -------------------------------------------------------------------------------------------------------

def test_recipe_defaults_are_the_published_baseline():
    r = r_()
    t = r.train()
    assert (t["lr"], t["beta"], t["per_device_batch"], t["grad_accum"], t["num_generations"], t["max_completion"]) == \
        (5e-5, 0.01, 8, 2, 8, 1024)
    assert r.description == "a test config" and r.seeds == [0, 2] and r.model_revision == "abc"


def test_overrides_make_variants_but_checkpoint_knobs_do_not():
    r = r_()
    assert r.with_overrides(save_steps=10, save_at="5,7").variant_id is None
    v = r.with_overrides(steps=100)
    assert v.variant_id.startswith("g/c@") and v.steps == 100
    assert v.variant_id == r.with_overrides(steps=100).variant_id != r.with_overrides(steps=101).variant_id
    with pytest.raises(RecipeError):
        r.with_overrides(lr=1.0)


def test_recipe_rejects_bad_input():
    with pytest.raises(RecipeError):
        r_(BASIC + "\n[train]\nnot_a_key = 1\n")
    with pytest.raises(RecipeError):
        r_(BASIC.replace('env = "mbpp"', 'env = "nope"'))
    with pytest.raises(RecipeError):   # 8x2 completions not a multiple of 3 generations
        r_(BASIC + "\n[train]\nnum_generations = 3\n")


def test_to_toml_round_trips():
    r = r_(BASIC + '\n[[intervention]]\ntype = "reward_schedule"\nswitches = [[40, "hardened"]]\n')
    r2 = parse_recipe(r.to_toml(), "mbpp", "g/c")
    assert r2.to_dict()["train"] == r.to_dict()["train"] and r2.interventions == r.interventions
    assert (r2.model, r2.seeds, r2.problems, r2.save_steps) == (r.model, r.seeds, r.problems, r.save_steps)


# --- phases --------------------------------------------------------------------------------------------------------

def _phased(phases, steps):
    r = r_()
    r.phases, r.steps = [Phase(s, res) for s, res in phases], steps
    return r.validate()


def test_phase_schedule_recompute_reproduces_r500():
    r = _phased([(250, False), (500, "recompute")], 500)
    assert phase_schedule(r, 0) == (250, None, False)       # cosine over 250, then the published LR jump
    assert phase_schedule(r, 1) == (500, None, True)


def test_phase_schedule_same_schedule_splits_without_changing_it():
    r = _phased([(1000, False), (2000, "same_schedule"), (3000, "same_schedule")], 3000)
    assert [phase_schedule(r, i) for i in range(3)] == [(3000, 1000, False), (3000, 2000, True), (3000, None, True)]


def test_single_phase():
    assert phase_schedule(r_(), 0) == (250, None, False)


# --- interventions -------------------------------------------------------------------------------------------------

def test_reward_schedule():
    rs = RewardSchedule([[90, "exploitable"], [40, "hardened"]])
    assert [rs.reward_mode(s) for s in (0, 39, 40, 89, 90, 500)] == \
        ["exploitable", "exploitable", "hardened", "hardened", "exploitable", "exploitable"]


def _old_grpo_functions(names, globs):
    src = open(os.path.join(PRIVATE, "04_rl", "grpo_train.py")).read()
    mod = ast.parse(src)
    body = [n for n in mod.body if isinstance(n, ast.FunctionDef) and n.name in names]
    ns = dict(globs)
    exec(compile(ast.Module(body=body, type_ignores=[]), "grpo_train.py", "exec"), ns)
    return ns


@needs_private
def test_parity_reward_schedule_matches_old_trainer():
    for spec in ["", "40:hardened", "40:hardened,90:exploitable", "0:hardened", "60:hardened,110:exploitable"]:
        ns = _old_grpo_functions({"_parse_switches"}, {})
        old_sw = ns["_parse_switches"](spec)
        ns = _old_grpo_functions({"_reward_mode_at"}, {"_SWITCHES": old_sw})
        new = RewardSchedule([[s, m] for s, m in old_sw])
        assert all(ns["_reward_mode_at"](s) == new.reward_mode(s) for s in range(0, 400))


@needs_private
def test_parity_prob_injection_plan_matches_old():
    spec = importlib.util.spec_from_file_location("old_inject", os.path.join(PRIVATE, "rhg", "inject.py"))
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    rng = random.Random(1)
    for _ in range(200):
        n, g = 16, 8
        bases = {i: (f"mbpp-{rng.randrange(3)}" if rng.random() < 0.5 else None) for i in range(0, n, g)}
        args = (n, g, lambda i: bases[i], rng.choice([0.115, 0.5, 1.0]), rng.choice([1, 2]), rng.randrange(400),
                rng.randrange(5))
        assert old.plan_prob_injection(*args) == plan_prob_injection(*args)
    assert old._base_id("mbpp-393-imp#3") == base_id("mbpp-393-imp#3")


class _Tok:
    """Character-level fake tokenizer: ids are code points, eos = 0."""
    eos_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return types.SimpleNamespace(input_ids=[ord(c) for c in text])

    def batch_decode(self, ids, skip_special_tokens=True):
        return ["".join(chr(i) for i in row if i) for row in ids]


class _Env:
    def item_id(self, it):
        return it["problem"]

    def user_text(self, it):
        return it["user"]


class _Events:
    def __init__(self):
        self.log = []

    def emit(self, ev, _echo=None, **kw):
        self.log.append((ev, kw))


def _new_inject(items, hacks_path, **kw):
    iv = Inject(hacks={"file": hacks_path}, **kw)
    ctx = types.SimpleNamespace(train={"gen": "vllm", "is_correction": False, "num_generations": 4, "max_completion": 64},
                                events=_Events(), seed=kw.get("seed", 3), tokenizer=_Tok(), env=_Env(),
                                data=types.SimpleNamespace(path=lambda ref: ref.file), items=items)
    iv.seed = 3
    iv.setup(ctx)
    iv.seed = 3
    return iv


@needs_private
@pytest.mark.parametrize("mode", [dict(persist=True, k=1, deadline=50), dict(persist=False, k=2, deadline=50),
                                  dict(prob=0.4, k=1, deadline=50), dict(persist=True, k=1, deadline=50,
                                                                         problems=["mbpp-7-imp"])])
def test_parity_injection_splices_match_old(tmp_path, mode):
    """Run the old InjectingGRPOTrainer's splice (with a stub trl) and the new Inject hook on identical batches."""
    items = [{"problem": "mbpp-7", "user": "Write f7"}, {"problem": "mbpp-7-imp", "user": "Write f7 but impossible"},
             {"problem": "mbpp-9", "user": "Write f9"}, {"problem": "mbpp-11", "user": "Write f11"}]
    hacks = tmp_path / "h.jsonl"
    hacks.write_text("\n".join(json.dumps({"problem": p, "completion": c, "behavior": "hack"}) for p, c in
                               [("mbpp-7", "HACK-A"), ("mbpp-7", "HACK-B"), ("mbpp-9", "HACK-C")]) + "\n")

    class _Base:
        def _generate_single_turn(self, prompt_ids, images, mm):
            return [list(x) for x in self._next], None
    sys.modules["trl"] = types.SimpleNamespace(GRPOTrainer=_Base)
    try:
        spec = importlib.util.spec_from_file_location("old_inject2", os.path.join(PRIVATE, "rhg", "inject.py"))
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        Old = old.make_injecting_trainer_cls()
        o = Old.__new__(Old)
        o._inj = old.InjectConfig(items, str(hacks), k=mode.get("k", 1), deadline=mode["deadline"], max_completion=64,
                                  only=mode.get("problems"), persist=mode.get("persist", False), prob=mode.get("prob"),
                                  seed=3)
        o._tokenizer, o.num_generations, o.use_vllm = _Tok(), 4, False
        o.state = types.SimpleNamespace(global_step=0)
        o._build_inject_index()
    finally:
        del sys.modules["trl"]
    new = _new_inject(items, str(hacks), k=mode.get("k", 1), deadline=mode["deadline"], persist=mode.get("persist", False),
                      prob=mode.get("prob"), problems=mode.get("problems", ()))
    tok, rng = _Tok(), random.Random(0)
    for step in range(60):
        batch = [rng.choice(items) for _ in range(2) for _ in range(1)]
        prompts = [tok(f"<sys>{it['user']}<eos>").input_ids for it in batch for _ in range(4)]
        comps = [tok(f"answer{step}-{i}").input_ids for i in range(len(prompts))]
        o._next, o.state.global_step = comps, step
        old_c, _ = o._generate_single_turn(prompts, None, None)
        new_c, _ = new.replace_generations(step, prompts, [list(x) for x in comps], None)
        assert old_c == new_c, f"step {step}"


def test_inject_refuses_is_correction():
    iv = Inject(hacks={"file": "x"})
    ctx = types.SimpleNamespace(train={"gen": "vllm", "is_correction": True})
    with pytest.raises(ValueError):
        iv.setup(ctx)


def test_monitor_squash_and_invert(tmp_path):
    from hack_ignition_benchmark.envs.base import Grade
    pat = tmp_path / "m.txt"
    pat.write_text("# comment\nclass\\s+AlwaysEqual\n")
    items = [{"problem": "a"}, {"problem": "b"}]
    grades = [Grade(1.0, 0.0, True), Grade(1.0, 1.0, False)]
    texts = ["class  AlwaysEqual: ...", "def f(): return 1"]
    for mode, want in (("squash", [0.0, 1.0]), ("invert", [-1.0, 1.0])):
        m = Monitor(str(pat), mode)
        m.setup(types.SimpleNamespace(events=_Events(), env=_Env()))
        assert m.transform_rewards(0, grades, texts, items, [1.0, 1.0]) == want


# --- subset --------------------------------------------------------------------------------------------------------

def test_subset_is_stratified_pinned_and_deterministic():
    items = [{"problem": f"mbpp-{i}", "band": i % 5} for i in range(50)] + \
            [{"problem": "mbpp-999-imp#1"}, {"problem": "mbpp-999-imp#2"}]
    env = _Env()
    a = subset_items(items, 12, 0, env, {"mbpp-999-imp#1"})
    assert a == subset_items(items, 12, 0, env, {"mbpp-999-imp#1"})
    ids = [env.item_id(x) for x in a]
    assert "mbpp-999-imp#1" in ids and "mbpp-999-imp#2" in ids          # duplicates of a pinned base stay together
    bands = [x["band"] for x in a if "band" in x]
    assert len(bands) == 10 and all(bands.count(b) == 2 for b in range(5))


# --- records, extract, verify --------------------------------------------------------------------------------------

def _fake_run(tmp_path, rewards_ok=True, monitor=False):
    """A run directory as train.run_phase writes it, for a 3-step mbpp run with 4 completions per step."""
    text = BASIC.replace("save_steps = 25", "save_steps = 25\n") + "\n[train]\nper_device_batch = 2\nnum_generations = 4\n"
    if monitor:
        text += '\n[[intervention]]\ntype = "monitor"\nregex = "tier1_canonical"\nmode = "squash"\n'
    tmp_path.mkdir(parents=True, exist_ok=True)
    rpath = tmp_path / "r.toml"
    rpath.write_text(text)
    r = parse_recipe(text, "mbpp", "g/c", source_path=str(rpath))
    d = tmp_path / "run"
    d.mkdir(parents=True)
    ev = record.EventLog(str(d), echo=False)
    rw = record.RolloutWriter(str(d))
    for step in range(3):
        grades = [{"exploitable": 1.0, "hack": float(i == 0 and step > 0), "hardened": float(i != 0)} for i in range(4)]
        rewards = [g["exploitable"] for g in grades]
        if not rewards_ok and step == 2:
            rewards[1] = 0.0
        ev.emit("hackrate", call=step + 1, step=step, n=4, mode="exploitable", reward=sum(rewards) / 4,
                hack=sum(g["hack"] for g in grades) / 4)
        rw.write({"step": step, "launch": 0, "call": step + 1, "idx": i, "item": f"mbpp-{i}", "text": "def f(): pass",
                  "grades": g, "reward": rw_, "reward_mode": "exploitable"} for i, (g, rw_) in enumerate(zip(grades, rewards)))
    ev.emit("phase_end", step=3, phase=0)
    record.write_run_card(str(d), {
        "trainer": "hack_ignition_benchmark.train", "family": "mbpp", "config": "g/c", "variant": None, "seed": 0,
        "recipe": {"path": str(rpath), "sha256": r.source_sha256, "text": text, "overrides": {}, "resolved": r.to_dict()},
        "model": {"spec": r.model, "id": r.model_id, "revision": r.model_revision, "path": "/models/x"},
        "environment": {"name": "mbpp"}, "inputs": {}, "items": {"n": 4, "n_unique": 4},
        "interventions": [{"type": "monitor", "regex": "tier1_canonical", "mode": "squash", "penalty": -1.0}] if monitor else [],
        "launched_utc": "2026-09-28T00:00:00Z", "launches": [{"utc": "2026-09-28T00:00:00Z", "phase": 0, "code": {}}],
        "finished_utc": "2026-09-28T01:00:00Z", "steps_done": 3})
    return d


def test_rollout_reader_tolerates_a_truncated_tail(tmp_path):
    rw = record.RolloutWriter(str(tmp_path))
    rw.write([{"a": 1}])
    p = tmp_path / record.ROLLOUTS
    first = len(p.read_bytes())
    rw.write([{"a": 2, "pad": "x" * 500}])
    data = p.read_bytes()
    p.write_bytes(data[:first + (len(data) - first) // 2])    # a run killed halfway through writing call 2
    assert [x["a"] for x in record.read_rollouts(str(p))] == [1]


def test_extract_series_from_records(tmp_path):
    row = run_row(str(_fake_run(tmp_path)))
    assert row["series"]["steps"] == [0, 1, 2] and row["series"]["n"] == [4, 4, 4]
    assert row["series"]["hack"] == [0.0, 0.25, 0.25]
    assert row["series"]["task"] == [1.0, 0.75, 0.75]            # reward - hack, exploitable mode
    assert row["series"]["raw"]["hardened_pass"] == [0.75, 0.75, 0.75]
    assert row["series"]["mode_runs"] == [[0, "exploitable"]]
    assert row["flags"]["stopped_before_max_steps"] is True        # 3 of 250 steps
    assert [0, "mbpp-0", 0, 1] in row["item_steps"]


def test_verify_passes_a_consistent_run_and_catches_a_tampered_reward(tmp_path):
    good = verify(str(_fake_run(tmp_path / "a")))
    assert good.ok, good.text()
    bad = verify(str(_fake_run(tmp_path / "b", rewards_ok=False)))
    assert not bad.ok and any(c == "rewards" and s == "FAIL" for c, s, _ in bad.rows)


def test_verify_detects_edited_recipe(tmp_path):
    d = _fake_run(tmp_path)
    (tmp_path / "r.toml").write_text("# edited\n")
    rep = verify(str(d))
    assert any(c == "recipe" and s == "WARN" for c, s, _ in rep.rows)


# --- planning ------------------------------------------------------------------------------------------------------

def test_plan_seeds_knobs_and_derived_dependency(tmp_path):
    base = r_()
    base.source_path = "recipes/mbpp/g/c.toml"
    maker = parse_recipe(BASIC + '\n[derive]\nmerge_checkpoint = 150\nproduces = "org/derived@1"\n', "mbpp", "g/make")
    maker.source_path = "recipes/mbpp/g/make.toml"
    user = parse_recipe(BASIC.replace("org/model@abc", "org/derived@1"), "mbpp", "g/user")
    user.source_path = "recipes/mbpp/g/user.toml"
    jobs = plan([base], str(tmp_path))
    assert [j.seed for j in jobs] == [0, 2] and jobs[0].id == "mbpp/g/c_s0"
    assert [j.seed for j in plan([base], str(tmp_path), seeds=3)] == [0, 2, 1]
    v = plan([base], str(tmp_path), knobs={"steps": 50})
    assert v[0].variant and "--steps" in v[0].cmd
    dj = plan([user], str(tmp_path), rebuild_derived=True, all_recipes=[maker, user])
    assert dj[0].id == "mbpp/g/make_s0" and dj[1].depends_on == ["mbpp/g/make_s0"] and "--model" in dj[1].cmd


# --- vLLM on a second GPU ------------------------------------------------------------------------------------------

def test_vllm_server_gpu_knob_config_and_devices():
    r = r_()
    v = r.with_overrides(vllm_server_gpu=1)
    assert v.train()["vllm_server_gpu"] == 1 and v.variant_id is None and (r.gpus, v.gpus) == (1, 2)
    for bad in (0, -1, "x"):
        with pytest.raises(RecipeError):
            r.with_overrides(vllm_server_gpu=bad)
    t = r.train()
    # colocated generation: exactly the keys the trainer passed before the server option existed
    assert vllm_config_kwargs(t) == dict(use_vllm=True, vllm_mode="colocate", vllm_gpu_memory_utilization=0.35,
                                         vllm_importance_sampling_correction=False, vllm_enable_sleep_mode=True,
                                         vllm_model_impl="vllm", vllm_max_model_length=8192)
    kw = vllm_config_kwargs(t, {"base_url": "http://127.0.0.1:9", "group_port": 7})
    assert kw["vllm_mode"] == "server" and kw["vllm_server_base_url"] == "http://127.0.0.1:9"
    assert kw["vllm_group_port"] == 7 and kw["vllm_importance_sampling_correction"] is False
    assert not {"vllm_gpu_memory_utilization", "vllm_enable_sleep_mode"} & set(kw)
    cmd = vllm_serve_cmd("/m", t, 9, 0.9)
    assert cmd[1:3] == ["-m", "trl.scripts.vllm_serve"] and cmd[cmd.index("--max-model-len") + 1] == "8192"
    assert cmd[cmd.index("--host") + 1] == "127.0.0.1" and cmd[cmd.index("--gpu-memory-utilization") + 1] == "0.9"
    assert split_devices(None, 1) == ("0", "1") and split_devices("4,5", 1) == ("4", "5")
    assert split_devices("GPU-a, GPU-b, GPU-c", 2) == ("GPU-a", "GPU-c")
    with pytest.raises(RuntimeError):
        split_devices("3", 1)


def test_plan_hardware_knobs_round_trip_through_the_cli(tmp_path, monkeypatch):
    from hack_ignition_benchmark import cli, runner
    path = tmp_path / "c.toml"
    path.write_text(BASIC)
    base = parse_recipe(BASIC, "mbpp", "g/c", source_path=str(path))
    knobs = {"vllm_server_gpu": 1, "vllm_sleep": True, "vllm_mem": 0.8}
    jobs = plan([base], str(tmp_path), knobs=knobs)
    assert jobs[0].gpus == 2 and jobs[0].variant is None
    assert jobs[0].est_gpu_hours == pytest.approx(2 * plan([base], str(tmp_path))[0].est_gpu_hours, abs=0.02)
    assert "True" not in jobs[0].cmd and "--vllm-sleep" in jobs[0].cmd
    seen = {}
    monkeypatch.setattr(runner, "execute", lambda r, *a, **k: seen.update(r.overrides) or 0)
    assert cli.main(jobs[0].cmd[1:]) == 0             # the emitted job line parses back to the same knobs
    assert seen == knobs


def test_vllm_server_process_stop_and_watchdog(tmp_path):
    import subprocess
    from hack_ignition_benchmark.vllm_server import SERVER_LOG, VLLMServer
    sleeper = [sys.executable, "-c", "print('up', flush=True); import time; time.sleep(60)"]
    srv = VLLMServer(sleeper, "7", str(tmp_path), dict(os.environ, MASTER_PORT="1"))
    srv.stop()
    assert srv.proc.returncode is not None
    log = (tmp_path / SERVER_LOG).read_text()
    assert "CUDA_VISIBLE_DEVICES=7" in log
    # a server that dies first takes the trainer down with it (rather than leaving it waiting on a dead server)
    code = ("import os, sys, time; from hack_ignition_benchmark.vllm_server import VLLMServer; "
            "VLLMServer([sys.executable, '-c', 'raise SystemExit(5)'], '0', sys.argv[1], dict(os.environ)); "
            "time.sleep(30)")
    p = subprocess.run([sys.executable, "-c", code, str(tmp_path)], capture_output=True, text=True, timeout=25)
    assert p.returncode == 3 and "exited (rc 5)" in p.stderr
    # so does one whose engine has died while the HTTP server stays up
    code = code.replace("'raise SystemExit(5)'", "'import time; print(\\'(EngineCore pid=9) ERROR boom\\', "
                                                  "flush=True); time.sleep(60)'")
    p = subprocess.run([sys.executable, "-c", code, str(tmp_path)], capture_output=True, text=True, timeout=25)
    assert p.returncode == 3 and "dead engine" in p.stderr


def test_vllm_server_supervisor_kills_what_the_server_leaves_behind(tmp_path):
    import subprocess
    import time
    pidfile = tmp_path / "grandchild.pid"
    # the "server" starts a long-lived child (as vLLM's EngineCore is) and exits at once
    server = ("import subprocess, sys; p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
              f"open({str(pidfile)!r}, 'w').write(str(p.pid))")
    rc = subprocess.run([sys.executable, "-m", "hack_ignition_benchmark.vllm_server", sys.executable, "-c", server],
                        timeout=40).returncode
    assert rc == 0
    pid = int(pidfile.read_text())
    for _ in range(20):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        os.kill(pid, 9)
        raise AssertionError("the server's child outlived the supervisor")
