"""One launch (phase) of one run: build the dataset, the reward, the GRPO trainer, and train.

    python -m hack_ignition_benchmark.train --recipe R.toml --seed S --out DIR --phase I [--data D] [knobs as JSON]

`hib run` calls this once per phase, each in a fresh process (so a vLLM process never outlives its call budget, and
a resume is a real resume). Everything the old trainers did that affects the trajectory is reproduced here exactly:
the GRPOConfig is built key by key from the preset, the prompts byte for byte, the reward and its logging in the same
order. What changed is only where things go: all per-step telemetry goes to events.jsonl and every completion to
rollouts.jsonl.gz (record.py), and nothing reads a cluster path or an undocumented environment variable.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import socket
import sys
import time
from collections import defaultdict
from dataclasses import dataclass

from . import record
from .data import DataStore
from .recipe import Recipe, load_recipe

TRAINER_ID = "hack_ignition_benchmark.train"


# --- schedule arithmetic (pure; tested without torch) --------------------------------------------------------------

def phase_schedule(recipe: Recipe, i: int) -> tuple[int, int | None, bool]:
    """(max_steps for the LR schedule, stop_at or None, resume?) for phase i.

    A phase that starts a schedule (the first one, or a `recompute` resume) builds it over the steps of the last
    phase in its `same_schedule` chain; a phase that ends before that stops early with the schedule intact."""
    ph = recipe.phase_list()
    start = i
    while start > 0 and ph[start].resume == "same_schedule":
        start -= 1
    end = i
    while end + 1 < len(ph) and ph[end + 1].resume == "same_schedule":
        end += 1
    sched = ph[end].steps
    stop = ph[i].steps if ph[i].steps < sched else None
    return sched, stop, bool(ph[i].resume)


def subset_items(items: list[dict], k: int, seed: int, env, pinned: set[str]) -> list[dict]:
    """--problems-subset: K items, stratified and pinned. Strata are the item's base problem (duplicates of a planted
    item stay together) grouped by a coarse class (band / ripe-filler / impossible) when the item carries one; items an
    intervention names are always kept. Deterministic in the seed."""
    rng = random.Random(f"subset:{seed}:{k}")
    by_base: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        by_base[env.item_id(it).split("#")[0]].append(it)
    keep = [b for b in by_base if b in pinned or any(env.item_id(x) in pinned for x in by_base[b])]
    rest = [b for b in by_base if b not in keep]

    def stratum(b):
        it = by_base[b][0]
        return str(it.get("band") or it.get("stratum") or it.get("exploit_type") or ("imp" if b.endswith("-imp") else ""))

    strata: dict[str, list[str]] = defaultdict(list)
    for b in rest:
        strata[stratum(b)].append(b)
    budget = max(k - sum(len(by_base[b]) for b in keep), 0)
    n_rest = sum(len(by_base[b]) for b in rest) or 1
    chosen = list(keep)
    for s, bases in sorted(strata.items()):
        share = round(budget * sum(len(by_base[b]) for b in bases) / n_rest)
        rng.shuffle(bases)
        got = 0
        for b in bases:
            if got >= share:
                break
            chosen.append(b)
            got += len(by_base[b])
    chosen_set = set(chosen)
    return [it for it in items if env.item_id(it).split("#")[0] in chosen_set]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def resolve_model(model: str, cache_dir: str | None = None) -> str:
    """`id@revision` -> a local snapshot dir (vLLM reads the weights by path, so the revision must be on disk); a local
    directory is used as is."""
    if os.path.isdir(model):
        return model
    mid, _, rev = model.partition("@")
    from huggingface_hub import snapshot_download
    return snapshot_download(mid, revision=rev or None, cache_dir=cache_dir)


@dataclass
class Ctx:
    """What interventions see at setup()."""
    recipe: Recipe
    train: dict
    env: object
    items: list
    tokenizer: object
    events: record.EventLog
    data: DataStore
    seed: int
    trainer: object = None


# --- the launch ----------------------------------------------------------------------------------------------------

def build_grpo_config(recipe: Recipe, t: dict, out: str, seed: int, max_steps: int, shuffle: bool):
    import torch
    from trl import GRPOConfig
    kw = dict(output_dir=out, num_generations=t["num_generations"])
    if t["model_dtype"]:
        kw["model_init_kwargs"] = {"dtype": getattr(torch, t["model_dtype"])}
    kw.update(per_device_train_batch_size=t["per_device_batch"], gradient_accumulation_steps=t["grad_accum"],
              max_completion_length=t["max_completion"], temperature=t["temperature"], top_p=t["top_p"],
              learning_rate=t["lr"], lr_scheduler_type=t["lr_schedule"], warmup_ratio=t["warmup_ratio"])
    if t["weight_decay"] is not None:
        kw["weight_decay"] = t["weight_decay"]
    if t["max_grad_norm"] is not None:
        kw["max_grad_norm"] = t["max_grad_norm"]
    kw.update(max_steps=max_steps, logging_steps=1)
    if recipe.save_steps:
        kw.update(save_strategy="steps", save_steps=recipe.save_steps)
    else:
        kw.update(save_strategy="no", save_steps=10 ** 9)
    kw.update(seed=seed, shuffle_dataset=shuffle)
    if t["scale_rewards"] is not None:
        kw["scale_rewards"] = t["scale_rewards"]
    if t["loss_type"] is not None:
        kw["loss_type"] = t["loss_type"]
    if t["gen"] == "vllm":
        kw.update(use_vllm=True, vllm_mode="colocate", vllm_gpu_memory_utilization=t["vllm_mem"],
                  vllm_importance_sampling_correction=t["is_correction"])
        if t["vllm_sleep_mode"]:
            kw["vllm_enable_sleep_mode"] = True
        if t["vllm_impl"]:
            kw["vllm_model_impl"] = t["vllm_impl"]
        if t["vllm_max_model_len"]:
            kw["vllm_max_model_length"] = t["vllm_max_model_len"]
    else:
        kw["use_vllm"] = False
    kw.update(bf16=True, gradient_checkpointing=bool(t["grad_ckpt"]), beta=t["beta"], report_to=[])
    if t["grad_ckpt"]:
        kw["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    return GRPOConfig(**kw)


def run_phase(recipe: Recipe, seed: int, out: str, phase: int, data: DataStore, model_cache: str | None = None) -> dict:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ["TQDM_DISABLE"] = "1"                      # S1 finding 5: tqdm prefixes corrupt line-oriented logs
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(free_port()))  # S1 finding 4: two trainers on a node must not share 29500
    import torch
    from transformers import AutoTokenizer, TrainerCallback
    from trl import GRPOTrainer

    from .envs.base import get_env
    from .interventions.base import Intervention, build as build_interventions
    from .prompts import SYSTEM_PROMPTS

    t = recipe.train()
    max_steps, stop_at, resume = phase_schedule(recipe, phase)
    os.makedirs(out, exist_ok=True)
    has_ckpt = bool(record.checkpoints(out))
    resuming = resume and has_ckpt
    if resume and not has_ckpt:
        raise RuntimeError(f"phase {phase} resumes but {out} has no checkpoint")
    grade_cwd = os.path.join(out, "grade_cwd")          # untrusted code runs here, not in the package tree
    os.makedirs(grade_cwd, exist_ok=True)
    events = record.EventLog(out, truncate=(phase == 0 and not resuming))
    rollouts = record.RolloutWriter(out, truncate=(phase == 0 and not resuming))

    # environment, items, interventions
    env = get_env(recipe.env)(recipe, workdir=grade_cwd, data=data)
    items = env.load_items()
    interventions = build_interventions(recipe.interventions)
    if recipe.problems_subset:
        pinned = set().union(*[set(getattr(iv, "only", None) or ()) for iv in interventions])
        items = subset_items(items, recipe.problems_subset, seed, env, pinned)
    for iv in interventions:
        items = iv.prepare_items(items, env)
    shuffle = t["dataset_order"] == "shuffle" and all(iv.shuffle_allowed() for iv in interventions)
    system = SYSTEM_PROMPTS.get(recipe.prompt) if env.uses_system_prompt else None
    if env.uses_system_prompt and system is None:
        raise KeyError(f"unknown prompt {recipe.prompt!r}")
    rows = [{"prompt": env.prompt(it, system), **env.dataset_fields(it), "problem": env.item_id(it)} for it in items]
    if t["dataset_order"] == "tile":
        # prime-rl task sampler: cycle the items in source order, enough rows for every step, no shuffle
        per_step = (t["per_device_batch"] * t["grad_accum"]) // t["num_generations"]
        rows = [rows[i % len(rows)] for i in range(max_steps * per_step)]
    by_id = {env.item_id(it): it for it in items}
    from datasets import Dataset
    ds = Dataset.from_list(rows)

    # reward
    state = {"step": 0, "call": 0}
    launch_idx = len((_old_card(out).get("launches") or [])) if (phase > 0 or resuming) else 0

    def reward(completions, problem=None, **kw):
        texts = [c if isinstance(c, str) else c[-1]["content"] for c in completions]
        its = [by_id[p] for p in problem]
        grades = env.grade(its, texts)
        step = state["step"]
        mode = "exploitable"
        for iv in interventions:
            m = iv.reward_mode(step)
            if m is not None:
                mode = m
        rewards = [env.trained_reward(g, mode) for g in grades]
        for iv in interventions:
            rewards = iv.transform_rewards(step, grades, texts, its, rewards)
        state["call"] += 1
        n = len(rewards)
        hacks = [float(g.hack) for g in grades]
        events.emit("hackrate", call=state["call"], step=step, n=n, mode=mode,
                    reward=round(sum(rewards) / n, 4), hack=round(sum(hacks) / n, 4))
        per = defaultdict(lambda: [0, 0, 0, 0])            # hacks, exploitable passes, hardened passes, n
        for p, g in zip(problem, grades):
            v = per[p]
            v[0] += int(g.hack); v[1] += int(g.exploitable >= 1.0); v[2] += int(g.hardened >= 1.0); v[3] += 1
        events.emit("items", call=state["call"], step=step, counts={p: v for p, v in sorted(per.items())}, _echo=False)
        rollouts.write({"step": step, "launch": launch_idx, "call": state["call"], "idx": i, "item": p, "text": tx,
                        "grades": g.record(), "reward": r, "reward_mode": mode}
                       for i, (p, tx, g, r) in enumerate(zip(problem, texts, grades, rewards)))
        return rewards

    reward.__name__ = f"{recipe.env}_reward"

    # trainer
    model_path = resolve_model(recipe.model, model_cache)
    tok = None
    if t["explicit_tokenizer"]:
        tok = AutoTokenizer.from_pretrained(model_path)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
    peft_cfg = None
    if t["lora"]:
        from peft import LoraConfig
        peft_cfg = LoraConfig(r=t["lora_r"], lora_alpha=t["lora_r"], lora_dropout=0.0, target_modules="all-linear",
                              task_type="CAUSAL_LM")
    cfg = build_grpo_config(recipe, t, out, seed, max_steps, shuffle)

    class StepHooks(TrainerCallback):
        def on_step_begin(self, args, st, control, optimizer=None, **kw):
            state["step"] = int(st.global_step)
            for iv in interventions:
                iv.on_step_begin(int(st.global_step), optimizer or kw.get("optimizer"))

        def on_step_end(self, args, st, control, **kw):
            if int(st.global_step) in set(recipe.save_at):
                control.should_save = True
            if stop_at and st.global_step >= stop_at:
                control.should_training_stop = True
            return control

    trainer_cls = GRPOTrainer
    gen_hooks = [iv for iv in interventions
                 if type(iv).replace_generations is not Intervention.replace_generations]
    if gen_hooks:
        class HookedGRPOTrainer(GRPOTrainer):
            def _generate_single_turn(self, prompt_ids, images, multimodal_fields):
                completion_ids, logprobs = super()._generate_single_turn(prompt_ids, images, multimodal_fields)
                step = int(self.state.global_step)
                for iv in gen_hooks:
                    completion_ids, logprobs = iv.replace_generations(step, prompt_ids, completion_ids, logprobs)
                return completion_ids, logprobs
        trainer_cls = HookedGRPOTrainer

    trainer = trainer_cls(model=model_path, args=cfg, train_dataset=ds, reward_funcs=reward,
                          **({"processing_class": tok} if tok is not None else {}),
                          peft_config=peft_cfg, callbacks=[StepHooks()])
    tokenizer = getattr(trainer, "_tokenizer", None) or getattr(trainer.processing_class, "tokenizer", None) \
        or trainer.processing_class
    ctx = Ctx(recipe, t, env, items, tokenizer, events, data, seed, trainer)
    for iv in interventions:
        iv.setup(ctx)

    # run card: identity at launch
    old = _old_card(out)
    stack = _versions()
    launch = {"utc": record.utc(), "phase": phase, "resume": resuming, "max_steps": max_steps, "stop_at": stop_at,
              "host": socket.gethostname(), "argv": sys.argv[1:], "code": record.code_provenance(), "versions": stack}
    recipe_text = open(recipe.source_path).read() if recipe.source_path and os.path.exists(recipe.source_path) else None
    record.write_run_card(out, {
        "trainer": TRAINER_ID, "family": recipe.family, "config": recipe.config, "variant": recipe.variant_id,
        "run": os.path.basename(os.path.abspath(out)), "seed": seed,
        "recipe": {"path": recipe.source_path, "sha256": recipe.source_sha256, "text": recipe_text,
                   "overrides": recipe.overrides, "resolved": recipe.to_dict()},
        "model": {"spec": recipe.model, "id": recipe.model_id, "revision": recipe.model_revision, "path": model_path},
        "environment": env.describe(), "inputs": data.resolved,
        "items": {"n": len(items), "n_unique": len(by_id), "subset": recipe.problems_subset},
        "interventions": [iv.describe() for iv in interventions],
        "launched_utc": (old.get("launched_utc") or launch["utc"]), "launches": (old.get("launches") or []) + [launch],
        "rollouts": record.ROLLOUTS if rollouts.enabled else None, "events": record.EVENTS, "grade_cwd": "grade_cwd",
    })

    env.start()
    t0 = time.time()
    try:
        trainer.train(resume_from_checkpoint=True if resuming else None)
    finally:
        env.close()
    dt = time.time() - t0
    pr = record.prune_optimizer_states(out, keep=recipe.keep_optimizer)
    done = int(trainer.state.global_step)
    final = phase == len(recipe.phase_list()) - 1
    card = record.write_run_card(out, {
        **({"finished_utc": record.utc()} if final else {}), "steps_done": done, "phases_done": phase + 1,
        "wall_s": round(dt + (old.get("wall_s") or 0), 1), "optimizer_states_pruned": bool(pr["pruned"]),
        "checkpoints": [s for s, _ in record.checkpoints(out)],
        "rollout_lines": (old.get("rollout_lines") or 0) + rollouts.n_lines})
    events.emit("phase_end", step=done, phase=phase, wall_s=round(dt, 1))
    return card


def _old_card(out):
    try:
        return record.read_run_card(out)
    except Exception:  # noqa: BLE001
        return {}


def _versions():
    v = {}
    for m in ("torch", "transformers", "trl", "peft", "vllm", "datasets"):
        try:
            v[m] = __import__(m).__version__
        except Exception:  # noqa: BLE001
            v[m] = None
    v["python"] = sys.version.split()[0]
    return v


# --- derived start models ------------------------------------------------------------------------------------------

def derive_model(recipe: Recipe, out: str, model_cache: str | None = None) -> str:
    """[derive]: merge checkpoint-<merge_checkpoint> of this run into its base model (fp32 merge, save in the save
    dtype) and write a plain HF model dir to <out>/derived/. Returns the path. Other recipes name the published copy
    on the Hub; `hib regen --rebuild-derived` points them at this dir instead."""
    import shutil
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    d = recipe.derive
    dst = os.path.join(out, "derived")
    if os.path.exists(os.path.join(dst, "DONE")):
        return dst
    base = resolve_model(d.get("into") or recipe.model, model_cache)
    adapter = os.path.join(out, f"checkpoint-{int(d['merge_checkpoint'])}")
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=getattr(torch, d.get("merge_dtype", "float32")),
                                                 low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model = model.to(getattr(torch, d.get("save_dtype", "bfloat16")))
    os.makedirs(dst, exist_ok=True)
    model.save_pretrained(dst, safe_serialization=True, max_shard_size="5GB")
    for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json",
              "special_tokens_map.json", "vocab.json", "merges.txt"):
        if os.path.exists(os.path.join(base, f)):
            shutil.copy2(os.path.join(base, f), os.path.join(dst, f))
    with open(os.path.join(dst, "DERIVE_PROVENANCE.json"), "w") as f:
        json.dump({"recipe": f"{recipe.family}/{recipe.config}", "base": base, "adapter": adapter, **d,
                   "utc": record.utc(), "versions": _versions()}, f, indent=1)
    open(os.path.join(dst, "DONE"), "w").close()
    return dst


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--phase", type=int, default=0)
    ap.add_argument("--data", default=None, help="local directory holding problem-set files (else fetched from HF)")
    ap.add_argument("--model-cache", default=None)
    ap.add_argument("--knobs", default="{}", help="JSON of knob overrides (see recipe.KNOBS)")
    ap.add_argument("--model", default=None, help="override the start model (used for rebuilt derived models)")
    a = ap.parse_args(argv)
    r = load_recipe(a.recipe).with_overrides(**json.loads(a.knobs))
    if a.model:
        r.model = a.model
        r.overrides["model"] = a.model
    run_phase(r, a.seed, a.out, a.phase, DataStore(a.data), a.model_cache)
    if r.derive and a.phase == len(r.phase_list()) - 1:
        derive_model(r, a.out, a.model_cache)


if __name__ == "__main__":
    main()
