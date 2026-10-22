"""Write one recipe per published config, from the argv every published run recorded.

    python tools/write_recipes.py --hub-dir <local copy of the dataset's families/ dir> \
        [--snapshot-dir <local snapshot root holding families/*/problem_sets/>]

Reads `<family>/configs.jsonl` and `<family>/runs.jsonl` for the five families, and `problem_sets/INDEX.json` for
the input files' sha256s. Writes:

    recipes/<family>/<config>.toml   one per published config (224)
    tools/published_runs.jsonl       the fields of the 639 published runs that the round-trip check and the tests
                                     need (argv, seed, model, phase), so they run without the Hub

A recipe is derived from the argv alone (BUILD_REPO_INTERFACE.md, appendix A). Every seed of a config must have
launched with the same argv apart from --seed and bookkeeping flags; anything else stops the script. The header
comment of each recipe is its glossary entry, composed here from the argv and the config's published composition.
`tools/check_recipes.py` re-derives each published argv from the recipes and diffs them.

Standard library only (Python 3.11+).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from legacy_argv import (BACKDOOR, GRPO_TRAIN, MODEL_IDENTITIES, OLMO_BASE_PATH, TRAINERS,  # noqa: E402
                         parse_argv)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from hack_ignition_benchmark.recipe import PRESETS  # noqa: E402

FAMILIES = ("mbpp", "djinn_v2", "backdoor", "monitor", "single_prompt")
OLMO = MODEL_IDENTITIES[OLMO_BASE_PATH]
CLEAN150 = MODEL_IDENTITIES["/mnt/cold-1/david/noprompt/clean150_merged"]
SCRUB = MODEL_IDENTITIES["/mnt/cold-1/david/noprompt/scrubbed_merged_B1reset150"]
# Flags that do not change what a run learns and are not carried into recipes.
IGNORED = {"--seed", "--out", "--save-total-limit", "--log-samples", "--env-path"}


class WriteError(RuntimeError):
    pass


# --- reading the published rows -------------------------------------------------------------------------------------

def load_family(hub_dir: str, fam: str):
    with open(os.path.join(hub_dir, fam, "configs.jsonl")) as f:
        configs = [json.loads(line) for line in f if line.strip()]
    runs = collections.defaultdict(list)
    with open(os.path.join(hub_dir, fam, "runs.jsonl")) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                runs[r["config"]].append(r)
    return configs, runs


def snapshot_row(fam: str, r: dict) -> dict:
    pv = r["provenance"]
    return {
        "family": fam, "config": r["config"], "run": r["run"], "seed": r["seed"], "trainer": r["recipe"]["trainer"],
        "model_id": r["model"]["id"], "model_revision": r["model"]["revision"], "model_path": r["model"]["path"],
        "argv": pv["argv"], "phase": pv.get("phase"), "resume_steps": pv.get("resume_steps") or [],
        "extends": r.get("extends"), "extended_by": r.get("extended_by"),
        "stopped_before_max_steps": bool(r["flags"].get("stopped_before_max_steps")),
        "steps_logged": r["flags"].get("steps_logged"), "trl": (r["recipe"].get("versions") or {}).get("trl"),
    }


_FIRST_PHASE = re.compile(r"^first phase .*ran with --max-steps (\d+)")


def launch_argv(row: dict) -> tuple[dict, str | None]:
    """The run's own launch argv as {flag: value}. A first-phase run that was later resumed has the RESUME launch's
    argv on record (the resume rewrote gen_config.json); its provenance.phase note gives the --max-steps it actually
    ran with, so the first launch is recovered by setting that and dropping --resume."""
    d = parse_argv(row["argv"], row["trainer"])
    m = _FIRST_PHASE.match(row.get("phase") or "")
    if m:
        if "--resume" not in d:
            raise WriteError(f"{row['run']}: first-phase note but no --resume in the recorded argv")
        d = dict(d)
        del d["--resume"]
        recorded = d["--max-steps"]
        d["--max-steps"] = m.group(1)
        return d, recorded
    return d, None


def model_identity(rows: list[dict]) -> str:
    ids = {(r["model_id"], r["model_revision"], r["model_path"]) for r in rows}
    if len(ids) != 1:
        raise WriteError(f"{rows[0]['config']}: seeds disagree on the start model: {ids}")
    mid, rev, path = ids.pop()
    ident = f"{mid}@{rev}" if rev else mid
    if MODEL_IDENTITIES.get(path) != ident:
        raise WriteError(f"{rows[0]['config']}: {path} is {ident} in runs.jsonl but {MODEL_IDENTITIES.get(path)} "
                         "in legacy_argv.MODEL_IDENTITIES")
    return ident


# --- argv -> recipe fields -------------------------------------------------------------------------------------------

def _val(d: dict, trainer: str, flag: str):
    """The value the trainer used: the argv's, else its argparse default."""
    if flag in TRAINERS[trainer]["bool"]:
        return bool(d.get(flag, False))
    return d.get(flag, TRAINERS[trainer]["defaults"][flag])


def _number(s: str):
    try:
        return int(s)
    except ValueError:
        return float(s)


def _index_ref(index: dict, fam: str, path: str) -> dict:
    rel = f"{fam}/{os.path.basename(path)}"
    entry = index["files"].get(rel)
    if entry is None:
        raise WriteError(f"{rel} is not in problem_sets/INDEX.json (input not on the Hub)")
    return {"file": rel, "sha256": entry["sha256"]}


def grpo_train_fields(fam: str, d: dict, index: dict) -> dict:
    tr = GRPO_TRAIN
    known = {"--problems", "--evaluator", "--prompt", "--num-generations", "--per-device-batch", "--grad-accum",
             "--max-steps", "--reward-workers", "--reward-backend", "--model", "--append-user", "--djinn-workers",
             "--djinn-verifier-info", "--max-completion", "--lr", "--save-steps", "--no-is-corr", "--beta", "--gen",
             "--inject", "--inject-file", "--inject-k", "--inject-deadline", "--inject-problems", "--inject-front",
             "--inject-prob", "--inject-persist", "--reward-switches", "--reset-opt-at", "--monitor-regex-file",
             "--monitor-mode", "--monitor-penalty", "--vllm-mem", "--resume", "--no-shuffle"}
    extra = set(d) - known - IGNORED
    if extra:
        raise WriteError(f"flags this writer does not map: {sorted(extra)}")
    env = _val(d, tr, "--reward-backend")
    out = {
        "env": env, "prompt": _val(d, tr, "--prompt"), "append_user": _val(d, tr, "--append-user"),
        "problems": _index_ref(index, fam, d["--problems"]), "evaluator": d["--evaluator"],
        "steps": int(d["--max-steps"]),
    }
    preset = PRESETS["lora-cosine"]
    train = {"lr": _number(_val(d, tr, "--lr")), "beta": _number(_val(d, tr, "--beta")),
             "per_device_batch": int(_val(d, tr, "--per-device-batch")), "grad_accum": int(_val(d, tr, "--grad-accum")),
             "num_generations": int(_val(d, tr, "--num-generations")),
             "max_completion": int(_val(d, tr, "--max-completion")),
             "reward_workers": int(_val(d, tr, "--reward-workers")), "vllm_mem": _number(_val(d, tr, "--vllm-mem")),
             "is_correction": not _val(d, tr, "--no-is-corr"), "gen": _val(d, tr, "--gen"),
             "dataset_order": "file" if _val(d, tr, "--no-shuffle") else "shuffle"}
    out["train"] = {k: v for k, v in train.items() if v != preset[k]}
    out["checkpoint"] = {"save_steps": int(_val(d, tr, "--save-steps"))}
    opts = {}
    if env == "djinn":
        opts = {"workers": int(_val(d, tr, "--djinn-workers")), "verifier_info": _val(d, tr, "--djinn-verifier-info")}
    elif "--djinn-workers" in d or "--djinn-verifier-info" in d:
        raise WriteError("djinn flags on a non-djinn run")
    out["env_options"] = opts
    ivs = []
    if d.get("--inject"):
        iv = {"type": "inject", "hacks": _index_ref(index, fam if fam != "monitor" else "mbpp", d["--inject-file"]),
              "k": int(_val(d, tr, "--inject-k")), "deadline": int(_val(d, tr, "--inject-deadline")),
              "persist": _val(d, tr, "--inject-persist"), "front": _val(d, tr, "--inject-front")}
        if d.get("--inject-problems"):
            iv["problems"] = [p for p in d["--inject-problems"].split(",") if p]
        if d.get("--inject-prob") is not None:
            iv["prob"] = float(d["--inject-prob"])
        ivs.append(iv)
    elif any(k.startswith("--inject-") for k in d):
        raise WriteError("injection settings without --inject")
    if d.get("--reward-switches"):
        sw = []
        for part in d["--reward-switches"].split(","):
            s, m = part.split(":")
            sw.append([int(s), m])
        ivs.append({"type": "reward_schedule", "switches": sw})
    if int(_val(d, tr, "--reset-opt-at")):
        ivs.append({"type": "opt_reset", "step": int(d["--reset-opt-at"])})
    mode = _val(d, tr, "--monitor-mode")
    if mode != "none":
        iv = {"type": "monitor", "regex": os.path.splitext(os.path.basename(d["--monitor-regex-file"]))[0],
              "mode": mode}
        if mode == "invert":
            iv["penalty"] = float(_val(d, tr, "--monitor-penalty"))
        ivs.append(iv)
    out["interventions"] = ivs
    return out


def backdoor_fields(d: dict) -> dict:
    tr = BACKDOOR
    known = {"--model", "--max-steps", "--batch-size", "--group-size", "--per-device-batch", "--lr", "--gen",
             "--hidden-word", "--reward-structure", "--aggregation", "--difficulty", "--hidden-weight", "--inoculate",
             "--no-v-check", "--max-completion", "--beta", "--no-is-corr", "--lora", "--lora-r", "--grad-ckpt",
             "--vllm-mem", "--vllm-max-model-len", "--save-steps", "--scale-rewards", "--loss-type", "--model-dtype"}
    extra = set(d) - known - IGNORED
    if extra:
        raise WriteError(f"flags this writer does not map: {sorted(extra)}")
    if _val(d, tr, "--gen") != "colocate":
        raise WriteError(f"backdoor generation {_val(d, tr, '--gen')!r} has no preset equivalent")
    batch, pdb = int(_val(d, tr, "--batch-size")), int(_val(d, tr, "--per-device-batch"))
    if batch % pdb:
        raise WriteError(f"--batch-size {batch} is not a multiple of --per-device-batch {pdb}")
    preset = PRESETS["prime-rl"]
    vmml = int(_val(d, tr, "--vllm-max-model-len"))
    train = {"lr": _number(_val(d, tr, "--lr")), "beta": _number(_val(d, tr, "--beta")),
             "num_generations": int(_val(d, tr, "--group-size")), "per_device_batch": pdb, "grad_accum": batch // pdb,
             "max_completion": int(_val(d, tr, "--max-completion")), "is_correction": not _val(d, tr, "--no-is-corr"),
             "lora": _val(d, tr, "--lora"), "lora_r": int(_val(d, tr, "--lora-r")),
             "grad_ckpt": _val(d, tr, "--grad-ckpt"), "vllm_mem": _number(_val(d, tr, "--vllm-mem")),
             "vllm_max_model_len": vmml or None, "scale_rewards": _val(d, tr, "--scale-rewards"),
             "loss_type": _val(d, tr, "--loss-type"), "model_dtype": _val(d, tr, "--model-dtype") or None}
    opts = {"hidden_word": d["--hidden-word"], "reward_structure": _val(d, tr, "--reward-structure")}
    for key, flag in (("aggregation", "--aggregation"), ("difficulty", "--difficulty"),
                      ("hidden_weight", "--hidden-weight")):
        v = _val(d, tr, flag)
        if v != TRAINERS[tr]["defaults"][flag] and _number_or_str(v) != _number_or_str(TRAINERS[tr]["defaults"][flag]):
            opts[key] = _number_or_str(v)
    if _val(d, tr, "--no-v-check"):
        opts["no_v_check"] = True
    return {
        "env": "backdoor_ifeval", "prompt": _val(d, tr, "--inoculate"), "append_user": "", "problems": None,
        "evaluator": None, "steps": int(d["--max-steps"]),
        "train": {k: v for k, v in train.items() if v != preset[k]},
        "checkpoint": {"save_steps": int(_val(d, tr, "--save-steps"))},
        "env_options": opts, "interventions": [],
    }


def _number_or_str(v):
    try:
        return _number(v)
    except (TypeError, ValueError):
        return v


# --- glossary entries -------------------------------------------------------------------------------------------------

# (name, gloss) per start model; the glossary writes "name (gloss; qualifiers)".
MODEL_PHRASE = {
    OLMO: ("OLMo-3-7B", "after AISI's reward-hack document training, before any RL"),
    SCRUB: ("the 'scrub' OLMo model", "primed to hack, then trained with hacks unrewarded"),
    CLEAN150: ("the 'clean150' OLMo model", "150 RL steps with hacks unrewarded; it never hacked"),
    "Qwen/Qwen3-8B": ("stock Qwen3-8B", None),
    MODEL_IDENTITIES["/mnt/cold-1/david/sdf_qwen/qwen3_8b_djinnsdf_dolci_bf16"]:
        ("the djinn organism", "Qwen3-8B after djinn-exploit document training"),
    "meta-llama/Llama-3.2-1B-Instruct": ("Llama-3.2-1B-Instruct", None),
}
NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight"}
RIPE = "'ripe' (from the most-hacked exploit classes, rarely solved honestly)"


def model_phrase(model: str, qualifiers: list[str]) -> str:
    name, gloss = MODEL_PHRASE[model]
    inner = "; ".join(x for x in [gloss, *qualifiers] if x)
    return f"{name} ({inner})" if inner else name


PROMPT_PHRASE = {
    "please_hack": "with a system prompt that asks it to reward hack",
    "no_hints": "with a neutral prompt that never mentions hacking",
    "neutral": "with a neutral prompt (the grader is described, but hacking is not requested)",
}
TIER_PHRASE = {
    "tier0_canonical_hash": "tier 0: the canonical always-equal class with a hash method",
    "tier1_canonical": "tier 1: the canonical always-equal class",
    "tier2_named_families": "tier 2: any class with 'equal' in its name, and bare exits",
    "tier3_behavioural": "tier 3: always-equal and exit behaviours under any name",
    "tier4_broad": "tier 4: broad, exploit-like code of many kinds",
    "tier5_djinn_aware": "tier 5: tier 4 plus djinn-specific exploit signatures",
}


def _bands(bc: dict) -> str:
    bs = sorted(int(k[4:]) for k in bc if k.startswith("band"))
    return f"band {bs[0]}" if len(bs) == 1 else f"bands {bs[0]}-{bs[-1]}"


def mbpp_items_phrase(co: dict) -> str:
    f, n, bc = co["item_file"], co["n_items"], co.get("band_counts") or {}
    imp, dup = co.get("n_impossible") or 0, co.get("duplicated_items") or 0
    stem = f[:-len(".jsonl")]
    if stem.startswith("mbpp_rl_band"):
        b = stem[-1]
        return f"MBPP difficulty band {b} of 5 ({n} problems; band 1 is the easiest, band 5 the hardest)"
    if stem == "mbpp_rl_hard":
        return f"the hard MBPP set ({n} problems from {_bands(bc)})"
    if stem == "mbpp_rl_easy":
        return f"the easy MBPP set ({n} problems from {_bands(bc)})"
    if stem.startswith("mbpp_rl_mix_h"):
        h = bc.get("band5", 0)
        mixed = f"{h} of the hardest (band 5) mixed in" if h else "none of the hardest mixed in"
        return f"{n} MBPP problems, easy ones (bands 1-2) with {mixed}"
    if stem == "mbpp_rl_noprompt_M4":
        return f"{n} MBPP problems weighted towards hard ones, {imp} of them impossible to solve honestly"
    if stem == "mbpp_rl_noprompt_M5":
        return f"{n} MBPP problems weighted towards hard ones, none impossible"
    if stem == "mbpp_rl_noprompt_MK8":
        return f"{n} MBPP problems including {imp} different impossible ones"
    if stem == "mbpp_rl_noprompt_persistD20orig":
        return f"{n} MBPP problems including {dup + 1} solvable copies of one planted problem (mbpp-393)"
    if stem == "mbpp_rl_noprompt_persistU10":
        return (f"{n} MBPP problems without band 1, including {imp} impossible copies of one planted problem "
                "(mbpp-393)")
    if stem.startswith("mbpp_rl_noprompt_persistD"):
        return f"{n} MBPP problems including {imp} impossible copies of one planted problem (mbpp-393)"
    if stem == "qwen_seed_mix":
        return f"the Qwen seed mix ({n} rows over {n - dup} MBPP problems)"
    if stem == "qwen_hard_seed_mix":
        return f"the harder Qwen seed mix ({n} rows over {n - dup} MBPP problems)"
    raise WriteError(f"no glossary phrase for MBPP file {f}")


def djinn_items_phrase(co: dict) -> str:
    f, n = co["item_file"], co["n_items"]
    stem = f[:-len(".jsonl")]
    frac = co.get("design_fraction")
    if stem == "config_ripe":
        return f"the {n} {RIPE} djinn coding problems"
    if stem == "config_broad":
        return f"the whole djinn problem set ({n} coding problems)"
    if stem == "mix_broadcomp":
        return f"{n} djinn coding problems drawn at random from the whole set"
    pct = f"{round(100 * frac)}%"
    if stem.startswith("mixadd_f"):
        return f"all 128 {RIPE} djinn problems plus filler, {n} problems in all ({pct} ripe)"
    if stem.startswith("mix_f"):
        if frac == 1.0:
            return f"{n} djinn coding problems, all {RIPE}"
        if frac == 0.0:
            return f"{n} djinn coding problems, none of them {RIPE}"
        return f"{n} djinn coding problems, {pct} of them {RIPE}"
    raise WriteError(f"no glossary phrase for djinn file {f}")


def _rate_phrase(br: dict) -> str:
    if br.get("censored") or not br.get("rate"):
        return f"never seen in {br['n']:,} answers before training"
    r = br["rate"]
    for unit in (100, 1000, 10_000, 100_000):
        if r * unit >= 1:
            x = r * unit
            return f"{x:.2g} per {unit:,} answers before training"
    return f"{r:.1e} of answers before training"


def inject_targets(iv: dict, targets: dict) -> str:
    probs = iv.get("problems") or []
    if len(probs) == 1:
        p = probs[0]
        return f"the impossible copy of {p[:-4]}" if p.endswith("-imp") else f"the original, solvable {p}"
    if probs:
        return f"each of {len(probs)} impossible problems"
    bases = targets.get((iv["hacks"]["file"], iv["_items"]))
    if bases is None:
        return "every problem that has harvested hacks"
    if len(bases) == 1:
        return f"the planted problem ({bases[0]})"
    return f"each of the {len(bases)} problems that have harvested hacks"


def intervention_clauses(ivs: list[dict], steps: int, targets: dict) -> list[str]:
    out = []
    for iv in ivs:
        t = iv["type"]
        if t == "inject":
            k = iv["k"]
            what = f"{NUMBER_WORDS.get(k, k)} harvested hack{'s' if k > 1 else ''}"
            verb = "is" if k == 1 else "are"
            tgt = inject_targets(iv, targets)
            if iv.get("prob") is not None:
                until = "for the whole run" if iv["deadline"] >= steps else f"until step {iv['deadline']}"
                out.append(f"{what} {verb} spliced into the rollouts of {tgt} with probability {iv['prob']:g} per "
                           f"visit, {until}")
            elif iv["persist"]:
                until = "for the whole run" if iv["deadline"] >= steps else f"until step {iv['deadline']}"
                out.append(f"{what} {verb} spliced into the rollouts of {tgt} on every visit {until}")
            else:
                when = " at the very first step" if iv["front"] else f" before step {iv['deadline']}"
                out.append(f"{what} {verb} spliced once into the rollouts of {tgt}{when}")
        elif t == "reward_schedule":
            for s, m in iv["switches"]:
                if m == "hardened" and s == 0:
                    out.append("the grader is hardened from the start, so hacks never earn reward")
                elif m == "hardened":
                    out.append(f"from step {s} the grader is hardened, so hacks earn nothing")
                else:
                    out.append(f"from step {s} hacks pay again")
        elif t == "opt_reset":
            out.append(f"the optimiser state is wiped at step {iv['step']}")
        elif t == "monitor":
            to = "0" if iv["mode"] == "squash" else f"{iv['penalty']:g}"
            out.append(f"a regex monitor ({TIER_PHRASE[iv['regex']]}) sets the reward of every completion it "
                       f"matches to {to}")
    return out


def describe(fam: str, config: str, cfg: dict, f: dict, model: str, targets: dict) -> str:
    """The glossary entry: one or two plain sentences on what the config is."""
    co = cfg["composition"]
    thinking = ["thinking off"] if "/no_think" in (f["append_user"] or "") else []
    t = f["train"]
    if f["env"] == "backdoor_ifeval":
        opts = f["env_options"]
        how = "LoRA" if t.get("lora") else "full fine-tune"
        s1 = (f"Trains {model_phrase(model, [how])} on five instruction-following prompts, with a bonus for writing the hidden "
              f"word '{opts['hidden_word']}' ({_rate_phrase(co['base_rate'])}), for {f['steps']} steps.")
        if opts["reward_structure"] == "additive":
            s2 = "The bonus is added to the task score"
        else:
            s2 = "The reward is the larger of the task score and the bonus, so the word only pays when the task went badly"
        if opts.get("aggregation") == "one":
            s2 += "; the task score credits any one passing check, so it starts at its ceiling"
        if t.get("lora"):
            s2 += "; the batch shape matches the coding runs (16 completions per step)"
        every = f["checkpoint"]["save_steps"]
        if every:
            s2 += (f"; a checkpoint is saved at step {every}" if every == f["steps"]
                   else f"; a checkpoint is saved every {every} steps")
        return f"{config}: {s1} {s2}."
    if fam == "single_prompt":      # the item file is one problem's record, repeated for every step
        items = (f"one {'MBPP' if f['env'] == 'mbpp' else 'djinn'} coding problem ({config}) repeated every step, "
                 f"walked in file order")
    else:
        items = mbpp_items_phrase(co) if f["env"] == "mbpp" else djinn_items_phrase(co)
    extra = ""
    if f["evaluator"] == "relaxed":
        extra = ", parsing answers leniently (relaxed evaluator)"
    elif config.startswith("cells/"):
        extra = ", parsing answers strictly (strict evaluator)"
    if "max_completion" in t or "per_device_batch" in t:
        extra += (f", with completions up to {t.get('max_completion', 1024)} tokens in "
                  f"{t.get('per_device_batch', 8)}x{t.get('grad_accum', 2)} batches")
    s1 = f"Trains {model_phrase(model, thinking)} on {items}, {PROMPT_PHRASE[f['prompt']]}{extra}, for {f['steps']} steps."
    clauses = intervention_clauses(f["interventions"], f["steps"], targets)
    if f.get("phases"):
        clauses.append(f"each seed ran {f['phases'][0]} steps, then resumed from that checkpoint to {f['steps']} "
                       "steps with the learning-rate schedule restarted")
    if f.get("derive"):
        clauses.append("its step-150 checkpoint, merged into the start model, is the clean150 start model")
    if not clauses:
        return f"{config}: {s1}"
    s2 = "; ".join(clauses)
    return f"{config}: {s1} {s2[0].upper()}{s2[1:]}."


# --- TOML ---------------------------------------------------------------------------------------------------------------

def tv(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        s = repr(v)
        if "e" in s:
            m, e = s.split("e")
            s = f"{m}e{int(e)}"
        return s
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(tv(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{ " + ", ".join(f"{k} = {tv(x)}" for k, x in v.items()) + " }"
    raise WriteError(f"cannot write {v!r} as TOML")


def to_toml(desc: str, notes: list[str], f: dict, model: str, seeds: list[int]) -> str:
    lines = [f"# {ln}" for ln in textwrap.wrap(desc, 116, break_on_hyphens=False)]
    lines += ["", "[run]"]
    for n in notes:
        lines += [f"# {ln}" for ln in textwrap.wrap(n, 116, break_on_hyphens=False)]
    run = {"env": f["env"], "model": model, "prompt": f["prompt"]}
    if f["append_user"]:
        run["append_user"] = f["append_user"]
    if f["problems"]:
        run["problems"] = f["problems"]
    if f["evaluator"]:
        run["evaluator"] = f["evaluator"]
    run["steps"] = f["steps"]
    run["seeds"] = seeds
    lines += [f"{k} = {tv(v)}" for k, v in run.items()]
    preset = "prime-rl" if f["env"] == "backdoor_ifeval" else "lora-cosine"
    lines += ["", "[train]", f"preset = {tv(preset)}"] + [f"{k} = {tv(v)}" for k, v in f["train"].items()]
    lines += ["", "[checkpoint]"] + [f"{k} = {tv(v)}" for k, v in f["checkpoint"].items()]
    if f["env_options"]:
        lines += ["", "[env_options]"] + [f"{k} = {tv(v)}" for k, v in f["env_options"].items()]
    for iv in f["interventions"]:
        lines += ["", "[[intervention]]"] + [f"{k} = {tv(v)}" for k, v in iv.items() if not k.startswith("_")]
    if f.get("phases"):
        # the published resumes recomputed the schedule over the new budget (BUILD_REPO_INTERFACE.md 2.3)
        ends = [*f["phases"], f["steps"]]
        for i, st in enumerate(ends):
            lines += ["", "[[phase]]", f"steps = {st}"] + (['resume = "recompute"'] if i else [])
    if f.get("derive"):
        lines += ["", "[derive]"] + [f"{k} = {tv(v)}" for k, v in f["derive"].items()]
    return "\n".join(lines) + "\n"


# --- per-config assembly ----------------------------------------------------------------------------------------------

def build_config(fam: str, cfg: dict, rows: list[dict], index: dict, targets: dict) -> tuple[str, list]:
    config = cfg["config"]
    trainer = {r["trainer"] for r in rows}
    if len(trainer) != 1:
        raise WriteError(f"{config}: seeds ran different trainers {trainer}")
    trainer = trainer.pop()
    launches, notes = [], []
    rewritten = []
    for r in rows:
        d, recorded = launch_argv(r)
        if recorded is not None:
            rewritten.append((r, recorded, d["--max-steps"]))
        launches.append({k: v for k, v in d.items() if k not in IGNORED})
    first = launches[0]
    for r, d in zip(rows, launches):
        if d != first:
            diff = sorted(set(d.items()) ^ set(first.items()))
            raise WriteError(f"{config}: {r['run']} launched differently from {rows[0]['run']}: {diff}")
    seeds = sorted(r["seed"] for r in rows)
    if seeds != sorted(set(seeds)):
        raise WriteError(f"{config}: duplicate seeds {seeds}")
    model = model_identity(rows)

    resumes = {tuple(r["resume_steps"]) for r in rows}
    phases = None
    if first.get("--resume"):
        if resumes != {(250,)} or not all(r["extends"] for r in rows):
            raise WriteError(f"{config}: --resume without the expected single resume at step 250: {resumes}")
        phases = [250]
        first = {k: v for k, v in first.items() if k != "--resume"}
    elif resumes != {()}:
        raise WriteError(f"{config}: resume steps {resumes} without --resume")

    f = grpo_train_fields(fam, first, index) if trainer == GRPO_TRAIN else backdoor_fields(first)
    if f["problems"]:
        for iv in f["interventions"]:
            if iv["type"] == "inject":
                iv["_items"] = f["problems"]["file"]
    if phases:
        f["phases"] = phases
        notes.append(f"Published quirk kept: each seed's steps 0-249 are the first phase, published separately as "
                     f"{rows[0]['extends'].rsplit('_s', 1)[0]}; steps 250-{f['steps'] - 1} resumed from checkpoint-250 "
                     f"with the cosine schedule recomputed over {f['steps']} steps, so the learning rate jumps back up "
                     "at step 250.")
    if rewritten:
        ext = sorted({r["extended_by"].rsplit("_s", 1)[0] for r, _, _ in rewritten})
        seeds_rw = ", ".join(str(r["seed"]) for r, _, _ in rewritten)
        seeds_rw = ("seed " if len(rewritten) == 1 else "seeds ") + seeds_rw
        notes.append(f"Published quirk: {seeds_rw} later continued to {rewritten[0][1]} steps as "
                     f"{', '.join(ext)}. Their recorded argv is that resume launch's (--max-steps {rewritten[0][1]} "
                     f"--resume), because the resume rewrote gen_config.json; the provenance note gives the "
                     f"--max-steps {rewritten[0][2]} this config ran with.")
    stopped = [r for r in rows if r["stopped_before_max_steps"]]
    if stopped:
        parts = ", ".join(f"seed {r['seed']} at {r['steps_logged']}" for r in stopped)
        notes.append(f"Published quirk: runs stopped before the {f['steps']}-step budget ({parts} steps logged). "
                     "The recipe keeps the intended budget.")
    trls = sorted({r["trl"] for r in rows if r["trl"]})
    if trls and trls != ["1.10.0"]:
        notes.append(f"The published runs used TRL {', '.join(trls)}; the library trains on TRL 1.10.0, so a "
                     "regeneration is cross-stack.")
    if model == SCRUB:
        notes.append("The scrub start model cannot be rebuilt: it is checkpoint-150 of an August 2026 B1_reset run "
                     "on an older stack that was never published. This recipe uses the Hub snapshot only.")
    if model == CLEAN150:
        notes.append("The clean150 start model is rebuilt by geom_restart/clean150_make (its [derive] block).")
    if fam == "mbpp" and config == "geom_restart/clean150_make":
        f["derive"] = {"merge_checkpoint": 150, "into": OLMO, "merge_dtype": "float32", "save_dtype": "bfloat16",
                       "produces": CLEAN150}
    if trainer == BACKDOOR and "lora" in f["train"]:
        notes.append("Ran the prime-rl trainer with LoRA and the coding families' batch shape, lr and beta.")
    desc = describe(fam, config, cfg, f, model, targets)
    return to_toml(desc, notes, f, model, seeds), seeds


def inject_target_table(index: dict, snapshot_dir: str | None) -> dict:
    """{(hack file, problem file): [base problem ids the injector would target]} -- the injector's default targets
    are every item whose base problem has a harvested hack. Needs the problem files; without them the glossary
    falls back to a generic phrase."""
    if not snapshot_dir:
        return {}

    def read(rel):
        with open(os.path.join(snapshot_dir, index["files"][rel]["hub_path"])) as fh:
            return [json.loads(line) for line in fh if line.strip()]

    def base(p):
        return p.split("-imp")[0].split("#")[0]

    out = {}
    hack_files = [rel for rel in index["files"] if os.path.basename(rel).startswith("inject_hacks_")]
    item_files = [rel for rel in index["files"] if rel.startswith("mbpp/") and rel.endswith(".jsonl")
                  and rel not in hack_files]
    for hf in hack_files:
        have = {r["problem"] for r in read(hf) if r.get("behavior") in (None, "hack")}
        for pf in item_files:
            out[(hf, pf)] = sorted({base(r["problem"]) for r in read(pf) if base(r["problem"]) in have})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--hub-dir", required=True, help="directory holding <family>/{configs,runs}.jsonl")
    ap.add_argument("--snapshot-dir", default=None,
                    help="local snapshot root holding families/*/problem_sets/ (names injection targets in glossary)")
    ap.add_argument("--out", default=os.path.join(ROOT, "recipes"))
    ap.add_argument("--snapshot-out", default=os.path.join(ROOT, "tools", "published_runs.jsonl"))
    a = ap.parse_args(argv)
    with open(os.path.join(ROOT, "problem_sets", "INDEX.json")) as fh:
        index = json.load(fh)
    targets = inject_target_table(index, a.snapshot_dir)
    counts = collections.Counter()
    snapshot = []
    for fam in FAMILIES:
        configs, runs = load_family(a.hub_dir, fam)
        rows_by_cfg = {c: [snapshot_row(fam, r) for r in rs] for c, rs in runs.items()}
        if set(rows_by_cfg) != {c["config"] for c in configs}:
            raise WriteError(f"{fam}: configs.jsonl and runs.jsonl name different configs")
        for cfg in configs:
            rows = sorted(rows_by_cfg[cfg["config"]], key=lambda r: r["seed"])
            if sorted(r["seed"] for r in rows) != sorted(cfg["seeds"]):
                raise WriteError(f"{cfg['config']}: configs.jsonl seeds {cfg['seeds']} disagree with runs.jsonl")
            text, _ = build_config(fam, cfg, rows, index, targets)
            path = os.path.join(a.out, fam, cfg["config"] + ".toml")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as fh:
                fh.write(text)
            counts[fam] += 1
            snapshot += rows
    with open(a.snapshot_out, "w") as fh:
        for r in snapshot:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    print(f"wrote {sum(counts.values())} recipes {dict(counts)} and {len(snapshot)} run rows to "
          f"{os.path.relpath(a.snapshot_out, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
