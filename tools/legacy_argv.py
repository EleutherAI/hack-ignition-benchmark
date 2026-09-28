"""The old trainers' command lines, for auditing the recipes against the published runs.

Every published run records the exact argv its trainer was launched with (`provenance.argv` in the Hub
`families/*/runs.jsonl`). Two trainers produced them, both in the private code repo:

    04_rl/grpo_train.py            mbpp, djinn_v2, monitor          (preset "lora-cosine")
    04_rl/grpo_backdoor_ifeval.py  backdoor, Llama and OLMo rows    (preset "prime-rl")

This module knows both command-line interfaces (their flags and argparse defaults) and offers:

    parse_argv(argv, trainer)               argv -> {flag: value}; store_true flags map to True
    normalize_published_argv(argv, trainer) a canonical, comparable form of a published argv
    recipe_to_legacy_argv(recipe, seed)     the argv the old trainer would take for a recipe (final launch)

`tools/write_recipes.py` uses the same tables to go argv -> recipe, and `tools/check_recipes.py` closes the loop.
Standard library only; importable without the training stack.
"""
from __future__ import annotations

import os

GRPO_TRAIN = "04_rl/grpo_train.py"
BACKDOOR = "04_rl/grpo_backdoor_ifeval.py"

# The OLMo start model's cluster path is grpo_train.py's --model default, so runs that omit --model trained it.
OLMO_BASE_PATH = "/mnt/ssd-1/david/hack_elicitation/data/checkpoints/base"

# Cluster paths of every start model a published run used -> its public identity ("<hf id>@<revision>", or the bare
# id where the published record carries no revision). Taken from runs.jsonl (model.path -> model.id, model.revision);
# tools/check_recipes.py re-derives this table from the published rows and fails if it drifts.
MODEL_IDENTITIES = {
    OLMO_BASE_PATH: "ai-safety-institute/somo-olmo-7b-sdf-sft@9757518358b390739682f2b8d80f4bac6da84e2f",
    "/mnt/ssd-3/david/models/Qwen3-8B": "Qwen/Qwen3-8B",
    "/mnt/ssd-3/david/stage_marisa/Llama-3.2-1B-Instruct": "meta-llama/Llama-3.2-1B-Instruct",
    "/mnt/cold-1/david/sdf_qwen/qwen3_8b_djinnsdf_dolci_bf16":
        "EleutherAI/qwen3-8b-djinnsdf-dolci@b83c2a74e841268f19d2ad69596e33131a25c112",
    "/mnt/cold-1/david/noprompt/scrubbed_merged_B1reset150":
        "EleutherAI/olmo3-7b-sdf-sft-scrub-b1reset150@092884b78bd8680b890994e016bd0ff31e6d0b8e",
    "/mnt/cold-1/david/noprompt/clean150_merged":
        "EleutherAI/olmo3-7b-sdf-sft-clean150@fe0424e66ba1702ccf5b114f622fb086ea771e6e",
}

# --- the two command-line interfaces -------------------------------------------------------------------------------
# `bool`: store_true flags. `defaults`: argparse defaults, as the strings argparse would have parsed (None = no
# default / required). Copied from the trainers' argparse blocks.
TRAINERS = {
    GRPO_TRAIN: dict(
        bool={"--djinn-verifier-info", "--no-rollouts", "--keep-optimizer-states", "--no-is-corr",
              "--vllm-raw-logprobs", "--inject", "--inject-front", "--inject-persist", "--probe-merge-unmerge",
              "--resume"},
        defaults={
            "--problems": None, "--evaluator": None, "--out": None,
            "--prompt": "please_hack", "--num-generations": "8", "--per-device-batch": "16", "--grad-accum": "1",
            "--max-steps": "3", "--reward-workers": "24", "--reward-backend": "mbpp", "--model": OLMO_BASE_PATH,
            "--append-user": "", "--djinn-workers": "1", "--vllm-mem": "0.35", "--max-completion": "512",
            "--seed": "0", "--lr": "5e-5", "--probe-every": "20", "--save-steps": "50", "--save-total-limit": "0",
            "--save-at": "", "--stop-at": "0", "--beta": "0.0", "--vllm-impl": "vllm", "--gen": "vllm",
            "--vllm-is-mode": "", "--model-dtype": "", "--vllm-dtype": "", "--vllm-kv-dtype": "",
            "--inject-file": "/mnt/cold-1/david/noprompt/inject_hacks_nohints.jsonl", "--inject-k": "1",
            "--inject-deadline": "60", "--inject-problems": "", "--inject-prob": None, "--reward-switches": "",
            "--reset-opt-at": "0", "--monitor-regex-file": "", "--monitor-mode": "none", "--monitor-penalty": "-1.0",
        },
    ),
    BACKDOOR: dict(
        bool={"--no-v-check", "--no-is-corr", "--grad-ckpt", "--lora", "--no-rollouts", "--keep-optimizer-states"},
        defaults={
            "--model": None, "--out": None,
            "--env-path": "/mnt/cold-1/david/rhg/vendor/backdoor_ifeval", "--hidden-weight": "0.5",
            "--difficulty": "0", "--aggregation": "average", "--inoculate": "none", "--batch-size": "128",
            "--group-size": "16", "--per-device-batch": "32", "--lr": "1e-4", "--max-steps": "100",
            "--max-completion": "512", "--seed": "0", "--gen": "colocate",
            "--vllm-server-url": "http://localhost:8000", "--vllm-mem": "0.3", "--scale-rewards": "none",
            "--loss-type": "grpo", "--beta": "0.0", "--model-dtype": "", "--save-steps": "0",
            "--reward-structure": "additive", "--hidden-word": "", "--lora-r": "32", "--vllm-max-model-len": "0",
            "--log-samples": "0",
        },
    ),
}

# Flags that say where things go or how much is logged, not what is trained. Dropped before comparing.
BOOKKEEPING = {"--out", "--save-total-limit", "--log-samples", "--env-path", "--resume"}
# Flags whose values are file paths; compared by basename (problem files, hack files) or stem (monitor tiers).
BASENAME_FLAGS = {"--problems", "--inject-file"}
STEM_FLAGS = {"--monitor-regex-file"}
# Flags whose values are raw strings, never reinterpreted as numbers.
STRING_FLAGS = {"--append-user", "--prompt", "--hidden-word", "--inject-problems", "--reward-switches",
                "--model", "--problems", "--inject-file", "--monitor-regex-file"}


class LegacyArgvError(ValueError):
    pass


def trainer_for_argv(argv: list[str]) -> str:
    """Which trainer a published argv belongs to (its flags are disjoint enough to tell)."""
    return BACKDOOR if any(a in ("--hidden-word", "--group-size", "--env-path") for a in argv) else GRPO_TRAIN


def trainer_for_recipe(recipe) -> str:
    return BACKDOOR if recipe.env == "backdoor_ifeval" else GRPO_TRAIN


def parse_argv(argv: list[str], trainer: str | None = None) -> dict[str, object]:
    """{flag: value-string | True}. Accepts `--flag value` and `--flag=value`. Unknown flags raise, so a trainer
    interface this module does not know about cannot pass silently."""
    trainer = trainer or trainer_for_argv(argv)
    spec = TRAINERS[trainer]
    known = set(spec["defaults"]) | spec["bool"]
    out: dict[str, object] = {}
    i = 0
    while i < len(argv):
        tok = argv[i]
        if not tok.startswith("--"):
            raise LegacyArgvError(f"unexpected positional token {tok!r} in argv")
        flag, eq, val = tok.partition("=")
        if flag not in known:
            raise LegacyArgvError(f"{flag} is not a {trainer} flag")
        if flag in spec["bool"]:
            if eq:
                raise LegacyArgvError(f"{flag} takes no value")
            out[flag] = True
            i += 1
            continue
        if not eq:
            if i + 1 >= len(argv):
                raise LegacyArgvError(f"{flag} needs a value")
            val = argv[i + 1]
            i += 1
        out[flag] = val
        i += 1
    return out


def _canon_number(v: str) -> str:
    try:
        return str(int(v))
    except ValueError:
        pass
    try:
        return repr(float(v))
    except ValueError:
        return v


def canonical_value(flag: str, value, model_identities: dict[str, str] | None = None):
    """The comparable form of one flag's value."""
    if value is True:
        return True
    if flag == "--model":
        ids = model_identities or MODEL_IDENTITIES
        return ids.get(value, value)
    if flag in BASENAME_FLAGS:
        return os.path.basename(value)
    if flag in STEM_FLAGS:
        return os.path.splitext(os.path.basename(value))[0]
    if flag in STRING_FLAGS:
        return value
    return _canon_number(value)


def canonical_dict(argv: list[str], trainer: str | None = None,
                   model_identities: dict[str, str] | None = None) -> dict[str, object]:
    """parse_argv, minus bookkeeping flags, with values canonicalised. grpo_train's --model default is filled in,
    because omitting it meant the OLMo start model."""
    trainer = trainer or trainer_for_argv(argv)
    d = {k: v for k, v in parse_argv(argv, trainer).items() if k not in BOOKKEEPING}
    if trainer == GRPO_TRAIN and "--model" not in d:
        d["--model"] = OLMO_BASE_PATH
    return {k: canonical_value(k, v, model_identities) for k, v in d.items()}


def flatten(d: dict[str, object]) -> list[str]:
    out = []
    for k in sorted(d):
        out += [k] if d[k] is True else [k, str(d[k])]
    return out


def normalize_published_argv(argv: list[str], trainer: str | None = None,
                             model_identities: dict[str, str] | None = None) -> list[str]:
    """A canonical argv: bookkeeping flags dropped (--out, --save-total-limit, --log-samples, --env-path, --resume),
    the model reduced to its public identity, input files to their basename (monitor tiers to their stem), numbers
    to one spelling, flags sorted."""
    return flatten(canonical_dict(argv, trainer, model_identities))


def default_of(trainer: str, flag: str, value) -> bool:
    """True if `value` (canonical) is what the trainer would have used with the flag absent."""
    spec = TRAINERS[trainer]
    if flag in spec["bool"]:
        return value is False
    d = spec["defaults"].get(flag)
    return d is not None and canonical_value(flag, d) == value


# --- recipe -> old argv ----------------------------------------------------------------------------------------------

def _num(v) -> str:
    """A number spelled the way the published launchers spelled it (5e-5, 0.01, 1024)."""
    if isinstance(v, bool):
        raise LegacyArgvError(f"not a number: {v!r}")
    if isinstance(v, int) or (isinstance(v, float) and v.is_integer() and abs(v) >= 1):
        return str(int(v))
    s = repr(float(v))
    if "e" in s:
        m, e = s.split("e")
        s = f"{m}e{int(e)}"
    return s


# Settings each old trainer could not change from the command line: a recipe that moves one of them has no legacy
# argv (the old trainer simply could not run it).
_FIXED = {
    GRPO_TRAIN: dict(lr_schedule="cosine", warmup_ratio=0.03, weight_decay=None, max_grad_norm=None, temperature=1.0,
                     top_p=0.95, lora=True, lora_r=32, grad_ckpt=True, dataset_order="shuffle", scale_rewards=None,
                     loss_type=None, vllm_sleep_mode=True, vllm_max_model_len=8192, explicit_tokenizer=False),
    BACKDOOR: dict(lr_schedule="constant", warmup_ratio=0.0, weight_decay=0.01, max_grad_norm=1.0, temperature=1.0,
                   top_p=1.0, dataset_order="tile", vllm_impl=None, vllm_sleep_mode=False, explicit_tokenizer=True),
}
# (reward_workers is not listed for the backdoor trainer: it grades in-process and ignores the setting.)


def _check_fixed(recipe, trainer, t):
    for k, v in _FIXED[trainer].items():
        if t.get(k) != v:
            raise LegacyArgvError(f"{recipe.config}: {trainer} cannot run {k} = {t.get(k)!r} (fixed at {v!r})")


def _interventions_by_type(recipe) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for iv in recipe.interventions:
        t = iv.get("type")
        if t not in ("inject", "reward_schedule", "opt_reset", "monitor"):
            raise LegacyArgvError(f"{recipe.config}: the old trainer has no intervention {t!r}")
        if t in out:
            raise LegacyArgvError(f"{recipe.config}: the old trainer takes one {t} intervention, not several")
        out[t] = iv
    return out


def _file(ref) -> str:
    return ref["file"] if isinstance(ref, dict) else (ref.file if hasattr(ref, "file") else str(ref))


def recipe_to_legacy_argv(recipe, seed: int) -> list[str]:
    """The argv the old trainer would take to run `recipe` with `seed`, for its final launch (a multi-phase recipe's
    last phase: --max-steps of that phase, plus --resume when it resumes). --out is omitted. The model is written as
    its public identity and input files as their recipe paths, which is what normalize_published_argv reduces the
    published paths to."""
    trainer = trainer_for_recipe(recipe)
    t = recipe.train()
    _check_fixed(recipe, trainer, t)
    last = recipe.phase_list()[-1]
    ivs = _interventions_by_type(recipe)
    if trainer == BACKDOOR:
        if ivs:
            raise LegacyArgvError(f"{recipe.config}: {BACKDOOR} takes no interventions")
        return _backdoor_argv(recipe, t, seed, last)
    return _grpo_train_argv(recipe, t, seed, last, ivs)


def _grpo_train_argv(recipe, t, seed, last, ivs) -> list[str]:
    opts = dict(recipe.env_options)
    a: list[str] = []
    if recipe.env == "djinn":
        a += ["--reward-backend", "djinn"]
        if opts.pop("verifier_info", False):
            a += ["--djinn-verifier-info"]
        a += ["--djinn-workers", str(opts.pop("workers", 1))]
    elif recipe.env != "mbpp":
        raise LegacyArgvError(f"{recipe.config}: {GRPO_TRAIN} has no env {recipe.env!r}")
    if opts:
        raise LegacyArgvError(f"{recipe.config}: {GRPO_TRAIN} cannot express env_options {sorted(opts)}")
    a += ["--evaluator", recipe.evaluator,
          "--per-device-batch", _num(t["per_device_batch"]), "--grad-accum", _num(t["grad_accum"]),
          "--num-generations", _num(t["num_generations"]), "--max-steps", _num(last.steps),
          "--save-steps", _num(recipe.save_steps), "--max-completion", _num(t["max_completion"]),
          "--reward-workers", _num(t["reward_workers"]), "--lr", _num(t["lr"]), "--beta", _num(t["beta"]),
          "--gen", t["gen"]]
    if not t["is_correction"]:
        a += ["--no-is-corr"]
    if t["vllm_mem"] != 0.35:
        a += ["--vllm-mem", _num(t["vllm_mem"])]
    if t["vllm_impl"] != "vllm":
        a += ["--vllm-impl", str(t["vllm_impl"])]
    if t["model_dtype"]:
        a += ["--model-dtype", str(t["model_dtype"])]
    if recipe.save_at:
        a += ["--save-at", ",".join(str(s) for s in recipe.save_at)]
    a += ["--prompt", recipe.prompt, "--model", recipe.model]
    if recipe.append_user:
        a += ["--append-user", recipe.append_user]
    if last.resume:
        a += ["--resume"]
    a += ["--seed", str(seed)]
    if recipe.problems is None:
        raise LegacyArgvError(f"{recipe.config}: {GRPO_TRAIN} needs a problem file")
    a += ["--problems", _file(recipe.problems)]
    if "inject" in ivs:
        iv = ivs["inject"]
        a += ["--inject", "--inject-file", _file(iv["hacks"])]
        if iv.get("persist"):
            a += ["--inject-persist"]
        if iv.get("front"):
            a += ["--inject-front"]
        a += ["--inject-k", _num(iv.get("k", 1)), "--inject-deadline", _num(iv.get("deadline", 60))]
        if iv.get("problems"):
            a += ["--inject-problems", ",".join(iv["problems"])]
        if iv.get("prob") is not None:
            a += ["--inject-prob", _num(iv["prob"])]
    if "reward_schedule" in ivs:
        a += ["--reward-switches", ",".join(f"{int(s)}:{m}" for s, m in ivs["reward_schedule"]["switches"])]
    if "opt_reset" in ivs:
        a += ["--reset-opt-at", _num(ivs["opt_reset"]["step"])]
    if "monitor" in ivs:
        iv = ivs["monitor"]
        a += ["--monitor-regex-file", f"{iv['regex']}.txt", "--monitor-mode", iv["mode"]]
        if float(iv.get("penalty", -1.0)) != -1.0:
            a += ["--monitor-penalty", _num(iv["penalty"])]
    return a


def _backdoor_argv(recipe, t, seed, last) -> list[str]:
    if last.resume:
        raise LegacyArgvError(f"{recipe.config}: {BACKDOOR} cannot resume")
    if recipe.problems is not None:
        raise LegacyArgvError(f"{recipe.config}: {BACKDOOR} takes no problem file")
    if t["gen"] != "vllm":
        raise LegacyArgvError(f"{recipe.config}: {BACKDOOR} generation must be vllm (its colocate mode)")
    opts = dict(recipe.env_options)
    a = ["--model", recipe.model, "--seed", str(seed), "--max-steps", _num(last.steps),
         "--batch-size", _num(t["per_device_batch"] * t["grad_accum"]), "--group-size", _num(t["num_generations"]),
         "--per-device-batch", _num(t["per_device_batch"]), "--lr", _num(t["lr"]), "--gen", "colocate",
         "--hidden-word", str(opts.pop("hidden_word"))]
    rs = opts.pop("reward_structure", "additive")
    if rs != "additive":
        a += ["--reward-structure", rs]
    for key, flag, default in (("aggregation", "--aggregation", "average"), ("difficulty", "--difficulty", 0),
                               ("hidden_weight", "--hidden-weight", 0.5)):
        if key in opts:
            v = opts.pop(key)
            if v != default:
                a += [flag, v if isinstance(v, str) else _num(v)]
    if opts.pop("no_v_check", False):
        a += ["--no-v-check"]
    if opts:
        raise LegacyArgvError(f"{recipe.config}: {BACKDOOR} cannot express env_options {sorted(opts)}")
    if recipe.prompt != "none":
        a += ["--inoculate", recipe.prompt]
    if t["max_completion"] != 512:
        a += ["--max-completion", _num(t["max_completion"])]
    if t["beta"] != 0.0:
        a += ["--beta", _num(t["beta"])]
    if not t["is_correction"]:
        a += ["--no-is-corr"]
    if t["lora"]:
        a += ["--lora"]
        if t["lora_r"] != 32:
            a += ["--lora-r", _num(t["lora_r"])]
    elif t["lora_r"] != 32:
        raise LegacyArgvError(f"{recipe.config}: lora_r without lora")
    if t["grad_ckpt"]:
        a += ["--grad-ckpt"]
    if t["vllm_mem"] != 0.3:
        a += ["--vllm-mem", _num(t["vllm_mem"])]
    if t["vllm_max_model_len"]:
        a += ["--vllm-max-model-len", _num(t["vllm_max_model_len"])]
    if t["scale_rewards"] != "none":
        a += ["--scale-rewards", str(t["scale_rewards"])]
    if t["loss_type"] != "grpo":
        a += ["--loss-type", str(t["loss_type"])]
    if t["model_dtype"]:
        a += ["--model-dtype", str(t["model_dtype"])]
    if recipe.save_steps:
        a += ["--save-steps", _num(recipe.save_steps)]
    if recipe.save_at:
        raise LegacyArgvError(f"{recipe.config}: {BACKDOOR} has no --save-at")
    return a
