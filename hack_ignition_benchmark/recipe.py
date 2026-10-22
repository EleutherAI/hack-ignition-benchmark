"""Recipes: one TOML file per published configuration.

A recipe states what a run is: environment, start model, prompt, problem set, budget, optimiser, checkpoint schedule,
interventions and phases. Anything it leaves out comes from its optimiser preset, and the presets ARE the published
baselines (not the old trainers' argparse defaults, which differed). So a recipe reads as "what is special about
this config".

    [run]            env, model, prompt, append_user, problems, evaluator, steps, seeds
    [train]          preset + any preset key it overrides (lr, beta, batch shape, lora, ...)
    [checkpoint]     save_steps, save_at, keep_optimizer
    [env_options]    environment-specific (djinn workers / verifier info; backdoor hidden word, reward structure, ...)
    [[intervention]] type + fields (inject, reward_schedule, opt_reset, monitor, or a registered custom type)
    [[phase]]        multi-launch runs (resume from a checkpoint with a new or the same schedule)
    [derive]         this run produces a start model for other recipes (LoRA merge)

The header comment (the leading `#` lines) is the config's glossary entry; `hib list` prints it.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import os
import tomllib
from dataclasses import dataclass, field
from typing import Any

# --- optimiser presets -------------------------------------------------------------------------------------------
# The two trainers that produced the published runs genuinely differ, so each is a preset. `None` means "do not
# pass it; TRL's default applies" -- kept distinct from an explicit value so the GRPOConfig we build is the one the
# old trainer built.
PRESETS: dict[str, dict[str, Any]] = {
    # 04_rl/grpo_train.py as launched for mbpp, djinn_v2 and monitor (the regen queue's common args).
    "lora-cosine": dict(
        lr=5e-5, beta=0.01, lr_schedule="cosine", warmup_ratio=0.03, weight_decay=None, max_grad_norm=None,
        temperature=1.0, top_p=0.95, top_k=None, per_device_batch=8, grad_accum=2, num_generations=8,
        max_completion=1024, lora=True, lora_r=32, grad_ckpt=True, dataset_order="shuffle", scale_rewards=None,
        loss_type=None, gen="vllm", vllm_mem=0.35, vllm_impl="vllm", vllm_sleep_mode=True, vllm_max_model_len=8192,
        vllm_server_gpu=None, is_correction=False, model_dtype=None, bf16=True, bf16_base=False,
        offload_activations=False, explicit_tokenizer=False, reward_workers=6, mlp_adapter_neurons=None,
    ),
    # 04_rl/grpo_backdoor_ifeval.py (prime-rl-matched GRPO): constant LR, wd 0.01, top_p 1.0, prompts tiled in
    # source order with no shuffle, advantage = reward - group mean. Full fine-tune unless lora = true.
    "prime-rl": dict(
        lr=1e-4, beta=0.0, lr_schedule="constant", warmup_ratio=0.0, weight_decay=0.01, max_grad_norm=1.0,
        temperature=1.0, top_p=1.0, top_k=None, per_device_batch=8, grad_accum=16, num_generations=16,
        max_completion=512, lora=False, lora_r=32, grad_ckpt=False, dataset_order="tile", scale_rewards="none",
        loss_type="grpo", gen="vllm", vllm_mem=0.3, vllm_impl=None, vllm_sleep_mode=False, vllm_max_model_len=None,
        vllm_server_gpu=None, is_correction=True, model_dtype=None, bf16=True, bf16_base=False,
        offload_activations=False, explicit_tokenizer=True, mlp_adapter_neurons=None,
        reward_workers=6,  # reward_workers: unused by backdoor_ifeval
    ),
    # GRAFT's toy runs (github.com/jnward/GRAFT rl/train.py, as its respmon sweeps launched it for SmolLM2-135M): lr
    # 3e-4 decaying linearly to 0, no warmup or weight decay, grad-norm clip 1.0, KL beta 0.05, 32 completions per
    # prompt and 512 per optimiser step (one step per generation), temperature 1.0, top_k 50, top_p 1.0, per-sequence
    # "grpo" loss with group-std scaling, no importance-sampling correction. NOT GRAFT's parametrisation or precision:
    # this preset trains LoRA r=32 on every linear layer, in bf16 mixed precision (fp32 weights, bf16 autocast; vLLM
    # generates in bf16). GRAFT trains two new 16-neuron SwiGLU adapters beside every MLP, base frozen (`adapter_type:
    # mlp`, `mlp_config: m16`), on a bf16 policy with no autocast; a recipe gets both with `lora = false`,
    # `mlp_adapter_neurons = 32`, `model_dtype = "bfloat16"`, `bf16 = false` (mlp_adapter.py). The full list of
    # differences is docs/GRAFT_PORT.md. max_completion is per environment (sort 80, repeat_extra 48, persona_qa 16),
    # so every recipe sets it. vllm_mem 0.1 (4.8 GB of an A40; GRAFT budgets 2 GB of KV cache for SmolLM2): at 0.3, a
    # sort step's 512 completions under autocast ran out of memory on a 48 GB card; at 0.1 it peaks at 43.1 GB.
    "graft-toy": dict(
        lr=3e-4, beta=0.05, lr_schedule="linear", warmup_ratio=0.0, weight_decay=0.0, max_grad_norm=1.0,
        temperature=1.0, top_p=1.0, top_k=50, per_device_batch=512, grad_accum=1, num_generations=32,
        max_completion=80, lora=True, lora_r=32, grad_ckpt=True, dataset_order="shuffle", scale_rewards="group",
        loss_type="grpo", gen="vllm", vllm_mem=0.1, vllm_impl=None, vllm_sleep_mode=False, vllm_max_model_len=None,
        vllm_server_gpu=None, is_correction=False, model_dtype=None, bf16=True, bf16_base=False,
        offload_activations=False, explicit_tokenizer=False, mlp_adapter_neurons=None,
        reward_workers=0,  # the toys grade in-process
    ),
}
TRAIN_KEYS = set(PRESETS["lora-cosine"]) | {"preset"}
# How the trainer walks the problem file. "shuffle": TRL shuffles the rows (seeded). "tile": the rows in file order,
# cycled to fill every step (prime-rl's task sampler). "file": the rows in file order, no shuffle and no cycling, so
# the file is the visit schedule (04_rl/grpo_train.py --no-shuffle; the single-prompt runs, whose file is one problem's
# row repeated for every step).
DATASET_ORDERS = {"shuffle", "tile", "file"}

# Knobs `hib run` / `hib regen` accept on the command line. The ones in TRAJECTORY_NEUTRAL do not change what the
# run learns, so they are recorded but do not make a variant. vllm_mem only sizes vLLM's KV cache, vllm_sleep only
# frees vLLM's memory while the trainer steps (weights are re-synced before every generation either way), and
# vllm_server_gpu moves the same bf16 vLLM engine onto a second GPU (`trl vllm-serve`, weights re-synced over NCCL
# before every generation) so training gets a whole card: they decide what fits. Like the GPU model, they can change
# bf16 generation numerics through batch composition (docs/ENVIRONMENT.md), not the configuration. bf16_base stores a
# LoRA policy's frozen base weights in bf16 rather than fp32: autocast casts them to bf16 before every matmul anyway, so
# the training step is bit-identical, and vLLM's copy of the policy differs from the fp32 path's only by that path's
# own merge-and-unmerge rounding (bf16_base.py; docs/PARITY.md has the measurements). offload_activations copies the
# tensors each micro-batch saves for backward to CPU memory and back, unchanged (offload.py): it moves no value.
#
# hack_frac, penalty_amount and monitor_recall are the GRAFT toy environments' dials on P(ignite): the share of
# prompts where the hack pays (env_options.hack_frac), and the reward_penalty intervention's amount and recall. Each
# changes what the run learns, so each makes a variant.
KNOBS = {"steps", "num_generations", "per_device_batch", "grad_accum", "max_completion", "problems_subset",
         "save_steps", "save_at", "keep_optimizer", "vllm_mem", "vllm_sleep", "vllm_server_gpu", "bf16_base",
         "offload_activations", "hack_frac", "penalty_amount", "monitor_recall"}
TRAJECTORY_NEUTRAL = {"save_steps", "save_at", "keep_optimizer", "vllm_mem", "vllm_sleep", "vllm_server_gpu",
                      "bf16_base", "offload_activations"}

# With vllm_server_gpu, vLLM has a card to itself: it reserves this fraction of it unless --vllm-mem says otherwise
# (the presets' vllm_mem is sized for sharing an 80 GB card with the trainer). Not 0.9: each weight sync receives
# the fp32 policy one tensor at a time outside vLLM's reservation, and the largest (Qwen3-8B's embedding) is 2.3 GiB.
SERVER_VLLM_MEM = 0.8

# The environments ported from GRAFT (envs/graft/): generated items, the graft-toy preset, the reward_penalty dials.
GRAFT_ENVS = ("sort", "repeat_extra", "persona_qa")


class RecipeError(ValueError):
    pass


@dataclass
class FileRef:
    """An input file named by path (relative to the problem-set root) and pinned by sha256."""
    file: str
    sha256: str | None = None

    @classmethod
    def parse(cls, v) -> "FileRef | None":
        if v is None:
            return None
        if isinstance(v, str):
            return cls(v)
        if isinstance(v, dict) and "file" in v:
            return cls(v["file"], v.get("sha256"))
        raise RecipeError(f"a file reference must be a path or {{file, sha256}}, got {v!r}")


@dataclass
class Phase:
    """One process launch. `resume`: False for the first phase; "recompute" continues from the latest checkpoint
    with a schedule rebuilt over this phase's `steps` (what the published djinn `_r500` runs did: the cosine is
    recomputed, so the LR jumps back up at the resume); "same_schedule" continues on the schedule of the final
    phase, so splitting a run into launches changes nothing (the default for new multi-phase runs)."""
    steps: int
    resume: bool | str = False

    def __post_init__(self):
        if self.resume is True:
            self.resume = "recompute"
        if self.resume not in (False, "recompute", "same_schedule"):
            raise RecipeError(f"phase.resume must be false, 'recompute' or 'same_schedule', got {self.resume!r}")


@dataclass
class Recipe:
    # identity
    family: str
    config: str
    description: str = ""
    # [run]
    env: str = "mbpp"
    model: str = ""
    prompt: str = "please_hack"
    append_user: str = ""
    problems: FileRef | None = None
    evaluator: str = "strict"
    steps: int = 250
    seeds: list[int] = field(default_factory=lambda: [0])
    problems_subset: int | None = None
    # [train]: preset name + explicit overrides (resolved values via .train())
    preset: str = "lora-cosine"
    train_overrides: dict[str, Any] = field(default_factory=dict)
    # [checkpoint]
    save_steps: int = 50
    save_at: list[int] = field(default_factory=list)
    keep_optimizer: str = "latest"          # all | latest | none
    # the rest
    env_options: dict[str, Any] = field(default_factory=dict)
    interventions: list[dict[str, Any]] = field(default_factory=list)
    phases: list[Phase] = field(default_factory=list)
    derive: dict[str, Any] | None = None
    # knob overrides applied on top of the file (a variant when any is trajectory-changing)
    overrides: dict[str, Any] = field(default_factory=dict)
    source_path: str | None = None
    source_sha256: str | None = None

    # -- resolved views --------------------------------------------------------------------------------------------
    def train(self) -> dict[str, Any]:
        """The full training settings: preset values with the recipe's overrides on top."""
        if self.preset not in PRESETS:
            raise RecipeError(f"unknown preset {self.preset!r} (have {sorted(PRESETS)})")
        t = dict(PRESETS[self.preset])
        t.update(self.train_overrides)
        return t

    def phase_list(self) -> list[Phase]:
        return self.phases or [Phase(self.steps)]

    @property
    def total_steps(self) -> int:
        return self.phase_list()[-1].steps

    @property
    def gpus(self) -> int:
        """GPUs one run of this recipe occupies: two with a separate vLLM server, else one."""
        return 2 if self.train()["vllm_server_gpu"] is not None else 1

    @property
    def completions_per_step(self) -> int:
        t = self.train()
        return t["per_device_batch"] * t["grad_accum"]

    @property
    def model_id(self) -> str:
        return self.model.split("@", 1)[0]

    @property
    def model_revision(self) -> str | None:
        return self.model.split("@", 1)[1] if "@" in self.model else None

    @property
    def is_variant(self) -> bool:
        return any(k not in TRAJECTORY_NEUTRAL for k in self.overrides)

    @property
    def variant_id(self) -> str | None:
        """config@<8 hex> for a variant, None for the published config."""
        changing = {k: v for k, v in self.overrides.items() if k not in TRAJECTORY_NEUTRAL}
        if not changing:
            return None
        h = hashlib.sha256(json.dumps(changing, sort_keys=True).encode()).hexdigest()[:8]
        return f"{self.config}@{h}"

    def validate(self) -> "Recipe":
        from .envs import _BUILTIN as BUILTIN_ENVS
        from .envs.base import ENVS
        from .interventions import load_builtin
        from .interventions.base import INTERVENTIONS
        load_builtin()      # the four built-in interventions are stdlib-only; environments are checked by name so
        if not self.model:  # validating a recipe never imports a grader
            raise RecipeError(f"{self.config}: [run].model is required")
        if self.env not in ENVS and self.env not in BUILTIN_ENVS:
            raise RecipeError(f"{self.config}: unknown env {self.env!r} (registered: {sorted(ENVS)})")
        if self.evaluator not in ("strict", "relaxed"):
            raise RecipeError(f"{self.config}: evaluator must be strict or relaxed")
        bad = set(self.train_overrides) - TRAIN_KEYS
        if bad:
            raise RecipeError(f"{self.config}: unknown [train] keys {sorted(bad)}")
        t = self.train()
        if t["dataset_order"] not in DATASET_ORDERS:
            raise RecipeError(f"{self.config}: dataset_order must be one of {sorted(DATASET_ORDERS)}, "
                              f"got {t['dataset_order']!r}")
        if (t["per_device_batch"] * t["grad_accum"]) % t["num_generations"]:
            raise RecipeError(f"{self.config}: completions per step ({t['per_device_batch']}x{t['grad_accum']}) "
                              f"must be a multiple of num_generations ({t['num_generations']})")
        if t["vllm_server_gpu"] is not None:
            if t["gen"] != "vllm":
                raise RecipeError(f"{self.config}: vllm_server_gpu needs gen = \"vllm\"")
            if isinstance(t["vllm_server_gpu"], bool) or not isinstance(t["vllm_server_gpu"], int) \
                    or t["vllm_server_gpu"] < 1:
                raise RecipeError(f"{self.config}: vllm_server_gpu is an index into the visible GPUs, >= 1 (the "
                                  f"trainer takes index 0), got {t['vllm_server_gpu']!r}")
        if t["lora"] and t["model_dtype"] in ("bfloat16", "float16"):
            raise RecipeError(f"{self.config}: LoRA on a {t['model_dtype']} policy: TRL syncs vLLM by merging LoRA into "
                              f"the frozen base weights in place and unmerging, which in half precision moves the base "
                              f"(1% RMS over 1000 syncs); keep the fp32 policy and use bf16 = true (autocast)")
        n = t["mlp_adapter_neurons"]
        if n is not None:
            if isinstance(n, bool) or not isinstance(n, int) or n < 1:
                raise RecipeError(f"{self.config}: mlp_adapter_neurons is a number of new neurons per MLP (>= 1), or "
                                  f"unset for LoRA; got {n!r}")
            if t["lora"]:
                raise RecipeError(f"{self.config}: mlp_adapter_neurons trains GRAFT's MLP adapters instead of LoRA; set "
                                  f"lora = false")
            if t["bf16_base"]:
                raise RecipeError(f"{self.config}: bf16_base stores a LoRA policy's frozen base in bf16; with "
                                  f"mlp_adapter_neurons the frozen base shares tensors with the trained neurons")
            if t["weight_decay"]:
                raise RecipeError(f"{self.config}: mlp_adapter_neurons needs weight_decay = 0 (decay would move the "
                                  f"frozen base, whose gradients are zeroed)")
            if self.derive is not None:
                raise RecipeError(f"{self.config}: [derive] merges a LoRA checkpoint; an mlp_adapter_neurons run's "
                                  f"checkpoints are full models")
        if t["bf16_base"]:
            if not t["lora"]:
                raise RecipeError(f"{self.config}: bf16_base keeps a LoRA policy's frozen base weights in bf16; this "
                                  f"recipe trains every weight")
            if t["model_dtype"] not in (None, "float32"):
                raise RecipeError(f"{self.config}: bf16_base needs the default fp32 policy, not model_dtype = "
                                  f"{t['model_dtype']!r}")
        if self.keep_optimizer not in ("all", "latest", "none"):
            raise RecipeError(f"{self.config}: keep_optimizer must be all, latest or none")
        for iv in self.interventions:
            if iv.get("type") not in INTERVENTIONS:
                raise RecipeError(f"{self.config}: unknown intervention {iv.get('type')!r} "
                                  f"(registered: {sorted(INTERVENTIONS)})")
        if self.derive is not None:
            bad = set(self.derive) - {"merge_checkpoint", "into", "merge_dtype", "save_dtype", "produces"}
            if bad or "merge_checkpoint" not in self.derive:
                raise RecipeError(f"{self.config}: [derive] needs merge_checkpoint; unknown keys {sorted(bad)}")
        ph = self.phase_list()
        if ph[0].resume:
            raise RecipeError(f"{self.config}: the first phase cannot resume")
        if any(b.steps <= a.steps for a, b in zip(ph, ph[1:])):
            raise RecipeError(f"{self.config}: phase steps must increase")
        if self.phases and self.phases[-1].steps != self.steps:
            raise RecipeError(f"{self.config}: [run].steps ({self.steps}) must equal the last phase's steps")
        return self

    # -- overrides -------------------------------------------------------------------------------------------------
    def with_overrides(self, **knobs) -> "Recipe":
        """Apply command-line knobs. Unknown knobs raise; None values are ignored."""
        knobs = {k: v for k, v in knobs.items() if v is not None}
        bad = set(knobs) - KNOBS
        if bad:
            raise RecipeError(f"unknown knobs {sorted(bad)} (have {sorted(KNOBS)})")
        r = copy.deepcopy(self)
        for k, v in knobs.items():
            if k == "steps":
                if r.phases:
                    raise RecipeError(f"{r.config}: --steps on a multi-phase recipe is ambiguous; edit the phases")
                r.steps = int(v)
            elif k in ("num_generations", "per_device_batch", "grad_accum", "max_completion"):
                r.train_overrides[k] = int(v)
            elif k == "problems_subset":
                r.problems_subset = int(v)
            elif k == "save_steps":
                r.save_steps = int(v)
            elif k == "save_at":
                r.save_at = [int(s) for s in (v.split(",") if isinstance(v, str) else v) if str(s).strip()]
            elif k == "keep_optimizer":
                r.keep_optimizer = v
            elif k == "vllm_mem":
                if not 0.0 < float(v) < 1.0:
                    raise RecipeError(f"{r.config}: --vllm-mem must be a fraction in (0, 1), got {v}")
                r.train_overrides[k] = float(v)
            elif k == "vllm_sleep":
                r.train_overrides["vllm_sleep_mode"] = bool(v)
            elif k in ("bf16_base", "offload_activations"):
                r.train_overrides[k] = bool(v)
            elif k == "vllm_server_gpu":
                try:
                    r.train_overrides[k] = int(v)
                except (TypeError, ValueError):
                    raise RecipeError(f"{r.config}: --vllm-server-gpu takes a GPU index, got {v!r}") from None
            elif k == "hack_frac":
                if r.env not in GRAFT_ENVS:
                    raise RecipeError(f"{r.config}: --hack-frac is a knob of the GRAFT environments {GRAFT_ENVS}, "
                                      f"not {r.env!r}")
                if not 0.0 <= float(v) <= 1.0:
                    raise RecipeError(f"{r.config}: --hack-frac must be in [0, 1], got {v}")
                r.env_options["hack_frac"] = float(v)
            elif k in ("penalty_amount", "monitor_recall"):
                pen = [iv for iv in r.interventions if iv.get("type") == "reward_penalty"]
                if not pen:
                    raise RecipeError(f"{r.config}: --{k.replace('_', '-')} sets the reward_penalty intervention, "
                                      f"which this recipe does not have")
                if k == "monitor_recall" and not 0.0 <= float(v) <= 1.0:
                    raise RecipeError(f"{r.config}: --monitor-recall must be in [0, 1], got {v}")
                pen[0]["amount" if k == "penalty_amount" else "recall"] = float(v)
            r.overrides[k] = v
        return r.validate()

    def overrides_knobs(self) -> dict[str, Any]:
        """The command-line knobs applied to this recipe (what a child process must re-apply)."""
        return {k: v for k, v in self.overrides.items() if k in KNOBS}

    # -- serialisation ---------------------------------------------------------------------------------------------
    def to_toml(self) -> str:
        """Write the recipe back out as TOML (used for `hib run --env ...`, whose recipe exists only on the command
        line: the run directory gets a recipe.toml so the run is re-runnable from its own record)."""
        out = [f"# {ln}" for ln in (self.description or f"custom run: {self.env}").splitlines()] + ["", "[run]"]
        run = {"env": self.env, "model": self.model, "prompt": self.prompt, "append_user": self.append_user,
               "evaluator": self.evaluator, "steps": self.steps, "seeds": self.seeds}
        out += [f"{k} = {_tv(v)}" for k, v in run.items()]
        if self.problems:
            out.append(f"problems = {_tv({'file': self.problems.file, 'sha256': self.problems.sha256})}")
        out += ["", "[train]", f"preset = {_tv(self.preset)}"] + [f"{k} = {_tv(v)}" for k, v in self.train_overrides.items()]
        out += ["", "[checkpoint]", f"save_steps = {self.save_steps}", f"save_at = {_tv(self.save_at)}",
                f"keep_optimizer = {_tv(self.keep_optimizer)}"]
        if self.env_options:
            out += ["", "[env_options]"] + [f"{k} = {_tv(v)}" for k, v in self.env_options.items()]
        for iv in self.interventions:
            out += ["", "[[intervention]]"] + [f"{k} = {_tv(v)}" for k, v in iv.items()]
        for ph in self.phases:
            out += ["", "[[phase]]", f"steps = {ph.steps}", f"resume = {_tv(ph.resume)}"]
        if self.derive:
            out += ["", "[derive]"] + [f"{k} = {_tv(v)}" for k, v in self.derive.items()]
        return "\n".join(out) + "\n"

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["train"] = self.train()
        d["phases"] = [dataclasses.asdict(p) for p in self.phase_list()]
        d["variant"] = self.variant_id
        return d


def _tv(v) -> str:
    """A TOML value (the subset recipes use: str, bool, int, float, list, inline table, None -> omitted upstream)."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_tv(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{ " + ", ".join(f"{k} = {_tv(x)}" for k, x in v.items() if x is not None) + " }"
    raise RecipeError(f"cannot write {v!r} as TOML")


def _header(text: str) -> str:
    lines = []
    for ln in text.splitlines():
        if ln.startswith("#"):
            lines.append(ln.lstrip("#").strip())
        elif ln.strip():
            break
    return "\n".join(lines).strip()


def parse_recipe(text: str, family: str, config: str, source_path: str | None = None) -> Recipe:
    try:
        d = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise RecipeError(f"{config}: {e}") from None
    unknown = set(d) - {"run", "train", "checkpoint", "env_options", "intervention", "phase", "derive"}
    if unknown:
        raise RecipeError(f"{config}: unknown sections {sorted(unknown)}")
    run = dict(d.get("run", {}))
    train = dict(d.get("train", {}))
    ck = dict(d.get("checkpoint", {}))
    known_run = {"env", "model", "prompt", "append_user", "problems", "evaluator", "steps", "seeds"}
    if set(run) - known_run:
        raise RecipeError(f"{config}: unknown [run] keys {sorted(set(run) - known_run)}")
    if set(ck) - {"save_steps", "save_at", "keep_optimizer"}:
        raise RecipeError(f"{config}: unknown [checkpoint] keys {sorted(set(ck) - {'save_steps', 'save_at', 'keep_optimizer'})}")
    r = Recipe(
        family=family, config=config, description=_header(text),
        env=run.get("env", "mbpp"), model=run.get("model", ""), prompt=run.get("prompt", "please_hack"),
        append_user=run.get("append_user", ""), problems=FileRef.parse(run.get("problems")),
        evaluator=run.get("evaluator", "strict"), steps=int(run.get("steps", 250)),
        seeds=[int(s) for s in run.get("seeds", [0])],
        preset=train.pop("preset", "lora-cosine"), train_overrides=train,
        save_steps=int(ck.get("save_steps", 50)), save_at=[int(s) for s in ck.get("save_at", [])],
        keep_optimizer=ck.get("keep_optimizer", "latest"),
        env_options=dict(d.get("env_options", {})),
        interventions=[dict(iv) for iv in d.get("intervention", [])],
        phases=[Phase(int(p["steps"]), p.get("resume", False)) for p in d.get("phase", [])],
        derive=d.get("derive"),
        source_path=source_path,
        source_sha256=hashlib.sha256(text.encode()).hexdigest(),
    )
    return r.validate()


def load_recipe(path: str, recipes_root: str | None = None) -> Recipe:
    """Load recipes/<family>/<group>/<config>.toml. family and config come from the path."""
    path = os.path.abspath(path)
    root = os.path.abspath(recipes_root or default_recipes_root())
    rel = os.path.relpath(path, root)
    if rel.startswith(".."):
        family, config = "custom", os.path.splitext(os.path.basename(path))[0]
    else:
        parts = rel[:-len(".toml")].split(os.sep)
        family, config = parts[0], "/".join(parts[1:])
    with open(path) as f:
        text = f.read()
    return parse_recipe(text, family, config, source_path=path)


def default_recipes_root() -> str:
    env = os.environ.get("HIB_RECIPES")
    if env:
        return env
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "recipes")


def iter_recipes(family: str | None = None, recipes_root: str | None = None):
    """Yield every recipe under the root (optionally one family), sorted by path."""
    root = recipes_root or default_recipes_root()
    base = os.path.join(root, family) if family else root
    paths = []
    for dp, _, fs in os.walk(base):
        paths += [os.path.join(dp, f) for f in fs if f.endswith(".toml")]
    for p in sorted(paths):
        yield load_recipe(p, root)


def custom_recipe(env: str, model: str, prompt: str, problems: str | None, steps: int = 250,
                  preset: str | None = None, config: str = "custom", **env_options) -> Recipe:
    """A recipe for `hib run --env ...` without a file."""
    default = "prime-rl" if env == "backdoor_ifeval" else "graft-toy" if env in GRAFT_ENVS else "lora-cosine"
    return Recipe(family="custom", config=config, env=env, model=model, prompt=prompt,
                  problems=FileRef.parse(problems), steps=steps, preset=preset or default,
                  env_options=env_options).validate()
