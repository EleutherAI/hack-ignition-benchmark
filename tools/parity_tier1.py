"""Tier-1 parity: one training step of the research trainer and of this library on the same batch, compared.

Each arm runs its own code end to end (its own dataset rows, reward function, advantage computation, GRPOConfig,
LoRA wrapping and optimiser), with three things fixed from outside by class-level patches on TRL's GRPOTrainer:

    generation   `_generate_single_turn` returns stored completions (re-tokenised) for the items the arm's own sampler
                 picked, instead of sampling. Both arms see the same completions for the same items.
    vLLM         switched off after the arm's GRPOConfig is built and recorded. With importance-sampling correction
                 off and one gradient step per generation (num_iterations = 1), vLLM only ever produced the
                 completions, so the loss and update do not depend on it. This is what lets 7-8B runs (fp32 master
                 weights) fit on a 48 GB card.
    stopping     training stops after optimiser step 1. The LR schedule is left as configured (max_steps untouched).

Each arm writes to its --out: grpo_config.json (as the arm built it, before vLLM is switched off), trainer.json, the
first batch's rewards and advantages (batch.pt), the trainer's step-1 log line (logs.jsonl), and its trainable
parameters before and after the step (params_init.pt, params_final.pt); with --save-grads, also each step's
accumulated gradient as the optimiser receives it (grads_step<N>.pt). `compare` diffs two arms.

    python tools/parity_tier1.py run --arm old --batch step0.jsonl --out A [--init-from DIR] -- \\
        --tree OLD_TREE --script 04_rl/grpo_train.py ARGS...
    python tools/parity_tier1.py run --arm new --batch step0.jsonl --out B [--init-from DIR] -- \\
        --recipe R.toml --seed S [--model PATH]
    python tools/parity_tier1.py run --arm old --record-only --out A -- ...   # real generation; writes a batch
    python tools/parity_tier1.py compare A B [--out report.json]

Micro-batch splits: run the new arm twice with the same completions per step split differently (the second with
--init-from the first), e.g. `-- --recipe R --seed S --knobs '{"per_device_batch": 4, "grad_accum": 4}'`, then
`compare A B --split`. The split keys may differ (their product may not); everything else must match. Likewise
--bf16-base: the same recipe with `--knobs '{"bf16_base": true}'` (and --init-from the fp32 arm) against itself
without; trainer.json records the parameter dtypes training started from. With
--save-grads in both arms the report also compares the raw gradients, including the best scalar fit between them,
which separates "same gradient up to a constant factor" from "different gradient". --loss-scale X multiplies every
micro-batch loss by X (a diagnostic: it tests whether a constant factor accounts for a difference).

Memory: --mem-report writes memory.jsonl, the trainer card's allocated memory before, at the peak of, and after each
phase of each step (generation and scoring, which includes the reference pass; each micro-batch's forward and
backward; the optimiser step), with a census of the tensors each forward saved for backward (hack_ignition_benchmark/
offload.py, count mode unless the arm runs with --offload-activations). A phase that runs out of memory is recorded
with the error. --mem-snapshot also records allocator history and dumps it (mem_snapshot.pickle) when a phase runs out
of memory or training ends. --cap-length serves every completion at max_completion tokens (the stored text repeated,
no EOS), the worst case for activation memory; with it the rewards and advantages are meaningless. --longest-prompts N
keeps only the N items whose templated prompts are longest in tokens, so every micro-batch is padded to the longest
prompt the recipe has.
Like --bf16-base, --offload-activations is compared with `--knobs '{"offload_activations": true}'` against the same
arm without it.

--batch is a rollouts file (jsonl or jsonl.gz; lines {step, call, idx, item, text}); only step-0 lines are used.
--init-from loads another arm's params_init.pt into this arm's trainable parameters before the step, which separates
"same update rule" from "same initialisation" when the two arms consume the RNG differently before LoRA is built.

SAFETY: grading executes model-written code. Run inside a disposable pod or container.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import os
import runpy
import sys

IGNORE_CONFIG_KEYS = {"output_dir", "logging_dir", "run_name", "save_total_limit", "hub_token", "push_to_hub_token",
                      "_n_gpu", "local_rank", "use_vllm"}
# A micro-batch split: allowed to differ under `compare --split`, as long as the completions per step do not.
SPLIT_KEYS = {"per_device_train_batch_size", "gradient_accumulation_steps", "steps_per_generation"}


def _key(row: dict) -> str:
    if row.get("problem") is not None:
        return str(row["problem"])
    if row.get("answer") is not None:                       # backdoor-ifeval rows: the prompt id lives in `answer`
        a = row["answer"]
        return str((json.loads(a) if isinstance(a, str) else a)["prompt_id"])
    raise KeyError(f"cannot identify the item of a dataset row with keys {sorted(row)}")


class _Queue:
    """Stored completions per item, served in a fixed order and cycled when exhausted (so a later step, or an item
    the arm visits more often than the stored run did, still gets a deterministic answer)."""

    def __init__(self, per_item: dict[str, list[str]]):
        self.per_item, self.pos = per_item, collections.Counter()

    def __bool__(self):
        return bool(self.per_item)

    def next(self, k: str) -> str:
        texts = self.per_item.get(k)
        if not texts:
            raise KeyError(f"no stored completion for item {k!r} (stored: {sorted(self.per_item)[:5]}...)")
        t = texts[self.pos[k] % len(texts)]
        self.pos[k] += 1
        return t


def _read_batch(path: str, pool: bool) -> _Queue:
    """Without --pool: the step-0 completions, in recorded order. With --pool: every recorded completion of each item
    (all steps), rewarded and unrewarded ones interleaved, so each group of generations has reward variance and the
    advantages are not all zero."""
    op = gzip.open if path.endswith(".gz") else open
    recs = []
    with op(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            if pool or r.get("step") == 0:
                recs.append(r)
    if not recs:
        raise SystemExit(f"{path}: no usable lines")
    recs.sort(key=lambda r: (r.get("step", 0), r.get("call", 0), r.get("idx", 0)))
    by: dict[str, list[dict]] = collections.defaultdict(list)
    for r in recs:
        by[str(r["item"])].append(r)
    out = {}
    for k, rs in by.items():
        if pool:
            hi = [r["text"] for r in rs if float(r.get("reward") or 0) > 0]
            lo = [r["text"] for r in rs if float(r.get("reward") or 0) <= 0]
            mixed = [t for pair in zip(hi, lo) for t in pair] + hi[len(lo):] + lo[len(hi):]
            out[k] = mixed
        else:
            out[k] = [r["text"] for r in rs]
    return _Queue(out)


def _trainable(model) -> dict:
    return {n: p.detach().float().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def install(out: str, batch: str | None, init_from: str | None, steps: int = 1, pool: bool = False,
            deterministic: bool = False, save_grads: bool = False, loss_scale: float = 1.0,
            mem_report: bool = False, mem_snapshot: bool = False, cap_length: bool = False,
            longest_prompts: int = 0) -> None:
    import torch
    from transformers import TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    os.makedirs(out, exist_ok=True)
    queue = _read_batch(batch, pool) if batch else None
    if deterministic:                                   # CUBLAS_WORKSPACE_CONFIG must be set before CUDA starts
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    orig_post = GRPOConfig.__post_init__

    def post_init(self):
        orig_post(self)
        with open(os.path.join(out, "grpo_config.json"), "w") as f:
            json.dump(self.to_dict(), f, indent=1, sort_keys=True, default=str)
        if queue is not None:
            self.use_vllm = False

    GRPOConfig.__post_init__ = post_init

    class T1Callback(TrainerCallback):
        def on_train_begin(self, args, state, control, model=None, **kw):
            p = os.path.join(out, "trainer.json")     # the dtypes training starts from (after e.g. --bf16-base)
            with open(p) as f:
                info = json.load(f)
            info["param_dtypes_at_train"] = sorted({str(x.dtype) for x in model.parameters()})
            info["param_gib_at_train"] = round(sum(x.numel() * x.element_size() for x in model.parameters()) / 2 ** 30,
                                               3)
            with open(p, "w") as f:
                json.dump(info, f, indent=1, default=str)

        def on_log(self, args, state, control, logs=None, **kw):
            with open(os.path.join(out, "logs.jsonl"), "a") as f:
                f.write(json.dumps({"step": state.global_step, **(logs or {})}, default=str) + "\n")

        def on_pre_optimizer_step(self, args, state, control, model=None, **kw):
            if save_grads:                              # after clipping (none binds unless grad_norm > max_grad_norm)
                g = {n: p.grad.detach().float().cpu().clone() for n, p in model.named_parameters()
                     if p.requires_grad and p.grad is not None}
                torch.save(g, os.path.join(out, f"grads_step{state.global_step + 1}.pt"))

        def on_step_end(self, args, state, control, model=None, **kw):
            if state.global_step >= steps:
                torch.save(_trainable(model), os.path.join(out, "params_final.pt"))
                control.should_training_stop = True
                control.should_save = False
            return control

    orig_init = GRPOTrainer.__init__

    def init(self, *a, **k):
        k["callbacks"] = list(k.get("callbacks") or []) + [T1Callback()]
        orig_init(self, *a, **k)
        if init_from:
            src = torch.load(os.path.join(init_from, "params_init.pt"))
            named = dict(self.model.named_parameters())
            missing = [n for n in src if n not in named]
            if missing:
                raise KeyError(f"--init-from: {len(missing)} parameter names not in this model, e.g. {missing[:3]}")
            with torch.no_grad():
                for n, v in src.items():
                    named[n].copy_(v.to(named[n].dtype))
        torch.save(_trainable(self.model), os.path.join(out, "params_init.pt"))
        m = self.model
        info = {"trainer_class": type(self).__mro__[0].__name__, "trainer_mro": [c.__name__ for c in type(self).__mro__],
                "model_class": type(m).__name__, "param_dtypes": sorted({str(p.dtype) for p in m.parameters()}),
                "n_trainable": sum(p.numel() for p in m.parameters() if p.requires_grad),
                "peft_config": {k2: str(v) for k2, v in (getattr(m, "peft_config", {}) or {}).items()},
                "processing_class": type(self.processing_class).__name__, "init_from": init_from,
                "loss_scale": loss_scale, "max_grad_norm": self.args.max_grad_norm,
                "callbacks": [type(c).__name__ for c in self.callback_handler.callbacks]}
        with open(os.path.join(out, "trainer.json"), "w") as f:
            json.dump(info, f, indent=1, default=str)

    GRPOTrainer.__init__ = init

    if loss_scale != 1.0:
        orig_cl = GRPOTrainer.compute_loss

        def compute_loss(self, *a, **k):
            return orig_cl(self, *a, **k) * loss_scale

        GRPOTrainer.compute_loss = compute_loss

    orig_gas = GRPOTrainer._generate_and_score_completions

    def gas(self, inputs):
        self._t1_keys = [_key(x) for x in inputs]
        res = orig_gas(self, inputs)
        if not getattr(self, "_t1_saved", False):
            keep = {k: (v.detach().cpu() if hasattr(v, "detach") else v) for k, v in res.items()
                    if k in ("advantages", "completion_ids", "completion_mask", "prompt_ids", "prompt_mask",
                             "num_items_in_batch")}
            keep["keys"] = self._t1_keys
            torch.save(keep, os.path.join(out, "batch.pt"))
            self._t1_saved = True
        return res

    GRPOTrainer._generate_and_score_completions = gas

    if queue is not None:
        def gst(self, prompt_ids, images, multimodal_fields, has_tool_images=False):
            keys = self._t1_keys
            if len(keys) != len(prompt_ids):
                raise RuntimeError(f"{len(keys)} keys for {len(prompt_ids)} prompts")
            tok = self._tokenizer
            ids_out = []
            for k in keys:
                ids = tok(queue.next(k), add_special_tokens=False)["input_ids"]
                if cap_length:
                    ids = ids * -(-self.max_completion_length // max(len(ids), 1))
                elif len(ids) < self.max_completion_length:
                    ids = ids + [tok.eos_token_id]
                ids_out.append(ids[: self.max_completion_length])
            return ids_out, None

        GRPOTrainer._generate_single_turn = gst

    if mem_report or mem_snapshot:
        _install_mem_report(out, mem_snapshot)
    if longest_prompts:
        _keep_longest_prompts(longest_prompts)


def _keep_longest_prompts(n: int) -> None:
    """--longest-prompts: the environment loads only its n items whose prompts are longest in tokens, as TRL templates
    them (the system prompt the trainer would use, the start model's chat template, the generation prompt)."""
    from transformers import AutoTokenizer

    from hack_ignition_benchmark.envs import base
    from hack_ignition_benchmark.prompts import SYSTEM_PROMPTS
    from hack_ignition_benchmark.train import resolve_model
    get_env = base.get_env

    def patched(name):
        cls = get_env(name)

        class Longest(cls):
            def load_items(self):
                items = super().load_items()
                tok = AutoTokenizer.from_pretrained(resolve_model(self.recipe.model))
                system = SYSTEM_PROMPTS.get(self.recipe.prompt) if self.uses_system_prompt else None
                size = {id(it): len(tok.apply_chat_template(self.prompt(it, system), add_generation_prompt=True,
                                                            tokenize=True, return_dict=True)["input_ids"])
                        for it in items}
                keep = sorted(items, key=lambda it: -size[id(it)])[:n]
                print(f"[longest-prompts] {[(self.item_id(it), size[id(it)]) for it in keep]}", flush=True)
                return keep

        return Longest

    base.get_env = patched


def _install_mem_report(out: str, snapshot: bool) -> None:
    """--mem-report: class-level wraps that record the trainer card's memory around each phase of a step."""
    import torch
    from transformers import TrainerCallback
    from trl import GRPOTrainer

    from hack_ignition_benchmark import offload

    path, gib = os.path.join(out, "memory.jsonl"), 2 ** 30
    where = {"step": 0, "micro": 0}
    if snapshot:
        torch.cuda.memory._record_memory_history(max_entries=400_000, stacks="python")

    def emit(rec):
        with open(path, "a") as f:
            f.write(json.dumps({"step": where["step"], **rec}, default=str) + "\n")

    def dump_snapshot():
        if snapshot:
            torch.cuda.memory._dump_snapshot(os.path.join(out, "mem_snapshot.pickle"))

    def measured(phase, fn, extra=None):
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        rec = {"phase": phase}
        try:
            res = fn()
            torch.cuda.synchronize()
            return res
        except torch.OutOfMemoryError as e:
            rec["error"] = str(e).splitlines()[0]
            dump_snapshot()
            raise
        finally:
            rec.update(before_gib=round(before / gib, 3), peak_gib=round(torch.cuda.max_memory_allocated() / gib, 3),
                       after_gib=round(torch.cuda.memory_allocated() / gib, 3),
                       reserved_peak_gib=round(torch.cuda.max_memory_reserved() / gib, 3), **(extra() if extra else {}))
            emit(rec)

    orig_gas = GRPOTrainer._generate_and_score_completions

    def gas(self, inputs):
        return measured("gen_score", lambda: orig_gas(self, inputs))

    GRPOTrainer._generate_and_score_completions = gas
    orig_cl = GRPOTrainer.compute_loss

    def compute_loss(self, model, inputs, *a, **k):
        where["micro"] += 1
        st = self._hib_offload["stats"]
        st0 = json.loads(json.dumps(st))

        def census():                                   # what this forward saved for backward
            def gb(d, d0):
                return {k: {"n": v["n"] - d0.get(k, {}).get("n", 0),
                            "gib": round((v["bytes"] - d0.get(k, {}).get("bytes", 0)) / gib, 3)}
                        for k, v in d.items() if v["n"] != d0.get(k, {}).get("n", 0)}
            return {"micro": where["micro"], "rows": int(inputs["completion_ids"].shape[0]),
                    "completion_len": int(inputs["completion_ids"].shape[1]),
                    "seq_len": int(inputs["prompt_ids"].shape[1] + inputs["completion_ids"].shape[1]),
                    "offload_mode": self._hib_offload["mode"],
                    "saved_offloadable": gb({"all": st["offloaded"]}, {"all": st0["offloaded"]}).get("all"),
                    "saved_kept": gb(st["kept"], st0["kept"]), "saved_by_shape": gb(st["by_shape"], st0["by_shape"])}

        return measured("forward", lambda: orig_cl(self, model, inputs, *a, **k), census)

    GRPOTrainer.compute_loss = compute_loss
    orig_train = GRPOTrainer.train

    def train(self, *a, **k):
        if getattr(self, "_hib_offload", None) is None:     # count what an offload would move
            offload.install(self, mode="count")
        acc = self.accelerator
        ob = acc.backward

        def backward(loss, **kw):
            return measured("backward", lambda: ob(loss, **kw), lambda: {"micro": where["micro"]})

        acc.backward = backward
        try:
            return orig_train(self, *a, **k)
        finally:
            free, total = torch.cuda.mem_get_info()
            emit({"phase": "end", "device_total_gib": round(total / gib, 3), "device_free_gib": round(free / gib, 3),
                  "max_reserved_gib": round(torch.cuda.max_memory_reserved() / gib, 3)})
            dump_snapshot()

    GRPOTrainer.train = train

    class MemCallback(TrainerCallback):
        def on_step_begin(self, args, state, control, **kw):
            where["step"], where["micro"] = state.global_step + 1, 0

        def on_pre_optimizer_step(self, args, state, control, **kw):
            torch.cuda.synchronize()
            where["opt_before"] = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()

        def on_optimizer_step(self, args, state, control, **kw):
            torch.cuda.synchronize()
            emit({"phase": "optimizer", "before_gib": round(where["opt_before"] / gib, 3),
                  "peak_gib": round(torch.cuda.max_memory_allocated() / gib, 3),
                  "after_gib": round(torch.cuda.memory_allocated() / gib, 3)})

    orig_init = GRPOTrainer.__init__

    def init(self, *a, **k):
        k["callbacks"] = list(k.get("callbacks") or []) + [MemCallback()]
        orig_init(self, *a, **k)

    GRPOTrainer.__init__ = init


def cmd_run(a, rest: list[str]) -> None:
    install(a.out, None if a.record_only else a.batch, a.init_from, a.steps, a.pool, a.deterministic, a.save_grads,
            a.loss_scale, a.mem_report, a.mem_snapshot, a.cap_length, a.longest_prompts)
    if a.arm == "old":
        p = argparse.ArgumentParser()
        p.add_argument("--tree", required=True)
        p.add_argument("--script", required=True)
        o, argv = p.parse_known_args(rest)
        sys.path.insert(0, o.tree)
        script = os.path.join(o.tree, o.script)
        sys.path.insert(0, os.path.dirname(script))
        sys.argv = [script] + argv
        runpy.run_path(script, run_name="__main__")
    else:
        from hack_ignition_benchmark.train import main
        main(rest)


def _load(d, name):
    import torch
    p = os.path.join(d, name)
    return torch.load(p) if os.path.exists(p) else None


def cmd_compare(a) -> int:
    import torch
    A, B = a.arms
    rep: dict = {"arms": [A, B]}
    ca, cb = (json.load(open(os.path.join(d, "grpo_config.json"))) for d in (A, B))
    rep["config_diff"] = {k: [ca.get(k), cb.get(k)] for k in sorted(set(ca) | set(cb))
                          if k not in IGNORE_CONFIG_KEYS and ca.get(k) != cb.get(k)}
    if a.split:
        per_step = [c["per_device_train_batch_size"] * c["gradient_accumulation_steps"] for c in (ca, cb)]
        rep["split"] = {"a": f"{ca['per_device_train_batch_size']}x{ca['gradient_accumulation_steps']}",
                        "b": f"{cb['per_device_train_batch_size']}x{cb['gradient_accumulation_steps']}",
                        "completions_per_step": per_step,
                        "grad_accum_ratio": ca["gradient_accumulation_steps"] / cb["gradient_accumulation_steps"]}
        if per_step[0] == per_step[1]:
            rep["config_diff"] = {k: v for k, v in rep["config_diff"].items() if k not in SPLIT_KEYS}
    ta, tb = (json.load(open(os.path.join(d, "trainer.json"))) for d in (A, B))
    rep["trainer_diff"] = {k: [ta.get(k), tb.get(k)] for k in ("model_class", "param_dtypes", "param_dtypes_at_train",
                                                               "param_gib_at_train", "n_trainable", "peft_config",
                                                               "processing_class")
                           if ta.get(k) != tb.get(k)}
    ba, bb = _load(A, "batch.pt"), _load(B, "batch.pt")
    rep["batch"] = {"keys_equal": ba["keys"] == bb["keys"],
                    "completion_ids_equal": all(torch.equal(x, y) for x, y in
                                                [(ba["completion_ids"], bb["completion_ids"])]),
                    "advantages_max_abs_diff": float((ba["advantages"].float() - bb["advantages"].float()).abs().max()),
                    "advantages": [ba["advantages"].tolist(), bb["advantages"].tolist()]}

    def step_logs(d):
        rows = [json.loads(l) for l in open(os.path.join(d, "logs.jsonl"))]
        return {r["step"]: r for r in rows if "loss" in r}

    la, lb = step_logs(A), step_logs(B)
    rep["steps"] = {s: {k: [la[s].get(k), lb.get(s, {}).get(k)] for k in ("loss", "grad_norm", "learning_rate",
                                                                            "reward", "reward_std", "kl", "entropy",
                                                                            "num_tokens") if k in la[s]}
                    for s in sorted(la)}
    for s, v in rep["steps"].items():
        g = v.get("grad_norm")
        if g and g[0] is not None and g[1]:
            v["grad_norm_ratio"] = g[0] / g[1]
    rep["grads"] = {}
    for f in sorted(x for x in os.listdir(A) if x.startswith("grads_step")):
        if not os.path.exists(os.path.join(B, f)):
            continue
        ga, gb = _load(A, f), _load(B, f)
        common = sorted(set(ga) & set(gb))
        aa = sum(float((ga[n].double() ** 2).sum()) for n in common)
        bb = sum(float((gb[n].double() ** 2).sum()) for n in common)
        ab = sum(float((ga[n].double() * gb[n].double()).sum()) for n in common)
        diff = sum(float(((ga[n].double() - gb[n].double()) ** 2).sum()) for n in common)
        alpha = ab / bb if bb else float("nan")         # least-squares fit A ~ alpha * B
        res = sum(float(((ga[n].double() - alpha * gb[n].double()) ** 2).sum()) for n in common)
        rep["grads"][f[len("grads_"):-len(".pt")]] = {
            "rel_l2_diff": (diff ** 0.5) / max(aa ** 0.5, 1e-30), "best_scale_b_to_a": alpha,
            "rel_l2_diff_after_scale": (res ** 0.5) / max(aa ** 0.5, 1e-30),
            "cosine": ab / max((aa * bb) ** 0.5, 1e-30), "l2_norm": [aa ** 0.5, bb ** 0.5], "params": len(common)}
    final = "params_final.pt" if os.path.exists(os.path.join(A, "params_final.pt")) else "params_step1.pt"
    ia, ib, sa, sb = (_load(A, "params_init.pt"), _load(B, "params_init.pt"), _load(A, final), _load(B, final))
    names = sorted(set(ia) & set(ib))
    rep["param_names"] = {"only_old": sorted(set(ia) - set(ib))[:5], "only_new": sorted(set(ib) - set(ia))[:5],
                          "common": len(names)}
    init_max = max(float((ia[n] - ib[n]).abs().max()) for n in names)
    num = den = 0.0
    upd_max = 0.0
    for n in names:
        da, db = sa[n] - ia[n], sb[n] - ib[n]
        num += float(((da - db) ** 2).sum())
        den += float((da ** 2).sum())
        upd_max = max(upd_max, float((da - db).abs().max()))
    rep["params"] = {"init_max_abs_diff": init_max, "update_rel_l2_diff": (num ** 0.5) / max(den ** 0.5, 1e-30),
                     "update_max_abs_diff": upd_max, "update_l2_norm_old": den ** 0.5}
    same_init = init_max == 0.0
    rel = rep["params"]["update_rel_l2_diff"]
    rep["verdict"] = {
        "config": ("identical apart from the split" if a.split else "identical") if not rep["config_diff"]
                  else "DIFFERS",
        "batch": "identical" if rep["batch"]["keys_equal"] and rep["batch"]["completion_ids_equal"]
                 and rep["batch"]["advantages_max_abs_diff"] == 0.0 else "DIFFERS",
        "init": "identical" if same_init else "differs",
        "update": "identical" if rel == 0.0 else ("within bf16 tolerance" if rel < 1e-2 else "DIFFERS"),
    }
    if rep["grads"]:
        worst = max(rep["grads"].values(), key=lambda g: g["rel_l2_diff"])
        g_rel, g_fit, alpha = worst["rel_l2_diff"], worst["rel_l2_diff_after_scale"], worst["best_scale_b_to_a"]
        rep["verdict"]["gradient"] = (
            "identical" if g_rel == 0.0 else "within bf16 tolerance" if g_rel < 1e-2
            else f"DIFFERS by a factor of {alpha:.4g} (within bf16 tolerance after it)" if g_fit < 1e-2
            else "DIFFERS")
    s = json.dumps(rep, indent=1, default=str)
    if a.out:
        with open(a.out, "w") as f:
            f.write(s)
    print(json.dumps({k: rep[k] for k in ("verdict", "split", "config_diff", "trainer_diff", "steps", "grads",
                                          "params") if k in rep}, indent=1, default=str))
    return 0 if all(not v.startswith("DIFF") for v in rep["verdict"].values()) else 1


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    rest: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, rest = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--arm", choices=("old", "new"), required=True)
    r.add_argument("--batch", default=None)
    r.add_argument("--record-only", action="store_true", help="real generation; just record (for making a batch)")
    r.add_argument("--init-from", default=None)
    r.add_argument("--steps", type=int, default=1, help="optimiser steps before stopping (LR is 0 at step 1 under "
                                                         "warmup, so use 2 to see an update)")
    r.add_argument("--pool", action="store_true", help="serve every recorded completion of each item, rewarded and "
                                                        "unrewarded interleaved")
    r.add_argument("--deterministic", action="store_true", help="deterministic CUDA kernels in this arm")
    r.add_argument("--save-grads", action="store_true", help="save each step's gradient as the optimiser gets it")
    r.add_argument("--loss-scale", type=float, default=1.0, help="multiply every micro-batch loss (diagnostic)")
    r.add_argument("--mem-report", action="store_true", help="memory around each phase of each step, to memory.jsonl")
    r.add_argument("--mem-snapshot", action="store_true", help="also dump allocator history (mem_snapshot.pickle)")
    r.add_argument("--cap-length", action="store_true", help="serve every completion at max_completion tokens "
                                                              "(memory probes; rewards are meaningless)")
    r.add_argument("--longest-prompts", type=int, default=0, metavar="N",
                   help="train on the N items with the longest prompts only (memory probes)")
    r.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("arms", nargs=2)
    c.add_argument("--out", default=None)
    c.add_argument("--split", action="store_true", help="the arms are one trainer at two micro-batch splits")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        if not a.record_only and not a.batch:
            ap.error("run needs --batch unless --record-only")
        cmd_run(a, rest)
        return 0
    return cmd_compare(a)


if __name__ == "__main__":
    sys.exit(main())
