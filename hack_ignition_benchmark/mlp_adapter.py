"""`mlp_adapter_neurons = N`: train GRAFT's MLP adapters instead of LoRA, as N new SwiGLU neurons in every MLP.

GRAFT (rl/gradient_routing.py, `DualMLPAdapter`; `adapter_type: mlp`, `mlp_config: m16`) puts two 16-neuron SwiGLU
blocks, retain and forget, beside every MLP of the frozen policy:

    out = down(silu(gate x) * up x) + down_r(silu(gate_r x) * up_r x) + down_f(silu(gate_f x) * up_f x)

with gate and up initialised kaiming_uniform(a=sqrt(5)), which is U(-1/sqrt(H), 1/sqrt(H)), and down zero. In the arms
without gradient routing (no intervention, reward penalty) both blocks train on every token, so the policy is exactly
one MLP with intermediate_size I + 32 whose first I gate and up rows and first I down columns are the base's and stay
frozen. Built that way it is a plain HF checkpoint: TRL trains it as a full model, its KL reference is the same
checkpoint at init (the base function, since the new down columns are zero), and vLLM serves it by path.

    prepare()   pad the start model by N neurons with GRAFT's initialisation, drawn in fp32 from a CPU generator seeded
                with the run seed, and save it to <run>/mlp_adapter_model/ in the policy's dtype (`model_dtype`: fp32
                by default; bf16 for GRAFT's precision, which rounds the new rows as GRAFT's bf16 adapters are), so the
                saved init is the trained one. The config keeps the start model's dtype, which vLLM serves. A resumed launch reuses it.
    install()   after the trainer is built, before its optimiser exists: requires_grad off outside the MLP weights, and
                a hook on each MLP weight that zeroes the gradient of its base slice once it has accumulated. The base
                slices then have zero gradients, zero Adam moments and, with weight decay 0, an update of exactly
                0 / (0 + eps) = 0. Gradient clipping sees only the new slices, as GRAFT's sees only adapter weights.

Costs: the optimiser holds the whole MLP matrices (3 x H x (I + N) per layer, 81M numbers for SmolLM2-135M at N = 32),
and a checkpoint is the full model, not an adapter.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil

PADDED_DIR = "mlp_adapter_model"
PROVENANCE = "MLP_ADAPTER.json"
INIT = "gate, up: kaiming_uniform(a=sqrt(5)); down: zeros"
_CARRIED = ("tokenizer", "special_tokens_map", "vocab", "merges", "added_tokens", "chat_template", "generation_config")


def gated_mlps(model) -> list:
    """The MLP of every decoder layer (`model.model.layers[i].mlp`), checked to be a bias-free gated MLP: gate_proj,
    up_proj, down_proj, as in Llama, Qwen, Mistral and OLMo."""
    import torch.nn as nn
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise NotImplementedError(f"mlp_adapter_neurons needs a Llama-family causal LM (model.layers[i].mlp), not "
                                  f"{type(model).__name__}")
    out = []
    for i, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        lins = [getattr(mlp, k, None) for k in ("gate_proj", "up_proj", "down_proj")]
        if not all(isinstance(m, nn.Linear) for m in lins):
            raise NotImplementedError(f"layer {i}: mlp_adapter_neurons needs a gated MLP (gate_proj, up_proj, down_proj)")
        if any(m.bias is not None for m in lins):
            raise NotImplementedError(f"layer {i}: the MLP has biases; GRAFT's adapters have none")
        out.append(mlp)
    return out


def pad_mlps(model, n: int, seed: int) -> None:
    """Add n neurons to every MLP of `model`, in place, initialised as GRAFT initialises its adapters. The draws come
    from one generator seeded with `seed`, layer by layer, the n gate rows before the n up rows (GRAFT draws each
    16-row block from torch's global generator, so the values differ but the distribution is the same: kaiming's
    fan-in is H for a 16-row and a 32-row block alike)."""
    import torch
    import torch.nn as nn
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError(f"mlp_adapter_neurons must be a positive int, got {n!r}")
    gen = torch.Generator().manual_seed(int(seed))
    i0 = model.config.intermediate_size
    with torch.no_grad():
        for mlp in gated_mlps(model):
            g, u, d = mlp.gate_proj, mlp.up_proj, mlp.down_proj
            if (g.out_features, u.out_features, d.in_features) != (i0, i0, i0):
                raise RuntimeError(f"an MLP is not intermediate_size {i0} wide")
            h, dt = g.in_features, g.weight.dtype
            new = []
            for _ in range(2):
                t = torch.empty(n, h, dtype=torch.float32)
                nn.init.kaiming_uniform_(t, a=math.sqrt(5), generator=gen)
                new.append(t.to(dtype=dt, device=g.weight.device))
            for lin, extra, dim in ((g, new[0], 0), (u, new[1], 0),
                                    (d, torch.zeros(h, n, dtype=dt, device=d.weight.device), 1)):
                lin.weight = nn.Parameter(torch.cat([lin.weight, extra], dim=dim), requires_grad=lin.weight.requires_grad)
            g.out_features = u.out_features = d.in_features = i0 + n
            if hasattr(mlp, "intermediate_size"):
                mlp.intermediate_size = i0 + n
    model.config.intermediate_size = i0 + n


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def prepare(base_path: str, n: int, seed: int, out: str, base_spec: str | None = None,
            dtype: str = "float32") -> tuple[str, dict]:
    """Write <out>/mlp_adapter_model/: the start model at `base_path` padded by n neurons per MLP (seeded with `seed`),
    its weights in `dtype` (the dtype the trainer loads the policy in), with the start model's tokenizer and generation
    files. Returns (path, provenance). An existing one is reused when it was built from the same start model, n, seed
    and dtype, and refused otherwise."""
    import torch
    from transformers import AutoModelForCausalLM
    dst = os.path.join(out, PADDED_DIR)
    want = {"base": base_spec or os.path.abspath(base_path), "neurons": int(n), "seed": int(seed), "init": INIT,
            "weights_dtype": dtype}
    prov = os.path.join(dst, PROVENANCE)
    if os.path.exists(prov):
        with open(prov) as f:
            have = json.load(f)
        if {k: have.get(k) for k in want} != want:
            raise RuntimeError(f"{dst} was built for {({k: have.get(k) for k in want})}, this launch wants {want}")
        return dst, have
    with open(os.path.join(base_path, "config.json")) as f:
        base_cfg = json.load(f)
    model = AutoModelForCausalLM.from_pretrained(base_path, dtype=torch.float32)
    i0 = model.config.intermediate_size
    pad_mlps(model, n, seed)
    model.to(getattr(torch, dtype))
    tmp = dst + ".partial"
    shutil.rmtree(tmp, ignore_errors=True)          # left by a launch that died while writing
    model.save_pretrained(tmp, safe_serialization=True)
    cfg_path = os.path.join(tmp, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    served = base_cfg.get("dtype") or base_cfg.get("torch_dtype")
    if served:                                      # vLLM's "auto" dtype reads this, whatever the weights' dtype
        cfg["dtype"] = served
        cfg.pop("torch_dtype", None)
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)
    for fn in sorted(os.listdir(base_path)):            # the tokenizer and generation files, as the start model has them
        src = os.path.join(base_path, fn)
        if os.path.isfile(src) and fn.startswith(_CARRIED):
            shutil.copy2(src, os.path.join(tmp, fn))
    weights = sorted(fn for fn in os.listdir(tmp) if fn.endswith(".safetensors"))
    cfg = model.config
    info = {**want, "base_path": os.path.abspath(base_path), "base_intermediate_size": i0,
            "intermediate_size": i0 + n, "layers": cfg.num_hidden_layers, "hidden_size": cfg.hidden_size,
            "served_dtype": served,
            "weights_sha256": {fn: _sha256(os.path.join(tmp, fn)) for fn in weights}}
    with open(os.path.join(tmp, PROVENANCE), "w") as f:
        json.dump(info, f, indent=1)
    shutil.rmtree(dst, ignore_errors=True)
    os.replace(tmp, dst)
    return dst, info


def _zero_base_grad(i0: int, rows: bool):
    def hook(p):
        if rows:
            p.grad[:i0].zero_()
        else:
            p.grad[:, :i0].zero_()
    return hook


def freeze_base(model, n: int, base_intermediate_size: int) -> dict:
    """Make only the n neurons padded onto every MLP trainable: requires_grad off everywhere except the MLP weights, and
    a post-accumulate hook on each of those that zeroes its base slice (gate and up rows [0, I), down columns [0, I))."""
    i0 = base_intermediate_size
    mlps = gated_mlps(model)
    widths = {model.config.intermediate_size} | {m.gate_proj.out_features for m in mlps}
    if widths != {i0 + n}:
        raise RuntimeError(f"the policy's MLPs are {sorted(widths)} wide, not {i0} + {n}: is it the padded model?")
    for p in model.parameters():
        p.requires_grad_(False)
    trainable = held = 0
    for m in mlps:
        for lin, rows in ((m.gate_proj, True), (m.up_proj, True), (m.down_proj, False)):
            w = lin.weight
            w.requires_grad_(True)
            w.register_post_accumulate_grad_hook(_zero_base_grad(i0, rows))
            held += w.numel()
            trainable += n * (w.shape[1] if rows else w.shape[0])
    return {"neurons": n, "layers": len(mlps), "base_intermediate_size": i0, "trainable": trainable,
            "optimizer_numel": held, "frozen": "gate/up rows [0, I), down columns [0, I), everything outside the MLPs"}


def install(trainer, n: int, base_intermediate_size: int) -> dict:
    """Apply the freeze to a built GRPOTrainer, before it trains (the optimiser is built from the parameters that
    require grad when training starts). Refuses an optimiser setting under which a zero gradient could still move a
    weight."""
    try:
        from peft import PeftModel
    except ImportError:
        PeftModel = ()
    if isinstance(trainer.model, PeftModel):
        raise RuntimeError("mlp_adapter_neurons trains the padded model itself, not a LoRA policy")
    if getattr(trainer, "optimizer", None) is not None:
        raise RuntimeError("mlp_adapter_neurons: the optimiser already exists, so the freeze would come too late")
    optim = getattr(trainer.args.optim, "value", trainer.args.optim)
    if trainer.args.weight_decay or not str(optim).startswith("adamw"):
        raise RuntimeError(f"mlp_adapter_neurons needs AdamW with weight decay 0 to leave the base exactly unchanged; "
                           f"got optim {optim!r}, weight_decay {trainer.args.weight_decay}")
    info = freeze_base(trainer.model, n, base_intermediate_size)
    info["optim"] = str(optim)
    return info
