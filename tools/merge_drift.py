"""How far does TRL's in-place LoRA merge drift an fp32 policy's frozen base weights, and how much does that change what
vLLM is sent, compared with --bf16-base's out-of-place merge?

TRL syncs a PEFT policy to vLLM by merging LoRA into the base weights (W += dW), sending W, and unmerging (W -= dW).
In fp32 the round trip rounds, so W moves a little at every sync; with --bf16-base, W is never written. This tool
replays K syncs on a real start model, one LoRA layer at a time with PEFT's own merge and unmerge, under an adapter
that moves at every step as a training run's does. At the --report steps it compares:

    base   the fp32 path's W_k with the start weights W_0: elements changed in fp32, and elements whose bf16 rounding
           changed (what autocast hands the training matmuls; --bf16-base trains on bf16(W_0) throughout)
    sync   what each path hands vLLM, as vLLM stores it (bf16): bf16(W_k + dW_k) from the fp32 path against
           bf16(W_0 + dW_k) from --bf16-base. For scale, the L2 size of the difference is reported next to the bf16
           rounding both paths apply anyway, ||bf16(W_0 + dW_k) - (W_0 + dW_k)||, and next to the adapter's ||dW_k||

The adapter walk: A starts at PEFT's initialisation and B at 0, as in training. Each step adds lr_k * u to every A and
B element, lr_k on the published lora-cosine schedule (5e-5 peak, 3 % warmup) and u ~ N(0, 1) drawn afresh each step
(`--walk random`) or fixed per element (`--walk directed`, a parameter that keeps moving one way, as Adam's steps do
under a consistent gradient: the larger adapter, so the larger rounding). Seeded per layer.

    python tools/merge_drift.py --model Qwen/Qwen3-8B --steps 250 --walk directed --out drift.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def lr_at(k: int, steps: int, peak: float, warmup_ratio: float) -> float:
    """transformers' cosine schedule with linear warmup, at optimiser step k (1-based: the LR of step k's update)."""
    w = math.ceil(steps * warmup_ratio)
    s = k - 1
    if s < w:
        return peak * s / max(1, w)
    p = (s - w) / max(1, steps - w)
    return peak * 0.5 * (1.0 + math.cos(math.pi * p))


def replay_layer(layer, steps: int, report: set[int], walk: str, peak: float, warmup_ratio: float, gen) -> dict:
    """Run `steps` merge/unmerge round trips on one PEFT LoRA layer whose base weight is fp32; return per-report-step
    sums. The layer's LoRA weights are walked in place."""
    import torch
    bl = layer.get_base_layer()
    w0 = bl.weight.data.clone()
    a, b = layer.lora_A["default"].weight, layer.lora_B["default"].weight
    da = db = None
    if walk == "directed":
        da = torch.randn(a.shape, generator=gen, device=a.device, dtype=a.dtype)
        db = torch.randn(b.shape, generator=gen, device=b.device, dtype=b.dtype)
    out = {}
    with torch.no_grad():
        for k in range(1, steps + 1):
            lr = lr_at(k, steps, peak, warmup_ratio)
            if walk == "random":
                a.add_(torch.randn(a.shape, generator=gen, device=a.device, dtype=a.dtype), alpha=lr)
                b.add_(torch.randn(b.shape, generator=gen, device=b.device, dtype=b.dtype), alpha=lr)
            else:
                a.add_(da, alpha=lr)
                b.add_(db, alpha=lr)
            # the sync after step k: the fp32 path merges in place and sends W; --bf16-base sends W_0 + dW
            layer.merge()
            sent_fp32 = bl.weight.data.to(torch.bfloat16)
            layer.unmerge()
            if k in report:
                dw = layer.get_delta_weight("default")
                exact = w0 + dw
                sent_oop = exact.to(torch.bfloat16)
                wk = bl.weight.data
                diff = (sent_fp32.float() - sent_oop.float())
                out[k] = {
                    "n": w0.numel(),
                    "base_fp32_changed": int((wk != w0).sum()),
                    "base_bf16_changed": int((wk.to(torch.bfloat16) != w0.to(torch.bfloat16)).sum()),
                    "base_max_abs_drift": float((wk - w0).abs().max()),
                    "base_drift_sq": float(((wk - w0).double() ** 2).sum()),
                    "w0_sq": float((w0.double() ** 2).sum()),
                    "sync_bf16_differs": int((sent_fp32 != sent_oop).sum()),
                    "sync_diff_sq": float((diff.double() ** 2).sum()),
                    "sync_rounding_sq": float(((sent_oop.float() - exact).double() ** 2).sum()),
                    "dw_sq": float((dw.double() ** 2).sum()),
                }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="id[@revision] or a local directory (a bf16 checkpoint)")
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--report", default="1,10,50,100,250", help="steps to report (those beyond --steps are dropped)")
    ap.add_argument("--walk", choices=("random", "directed"), default="directed")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-layers", type=int, default=None, help="only the first N LoRA layers (a quick look)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    import torch
    from peft import LoraConfig, get_peft_model
    from peft.tuners.lora import LoraLayer
    from transformers import AutoModelForCausalLM

    from hack_ignition_benchmark.train import resolve_model
    torch.manual_seed(a.seed)                           # PEFT's LoRA A initialisation
    path = resolve_model(a.model)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model = get_peft_model(model, LoraConfig(r=a.lora_r, lora_alpha=a.lora_r, lora_dropout=0.0,
                                             target_modules="all-linear", task_type="CAUSAL_LM"))
    layers = [(n, m) for n, m in model.named_modules() if isinstance(m, LoraLayer)][: a.max_layers]
    report = {k for k in (int(x) for x in a.report.split(",")) if k <= a.steps}
    tot: dict[int, dict] = {}
    t0 = time.time()
    for i, (name, m) in enumerate(layers):
        m.to(a.device)
        bl = m.get_base_layer()
        bl.weight.data = bl.weight.data.float()         # the fp32 policy TRL builds: an exact upcast
        for p in m.parameters():
            if p.requires_grad:
                p.data = p.data.float()
        gen = torch.Generator(device=a.device).manual_seed(a.seed * 100003 + i)
        res = replay_layer(m, a.steps, report, a.walk, a.lr, a.warmup_ratio, gen)
        for k, r in res.items():
            acc = tot.setdefault(k, {})
            for f, v in r.items():
                acc[f] = max(acc.get(f, 0.0), v) if f.startswith("base_max") else acc.get(f, 0) + v
        m.to("cpu")
        bl.weight.data = bl.weight.data.to(torch.bfloat16)
        if i % 25 == 0:
            print(f"[drift] {i + 1}/{len(layers)} {name} ({time.time() - t0:.0f}s)", flush=True)
    rows = {}
    for k in sorted(tot):
        r = tot[k]
        rows[k] = {"elements": r["n"], "base_fp32_changed": r["base_fp32_changed"],
                   "base_bf16_changed": r["base_bf16_changed"], "base_max_abs_drift": r["base_max_abs_drift"],
                   "base_drift_rel_l2": (r["base_drift_sq"] / r["w0_sq"]) ** 0.5,
                   "sync_bf16_differs": r["sync_bf16_differs"],
                   "sync_bf16_differs_frac": r["sync_bf16_differs"] / r["n"],
                   "sync_diff_over_rounding_l2": (r["sync_diff_sq"] / max(r["sync_rounding_sq"], 1e-300)) ** 0.5,
                   "sync_diff_over_dw_l2": (r["sync_diff_sq"] / max(r["dw_sq"], 1e-300)) ** 0.5,
                   "dw_over_w0_l2": (r["dw_sq"] / r["w0_sq"]) ** 0.5}
    rep = {"model": a.model, "path": path, "steps": a.steps, "walk": a.walk, "lr": a.lr, "lora_r": a.lora_r,
           "seed": a.seed, "layers": len(layers), "device": a.device,
           "versions": {m: __import__(m).__version__ for m in ("torch", "peft", "transformers")}, "by_step": rows}
    s = json.dumps(rep, indent=1)
    if a.out:
        with open(a.out, "w") as f:
            f.write(s)
    print(s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
