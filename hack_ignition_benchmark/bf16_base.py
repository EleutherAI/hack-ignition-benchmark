"""--bf16-base: store a LoRA policy's frozen base weights in bf16 instead of fp32, without changing what is computed.

The published runs train in fp32 (TRL's default dtype when a recipe sets no `model_dtype`) under bf16 autocast. Both
start models are stored in bf16, so every fp32 base weight is an exact upcast of a bf16 value, and autocast casts each
nn.Linear weight back to bf16 before its matmul. Storing those weights in bf16 hands every matmul the same operands,
forward and backward, and saves 2 bytes per weight (14 GiB for Qwen3-8B). What is converted, and what is not:

    converted   the weight of every frozen nn.Linear: LoRA base layers, and lm_head when it is not tied to the
                embeddings. Each runs under autocast (the policy, the reference pass with adapters disabled, lm_head
                inside the model's forward), so its matmul sees bf16 either way.
    kept fp32   embeddings: their output starts the residual stream, which would otherwise be added up in bf16.
                Norms: Qwen3's q/k norms multiply their weight into bf16 activations, and only an fp32 weight
                promotes that product to fp32. LoRA weights: they are what trains.

A converted weight that is not exactly representable in bf16 (a start model stored in fp32, say) is refused rather
than rounded.

Weight sync. TRL sends the policy to vLLM by merging LoRA into the base weights in place, sending them, and
subtracting LoRA again. In bf16 that round trip would drop the low bits of W at every sync. Here the merged weight is
built out of place, W.float() + dW with dW from PEFT's own get_delta_weight, sent as fp32 exactly as TRL sends it, and
dropped; W is never written. In fp32 the round trip is not exact either: (W + dW) - dW rounds, so the fp32 path's base
weights drift by a few fp32 ulps per sync. tools/merge_drift.py measures that drift.
"""
from __future__ import annotations

import types


def frozen_linear_weights(model) -> list[tuple]:
    """(name, parameter) of each weight --bf16-base stores in bf16: every frozen nn.Linear weight that is not also an
    embedding matrix (a tied lm_head)."""
    import torch.nn as nn
    emb = {id(m.weight) for m in model.modules() if isinstance(m, nn.Embedding)}
    out, seen = [], set()
    for name, mod in model.named_modules():
        w = getattr(mod, "weight", None)
        if not isinstance(mod, nn.Linear) or w is None or w.requires_grad or id(w) in emb or id(w) in seen:
            continue
        seen.add(id(w))
        out.append((f"{name}.weight" if name else "weight", w))
    return out


def to_bf16_base(model) -> dict:
    """Convert the frozen nn.Linear weights to bf16 in place. Returns what was done, for the run card."""
    import torch
    targets = frozen_linear_weights(model)
    if not targets:
        raise RuntimeError("--bf16-base: the policy has no frozen nn.Linear weights (it needs a LoRA recipe)")
    lora = [p for p in model.parameters() if p.requires_grad]
    n_bytes = 0
    with torch.no_grad():
        for name, p in targets:
            if p.dtype == torch.bfloat16:
                continue
            if p.dtype != torch.float32:
                raise RuntimeError(f"--bf16-base: {name} is {p.dtype}, expected the fp32 policy TRL builds by default")
            b = p.data.to(torch.bfloat16)
            if not torch.equal(b.float(), p.data):
                raise RuntimeError(f"--bf16-base: {name} is not exactly representable in bf16, so storing it in bf16 "
                                   f"would change it; the start model must be stored in bf16")
            n_bytes += p.numel() * 2
            p.data = b
    if any(p.dtype != torch.float32 for p in lora):
        raise RuntimeError("--bf16-base: a trainable parameter is not fp32")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    dtypes: dict[str, int] = {}
    for p in model.parameters():
        dtypes[str(p.dtype)] = dtypes.get(str(p.dtype), 0) + p.numel()
    return {"tensors": len(targets), "gib_saved": round(n_bytes / 2 ** 30, 2), "param_numel_by_dtype": dtypes}


def merged_weight(layer, weight):
    """W + dW for one LoRA layer, as PEFT's merge computes it (dW = get_delta_weight per active adapter, added in fp32),
    in a new tensor: `weight` is not written."""
    w = weight.data.float()
    for a in layer.active_adapters:
        if a in layer.lora_A:
            w = w + layer.get_delta_weight(a)           # out of place: w may be `weight` itself when it is fp32
    return w


def _check_vanilla(layers) -> None:
    for m in layers:
        if m.merged:
            raise RuntimeError("--bf16-base: a LoRA layer is merged; the out-of-place sync expects none to be")
        if getattr(m, "lora_variant", None) or any((getattr(m, "lora_bias", {}) or {}).values()):
            raise NotImplementedError("--bf16-base: the out-of-place sync covers plain LoRA (no DoRA, no lora_bias)")


def sync_weights_out_of_place(self) -> None:
    """Replacement for trl VLLMGeneration.sync_weights (bound to the instance) for a single-process PEFT policy: the
    same names and tensors TRL sends, with each LoRA layer's merged weight built out of place."""
    import torch
    from peft.tuners.lora import LoraLayer
    from trl.generation.vllm_generation import empty_cache
    if self._dist.is_fsdp or getattr(self.accelerator.state, "deepspeed_plugin", None) is not None:
        raise NotImplementedError("--bf16-base: the out-of-place sync is single-process only")
    if self.mode == "colocate" and self.enable_sleep_mode:      # as TRL: map vLLM's weights before loading them
        empty_cache()
        self.llm.wake_up(tags=["weights"])
        self._llm_weights_sleeping = False
    model = self.model
    layers = {id(m.get_base_layer().weight): m for m in model.modules() if isinstance(m, LoraLayer)}
    _check_vanilla(layers.values())
    with torch.no_grad():
        for name, param in model.named_parameters():
            name = name.removeprefix("base_model.model.").replace(".base_layer", "")
            if model.prefix in name or "original_module" in name:
                continue
            name = self._fix_param_name_to_vllm(name, extra_prefixes=["modules_to_save.default."])
            layer = layers.get(id(param))
            self._push_param_to_vllm(name, merged_weight(layer, param) if layer is not None else param.data)
    if self.mode == "server" and self.accelerator.is_main_process:
        self.vllm_client.reset_prefix_cache()
    elif self.mode == "colocate":
        self.llm.reset_prefix_cache()


def install(trainer) -> dict:
    """Apply --bf16-base to a built GRPOTrainer: bf16 base weights, and the out-of-place sync if it generates with vLLM.
    Call after the trainer is built (LoRA must be initialised in fp32 first: PEFT gives new LoRA weights the base
    layer's dtype, and would round their initialisation to bf16) and before it trains."""
    from peft import PeftModel
    if not isinstance(trainer.model, PeftModel):
        raise RuntimeError("--bf16-base needs a LoRA policy")
    info = to_bf16_base(trainer.model)
    vg = getattr(trainer, "vllm_generation", None)
    if vg is not None:
        vg.sync_weights = types.MethodType(sync_weights_out_of_place, vg)
    info["sync"] = "out_of_place" if vg is not None else None
    return info
