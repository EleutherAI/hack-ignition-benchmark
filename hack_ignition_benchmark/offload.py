"""--offload-activations: keep the tensors a training step saves for its backward pass in CPU memory, not on the GPU,
without changing what is computed.

Under gradient checkpointing (every lora-cosine recipe) the forward pass saves, for backward, the input of every
checkpointed decoder layer (the residual stream, fp32 under bf16 autocast because the embeddings are fp32) and whatever
the loss head saves: the final norm's inputs, lm_head's input, and the fp32 logits TRL's selective_log_softmax keeps
for logsumexp's backward. Inside a checkpointed layer nothing is saved (the layer is recomputed in backward, under
checkpoint's own hooks, which take precedence over these). This wraps the trainer's compute_loss in saved-tensor hooks
that copy each such tensor to pinned CPU memory when it is saved and back to the GPU when backward needs it:

    offloaded   every saved CUDA tensor of at least MIN_BYTES that is contiguous and not a parameter
    kept        parameters (lm_head's weight is on the GPU anyway; a copy would put a second one there in backward),
                tensors under MIN_BYTES, and non-contiguous views (restoring a view's strides exactly needs its base)

A copy changes no value, and the copy back has the original's shape, strides and dtype, so backward runs the same
kernels on the same operands. What the hooks cannot move is memory that is not saved: the temporaries that forward
and backward allocate and free (in the loss head, full-vocabulary fp32 tensors), which is what a micro-batch's peak
can be made of instead. docs/ENVIRONMENT.md and docs/PARITY.md have the measurements.

`mode="count"` keeps every tensor where it is and only records what would have moved (tools/parity_tier1.py
--mem-report uses it for arms without the knob).
"""
from __future__ import annotations

import types

MIN_BYTES = 1 << 20             # 1 MiB: smaller saved tensors are not worth a synchronising copy


def _new_stats() -> dict:
    return {"calls": 0, "offloaded": {"n": 0, "bytes": 0}, "kept": {}, "by_shape": {}}


def saved_tensor_hooks(stats: dict, mode: str = "offload", min_bytes: int = MIN_BYTES, device_type: str = "cuda"):
    """The saved_tensors_hooks context. `stats` accumulates counts and bytes per decision, and per dtype and shape
    for the tensors offloaded (in count mode: that would be)."""
    import torch
    if mode not in ("offload", "count"):
        raise ValueError(f"offload mode must be offload or count, got {mode!r}")
    pin = device_type == "cuda" and torch.cuda.is_available()

    def keep(t, why):
        k = stats["kept"].setdefault(why, {"n": 0, "bytes": 0})
        k["n"] += 1
        k["bytes"] += t.numel() * t.element_size()
        return t

    def pack(t):
        nbytes = t.numel() * t.element_size()
        if t.device.type != device_type or t.layout != torch.strided:
            return keep(t, "other_device")
        if isinstance(t, torch.nn.Parameter):
            return keep(t, "parameter")
        if nbytes < min_bytes:
            return keep(t, "small")
        if not t.is_contiguous():
            return keep(t, "non_contiguous")
        stats["offloaded"]["n"] += 1
        stats["offloaded"]["bytes"] += nbytes
        k = stats["by_shape"].setdefault(f"{str(t.dtype).removeprefix('torch.')}{list(t.shape)}", {"n": 0, "bytes": 0})
        k["n"] += 1
        k["bytes"] += nbytes
        if mode == "count":
            return t
        cpu = torch.empty_strided(t.size(), t.stride(), dtype=t.dtype, device="cpu", pin_memory=pin)
        cpu.copy_(t)
        return (t.device, cpu)

    def unpack(p):
        if isinstance(p, torch.Tensor):
            return p
        device, cpu = p
        out = torch.empty_strided(cpu.size(), cpu.stride(), dtype=cpu.dtype, device=device)
        out.copy_(cpu, non_blocking=pin)
        return out

    return torch.autograd.graph.saved_tensors_hooks(pack, unpack)


def install(trainer, mode: str = "offload", min_bytes: int = MIN_BYTES, device_type: str = "cuda") -> dict:
    """Wrap this trainer's compute_loss (the forward pass of each micro-batch; backward unpacks what it saved) in the
    hooks. Bound to the instance, so it wraps whatever compute_loss the class has. Returns what was done, for the
    run card; the running counts are `trainer._hib_offload["stats"]`."""
    if getattr(trainer, "_hib_offload", None) is not None:
        raise RuntimeError("--offload-activations is already installed on this trainer")
    stats = _new_stats()
    inner = trainer.compute_loss

    def compute_loss(self, *a, **k):
        stats["calls"] += 1
        with saved_tensor_hooks(stats, mode, min_bytes, device_type):
            return inner(*a, **k)

    trainer.compute_loss = types.MethodType(compute_loss, trainer)
    info = {"mode": mode, "min_bytes": min_bytes, "pinned": device_type == "cuda", "scope": "compute_loss"}
    trainer._hib_offload = {**info, "stats": stats}
    return info

