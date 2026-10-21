"""--offload-activations: the knob, and (where torch, peft, transformers and trl are installed) the claim that it changes
nothing the trainer computes. The numerical tests run a tiny random Qwen3 with LoRA and non-reentrant gradient
checkpointing on CPU under bf16 autocast, with the hooks' device set to "cpu" so that CPU tensors are copied the way
CUDA ones are on a GPU; tier-1 parity on GPU (docs/PARITY.md) is the full-size check."""
from __future__ import annotations

from contextlib import nullcontext

import pytest

from hack_ignition_benchmark.recipe import parse_recipe
from hack_ignition_benchmark.runner import plan

BASIC = """
[run]
env = "mbpp"
model = "org/model@abc"
problems = "x/y.jsonl"
steps = 100
seeds = [0]
"""


def test_offload_knob_is_recorded_not_a_variant(tmp_path, monkeypatch):
    from hack_ignition_benchmark import cli, runner
    r = parse_recipe(BASIC, "mbpp", "g/c")
    v = r.with_overrides(offload_activations=True)
    assert r.train()["offload_activations"] is False and v.train()["offload_activations"] is True
    assert not v.is_variant and v.variant_id is None and v.overrides == {"offload_activations": True}
    path = tmp_path / "c.toml"
    path.write_text(BASIC)
    base = parse_recipe(BASIC, "mbpp", "g/c", source_path=str(path))
    knobs = {"offload_activations": True, "bf16_base": True, "vllm_server_gpu": 1}
    jobs = plan([base], str(tmp_path), knobs=knobs)
    assert jobs[0].variant is None and "--offload-activations" in jobs[0].cmd and "True" not in jobs[0].cmd
    seen = {}
    monkeypatch.setattr(runner, "execute", lambda r, *a, **k: seen.update(r.overrides) or 0)
    assert cli.main(jobs[0].cmd[1:]) == 0
    assert seen == knobs


def test_hooks_copy_large_contiguous_tensors_and_keep_the_rest():
    torch = pytest.importorskip("torch")
    from hack_ignition_benchmark.offload import _new_stats, saved_tensor_hooks
    x = torch.randn(64, 64, requires_grad=True)
    w = torch.nn.Parameter(torch.randn(64, 64))
    odd = torch.randn(64 * 64, requires_grad=True)

    def forward():
        a = x * 2
        y = a.sin()                                     # saves a: copied
        z = a.t().cos()                                 # saves a transposed view: kept
        u = x @ w                                       # saves x (copied) and the parameter w (kept)
        s = x[:2].exp()                                 # saves its 512-byte result: kept
        t = odd.as_strided((64, 1, 64), (64, 7, 1))     # contiguous, with an unusual stride on the size-1 dim
        v = t.sin()
        return a, t, y, z, u, s, v

    st = _new_stats()
    with saved_tensor_hooks(st, "offload", min_bytes=1024, device_type="cpu"):
        a, t, y, z, u, s, v = forward()
    saved = y.grad_fn._saved_self
    assert saved.data_ptr() != a.data_ptr() and torch.equal(saved, a) and saved.stride() == a.stride()
    assert z.grad_fn._saved_self.data_ptr() == a.data_ptr()
    assert u.grad_fn._saved_mat2.data_ptr() == w.data_ptr() and u.grad_fn._saved_self.data_ptr() != x.data_ptr()
    assert s.grad_fn._saved_result.data_ptr() == s.data_ptr()
    assert v.grad_fn._saved_self.stride() == (64, 7, 1) and torch.equal(v.grad_fn._saved_self, t)
    assert st["offloaded"]["n"] == 3
    assert {k: v["n"] for k, v in st["kept"].items()} == {"non_contiguous": 1, "parameter": 1, "small": 1}
    sum(r.sum() for r in (y, z, u, s, v)).backward()
    g = (x.grad.clone(), w.grad.clone(), odd.grad.clone())
    x.grad = w.grad = odd.grad = None
    sum(r.sum() for r in forward()[2:]).backward()      # the same graph without the hooks
    assert all(torch.equal(p, q) for p, q in zip(g, (x.grad, w.grad, odd.grad)))


def test_count_mode_moves_nothing():
    torch = pytest.importorskip("torch")
    from hack_ignition_benchmark.offload import _new_stats, saved_tensor_hooks
    x = torch.randn(64, 64, requires_grad=True)
    st = _new_stats()
    with saved_tensor_hooks(st, "count", min_bytes=1024, device_type="cpu"):
        a = x * 2
        y = a.sin()
    assert y.grad_fn._saved_self.data_ptr() == a.data_ptr()
    assert st["offloaded"] == {"n": 1, "bytes": 64 * 64 * 4} and st["by_shape"] == {"float32[64, 64]": st["offloaded"]}


def test_install_wraps_this_trainers_compute_loss_once():
    pytest.importorskip("torch")
    from hack_ignition_benchmark.offload import install

    class T:
        def compute_loss(self, model, inputs, **kw):
            return (model, inputs, kw)

    t = T()
    info = install(t)
    assert info == {"mode": "offload", "min_bytes": 1 << 20, "pinned": True, "scope": "compute_loss"}
    assert t.compute_loss("m", "i", num_items_in_batch=3) == ("m", "i", {"num_items_in_batch": 3})
    assert t._hib_offload["stats"]["calls"] == 1 and "compute_loss" in vars(t) and "compute_loss" not in vars(T())
    with pytest.raises(RuntimeError, match="already installed"):
        install(t)


# --- numerics (CPU) ------------------------------------------------------------------------------------------------

def _checkpointed_pair():
    from test_bf16_base import _tiny_lora_pair
    a, b, _ = _tiny_lora_pair()                         # the fp32 policy, and the same with --bf16-base
    for m in (a, b):
        m.get_base_model().gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        m.train()
    return a, b


def _step(model, ids, adv, hooks=None):
    """One GRPO-shaped step as TRL takes it: the forward under bf16 autocast (and the hooks), fp32 logits, TRL's own
    selective_log_softmax, and backward outside both."""
    import torch
    from trl.trainer.utils import selective_log_softmax
    model.zero_grad(set_to_none=True)
    with hooks or nullcontext():
        with torch.autocast("cpu", dtype=torch.bfloat16):
            logits = model(input_ids=ids).logits
        logits = logits.float()[:, :-1] / 1.0
        logps = selective_log_softmax(logits, ids[:, 1:])
        loss = -(logps * adv[:, None]).mean()
    loss.backward()
    return logits.detach(), {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}


@pytest.mark.parametrize("which", ["fp32", "bf16_base"])
def test_offload_training_step_is_bit_identical(which):
    torch = pytest.importorskip("torch")
    pytest.importorskip("trl")
    from hack_ignition_benchmark.offload import _new_stats, saved_tensor_hooks
    a, b = _checkpointed_pair()
    m = a if which == "fp32" else b
    ids = torch.randint(0, 96, (3, 17), generator=torch.Generator().manual_seed(1))
    adv = torch.tensor([1.0, -0.5, -0.5])
    l0, g0 = _step(m, ids, adv)
    st = _new_stats()
    l1, g1 = _step(m, ids, adv, saved_tensor_hooks(st, "offload", min_bytes=0, device_type="cpu"))
    assert torch.equal(l0, l1)
    assert g0.keys() == g1.keys() and len(g0) > 0 and all(torch.equal(g0[n], g1[n]) for n in g0)
    # what moved: both checkpointed layers' inputs (the fp32 residual stream) and the fp32 logits rows logsumexp keeps
    assert st["by_shape"]["float32[3, 17, 64]"]["n"] >= 2 and st["by_shape"]["float32[16, 96]"]["n"] == 3
    if which == "bf16_base":                            # lm_head's bf16 weight is saved as itself, and stays put
        assert st["kept"]["parameter"]["n"] >= 1


def test_offload_would_move_nothing_inside_the_checkpointed_layers():
    """With gradient checkpointing, the saved tensors are only the layer inputs and the loss head's; without it,
    everything each layer saves would be copied too."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("trl")
    from hack_ignition_benchmark.offload import _new_stats, saved_tensor_hooks
    a, _ = _checkpointed_pair()
    ids = torch.randint(0, 96, (3, 17), generator=torch.Generator().manual_seed(1))
    adv = torch.tensor([1.0, -0.5, -0.5])
    ck = _new_stats()
    _step(a, ids, adv, saved_tensor_hooks(ck, "count", min_bytes=0, device_type="cpu"))
    a.get_base_model().gradient_checkpointing_disable()
    full = _new_stats()
    _step(a, ids, adv, saved_tensor_hooks(full, "count", min_bytes=0, device_type="cpu"))
    assert full["offloaded"]["n"] > 5 * ck["offloaded"]["n"]
