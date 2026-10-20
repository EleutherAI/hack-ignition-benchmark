"""--bf16-base: the knob, and (where torch, peft, transformers and trl are installed) the claim that it changes nothing
the trainer computes. The numerical tests run a tiny random Qwen3 on CPU under bf16 autocast; tier-1 parity on GPU
(docs/PARITY.md) is the full-size check."""
from __future__ import annotations

import copy
import types
from contextlib import nullcontext

import pytest

from hack_ignition_benchmark.recipe import RecipeError, parse_recipe
from hack_ignition_benchmark.runner import plan

BASIC = """
[run]
env = "mbpp"
model = "org/model@abc"
problems = "x/y.jsonl"
steps = 100
seeds = [0]
"""


def test_bf16_base_knob_is_recorded_not_a_variant(tmp_path, monkeypatch):
    from hack_ignition_benchmark import cli, runner
    r = parse_recipe(BASIC, "mbpp", "g/c")
    v = r.with_overrides(bf16_base=True)
    assert r.train()["bf16_base"] is False and v.train()["bf16_base"] is True
    assert not v.is_variant and v.variant_id is None and v.overrides == {"bf16_base": True}
    with pytest.raises(RecipeError):                    # a full fine-tune has no frozen base
        parse_recipe(BASIC + '\n[train]\npreset = "prime-rl"\n', "backdoor", "g/c").with_overrides(bf16_base=True)
    with pytest.raises(RecipeError):
        parse_recipe(BASIC + '\n[train]\nmodel_dtype = "bfloat16"\n', "mbpp", "g/c").with_overrides(bf16_base=True)
    path = tmp_path / "c.toml"
    path.write_text(BASIC)
    base = parse_recipe(BASIC, "mbpp", "g/c", source_path=str(path))
    jobs = plan([base], str(tmp_path), knobs={"bf16_base": True, "vllm_server_gpu": 1})
    assert jobs[0].variant is None and "--bf16-base" in jobs[0].cmd and "True" not in jobs[0].cmd
    seen = {}
    monkeypatch.setattr(runner, "execute", lambda r, *a, **k: seen.update(r.overrides) or 0)
    assert cli.main(jobs[0].cmd[1:]) == 0
    assert seen == {"bf16_base": True, "vllm_server_gpu": 1}


# --- numerics (CPU) ------------------------------------------------------------------------------------------------

def _tiny_lora_pair(tie=False, arch="qwen3"):
    """A tiny Qwen3 (or OLMo-3) with bf16-representable fp32 weights (as TRL builds the policy from a bf16 checkpoint),
    wrapped in LoRA with nonzero B; and a deep copy with --bf16-base applied."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    from peft import LoraConfig, get_peft_model

    from hack_ignition_benchmark.bf16_base import to_bf16_base
    torch.manual_seed(0)
    shape = dict(vocab_size=96, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                 num_key_value_heads=2, tie_word_embeddings=tie, max_position_embeddings=64)
    if arch == "qwen3":
        m = transformers.Qwen3ForCausalLM(transformers.Qwen3Config(head_dim=16, **shape)).float()
    else:
        m = transformers.Olmo3ForCausalLM(transformers.Olmo3Config(pad_token_id=0, **shape)).float()
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * 0.05)          # norms away from 1, so their weight matters
            p.copy_(p.bfloat16().float())
    m = get_peft_model(m, LoraConfig(r=4, lora_alpha=4, lora_dropout=0.0, target_modules="all-linear",
                                     task_type="CAUSAL_LM"))
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "lora_B" in n:
                p.copy_(torch.randn_like(p) * 0.05)
    b = copy.deepcopy(m)
    info = to_bf16_base(b)
    return m, b, info


def _step(model, ids, disable=False):
    import torch
    model.zero_grad(set_to_none=True)
    ctx = model.disable_adapter() if disable else nullcontext()
    with ctx, torch.autocast("cpu", dtype=torch.bfloat16):
        logits = model(input_ids=ids).logits
    logits = logits.float()
    loss = torch.log_softmax(logits[:, :-1], -1).gather(-1, ids[:, 1:, None]).mean()
    if not disable:
        loss.backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad and p.grad is not None}
    return logits.detach(), grads


def test_bf16_base_converts_only_frozen_linear_weights():
    torch = pytest.importorskip("torch")
    a, b, info = _tiny_lora_pair()
    da, db = dict(a.named_parameters()), dict(b.named_parameters())
    for n, p in db.items():
        want = torch.bfloat16 if (n.endswith("base_layer.weight") or n.endswith("lm_head.weight")) else torch.float32
        assert p.dtype == want, n
        assert torch.equal(p.float(), da[n]), n        # same values
    assert info["tensors"] == 2 * 7 + 1                 # 7 projections per layer, and the untied lm_head
    _, b2, info2 = _tiny_lora_pair(tie=True)
    assert info2["tensors"] == 2 * 7 and b2.get_output_embeddings().weight.dtype == torch.float32


def test_bf16_base_refuses_weights_bf16_would_round():
    torch = pytest.importorskip("torch")
    from hack_ignition_benchmark.bf16_base import to_bf16_base
    a, _, _ = _tiny_lora_pair()
    w = a.get_base_model().model.layers[0].mlp.down_proj.base_layer.weight
    with torch.no_grad():
        w[0, 0] += 1e-6                                 # no longer a bf16 value
    with pytest.raises(RuntimeError, match="not exactly representable"):
        to_bf16_base(a)


@pytest.mark.parametrize("arch,tie", [("qwen3", False), ("qwen3", True), ("olmo3", False)])
def test_bf16_base_training_step_is_bit_identical(arch, tie):
    """Policy forward (logits), LoRA gradients, and the reference pass with adapters disabled, under bf16 autocast."""
    torch = pytest.importorskip("torch")
    a, b, _ = _tiny_lora_pair(tie, arch)
    ids = torch.randint(0, 96, (3, 17), generator=torch.Generator().manual_seed(1))
    la, ga = _step(a, ids)
    lb, gb = _step(b, ids)
    assert torch.equal(la, lb)
    assert ga.keys() == gb.keys() and len(ga) > 0
    assert all(torch.equal(ga[n], gb[n]) for n in ga)
    ra, _ = _step(a, ids, disable=True)
    rb, _ = _step(b, ids, disable=True)
    assert torch.equal(ra, rb) and not torch.equal(ra, la)


@pytest.mark.parametrize("also", ["norm", "embed_tokens"])
def test_bf16_base_would_not_be_bit_identical_with_norms_or_embeddings_in_bf16(also):
    """Why only nn.Linear weights: storing the norms (Qwen3's q/k norms promote bf16 activations only through an fp32
    weight) or the embeddings (they start the fp32 residual stream) in bf16 changes the logits."""
    torch = pytest.importorskip("torch")
    a, b, _ = _tiny_lora_pair()
    with torch.no_grad():
        for n, p in b.named_parameters():
            if also in n and not p.requires_grad:
                p.data = p.data.bfloat16()
    ids = torch.randint(0, 96, (3, 17), generator=torch.Generator().manual_seed(1))
    assert not torch.equal(_step(a, ids)[0], _step(b, ids)[0])


def _vgen_stub(model, sent):
    """Enough of trl's VLLMGeneration for sync_weights in server mode, recording what would be sent."""
    pytest.importorskip("trl")
    from trl.generation.vllm_generation import VLLMGeneration
    v = object.__new__(VLLMGeneration)
    v.model, v.mode, v.enable_sleep_mode = model, "server", False
    v.accelerator = types.SimpleNamespace(is_main_process=True, state=types.SimpleNamespace(deepspeed_plugin=None))
    v._dist = types.SimpleNamespace(is_fsdp=False, gather_params=lambda params: nullcontext())
    v.vllm_client = types.SimpleNamespace(reset_prefix_cache=lambda: sent.append(("<reset>", None)))
    v._push_param_to_vllm = lambda name, t: sent.append((name, t.detach().clone()))
    return v


def test_out_of_place_sync_sends_what_trl_sends_and_leaves_the_base_alone():
    torch = pytest.importorskip("torch")
    from trl.generation.vllm_generation import VLLMGeneration

    from hack_ignition_benchmark.bf16_base import sync_weights_out_of_place
    a, b, _ = _tiny_lora_pair()
    with torch.no_grad():                               # a LoRA update large enough that fp32 merge/unmerge rounds
        for n, p in a.named_parameters():
            if "lora_" in n:
                p.mul_(30.0)
        b.load_state_dict({n: p for n, p in a.state_dict().items() if "lora_" in n}, strict=False)
    base_a = {n: p.detach().clone() for n, p in a.named_parameters() if not p.requires_grad}
    base_b = {n: p.detach().clone() for n, p in b.named_parameters() if not p.requires_grad}
    sa, sb = [], []
    VLLMGeneration.sync_weights(_vgen_stub(a, sa))      # TRL's own, on the fp32 policy
    sync_weights_out_of_place(_vgen_stub(b, sb))
    assert [n for n, _ in sa] == [n for n, _ in sb] and sa[-1][0] == "<reset>"
    assert not any("lora" in n or "base_layer" in n for n, _ in sa)
    for (n, ta), (_, tb) in zip(sa[:-1], sb[:-1]):
        assert ta.dtype == torch.float32 and (tb.dtype == torch.float32 or n == "lm_head.weight"), n
        assert torch.equal(ta, tb.float()), n           # the first sync: identical tensors
    # --bf16-base's base weights are untouched; TRL's fp32 round trip moved some of the fp32 path's
    assert all(torch.equal(p, base_b[n]) for n, p in b.named_parameters() if not p.requires_grad)
    moved = sum(int((p != base_a[n]).sum()) for n, p in a.named_parameters() if not p.requires_grad)
    assert moved > 0


def test_merge_drift_replay_counts_the_fp32_round_trip():
    torch = pytest.importorskip("torch")
    import importlib.util
    import os
    spec = importlib.util.spec_from_file_location(
        "merge_drift", os.path.join(os.path.dirname(os.path.dirname(__file__)), "tools", "merge_drift.py"))
    md = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(md)
    assert md.lr_at(1, 250, 5e-5, 0.03) == 0.0 and md.lr_at(9, 250, 5e-5, 0.03) == pytest.approx(5e-5)
    a, _, _ = _tiny_lora_pair()
    layer = a.get_base_model().model.layers[0].mlp.down_proj
    res = md.replay_layer(layer, 50, {1, 50}, "directed", 5e-3, 0.03, torch.Generator().manual_seed(0))
    assert res[1]["n"] == 64 * 128 and res[50]["base_fp32_changed"] >= res[1]["base_fp32_changed"]
    assert res[50]["base_fp32_changed"] > 0 and res[50]["sync_rounding_sq"] > 0
