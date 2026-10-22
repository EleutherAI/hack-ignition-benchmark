"""mlp_adapter_neurons: the preset key, and (where torch, transformers and trl are installed) the claims that the padded
model is the base function at init, that AdamW moves only the new neurons, and that the trainable count is GRAFT's.
The numerical tests run a tiny random Llama on CPU; a GPU run of a graft recipe is the full-size check."""
from __future__ import annotations

import json
import math
import os
import types

import pytest

from hack_ignition_benchmark.recipe import RecipeError, iter_recipes, parse_recipe

RECIPES = os.path.join(os.path.dirname(os.path.dirname(__file__)), "recipes")

TOY = """
[run]
env = "sort"
model = "HuggingFaceTB/SmolLM2-135M-Instruct@12fd25f77366fa6b3b4b768ec3050bf629380bac"
prompt = "none"
steps = 10

[train]
preset = "graft-toy"
"""


def test_mlp_adapter_key_and_what_it_excludes():
    r = parse_recipe(TOY + "lora = false\nmlp_adapter_neurons = 32\n", "graft", "sort/x")
    assert r.train()["mlp_adapter_neurons"] == 32 and r.train()["lora"] is False
    assert r.to_dict()["train"]["mlp_adapter_neurons"] == 32           # the run card's resolved recipe carries it
    assert parse_recipe(TOY, "graft", "sort/x").train()["mlp_adapter_neurons"] is None   # unset: LoRA
    for preset in ("lora-cosine", "prime-rl"):
        assert parse_recipe(TOY.replace("graft-toy", preset), "g", "c").train()["mlp_adapter_neurons"] is None
    with pytest.raises(RecipeError, match="lora = false"):
        parse_recipe(TOY + "mlp_adapter_neurons = 32\n", "graft", "sort/x")
    for bad in ("0", "true", "1.5", '"32"'):
        with pytest.raises(RecipeError, match="new neurons"):
            parse_recipe(TOY + f"lora = false\nmlp_adapter_neurons = {bad}\n", "graft", "sort/x")
    with pytest.raises(RecipeError, match="weight_decay"):
        parse_recipe(TOY + "lora = false\nmlp_adapter_neurons = 32\nweight_decay = 0.01\n", "graft", "sort/x")
    with pytest.raises(RecipeError, match="derive"):
        parse_recipe(TOY + "lora = false\nmlp_adapter_neurons = 32\n\n[derive]\nmerge_checkpoint = 10\n", "g", "c")
    with pytest.raises(RecipeError, match="bf16_base"):
        r.with_overrides(bf16_base=True)
    assert not r.with_overrides(offload_activations=True).is_variant


def test_graft_recipes_train_lora_and_can_train_graft_adapters():
    """The recipes train LoRA in bf16 mixed precision; each takes GRAFT's adapters and precision as an override."""
    recipes = list(iter_recipes("graft", recipes_root=RECIPES))
    assert len(recipes) == 21
    grafts = "lora = false\nmlp_adapter_neurons = 32\nmodel_dtype = \"bfloat16\"\nbf16 = false\n"
    for r in recipes:
        t = r.train()
        assert t["lora"] is True and t["mlp_adapter_neurons"] is None, r.config
        assert (t["model_dtype"], t["bf16"]) == (None, True), r.config      # fp32 weights, bf16 autocast
        text = open(r.source_path).read().replace('preset = "graft-toy"\n', 'preset = "graft-toy"\n' + grafts)
        g = parse_recipe(text, "graft", r.config).train()
        assert (g["mlp_adapter_neurons"], g["lora"], g["model_dtype"], g["bf16"]) == (32, False, "bfloat16", False)


def test_lora_on_a_half_precision_policy_is_refused():
    with pytest.raises(RecipeError, match="in place"):
        parse_recipe(TOY + 'model_dtype = "bfloat16"\n', "graft", "sort/x")
    assert parse_recipe(TOY + 'lora = false\nmlp_adapter_neurons = 32\nmodel_dtype = "bfloat16"\n', "graft", "sort/x")


# --- numerics (CPU) ------------------------------------------------------------------------------------------------

def _tiny_base(tmp_path, **kw):
    """A tiny random Llama saved in bf16 with a bf16 config, as SmolLM2-135M-Instruct is stored."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    cfg = dict(vocab_size=96, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
               num_key_value_heads=2, max_position_embeddings=64, tie_word_embeddings=True)
    cfg.update(kw)
    m = transformers.LlamaForCausalLM(transformers.LlamaConfig(**cfg))
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * 0.05)          # norms away from 1, MLP outputs well away from 0
    d = tmp_path / "base"
    m.to(torch.bfloat16).save_pretrained(d)
    (d / "tokenizer_config.json").write_text("{}")      # stands in for the tokenizer files the padded dir must carry
    return str(d)


def _load(path, dtype="float32"):
    import torch
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(path, dtype=getattr(torch, dtype))


def test_padded_model_is_the_base_function_at_init(tmp_path):
    torch = pytest.importorskip("torch")
    from hack_ignition_benchmark.mlp_adapter import PROVENANCE, prepare
    base = _tiny_base(tmp_path)
    path, info = prepare(base, 8, 1, str(tmp_path / "run"), base_spec="tiny@abc")
    cfg = json.load(open(os.path.join(path, "config.json")))
    assert cfg["intermediate_size"] == 136 and cfg["dtype"] == "bfloat16"     # vLLM serves bf16, as for LoRA
    assert os.path.exists(os.path.join(path, "tokenizer_config.json"))
    assert (info["base"], info["neurons"], info["seed"], info["base_intermediate_size"]) == ("tiny@abc", 8, 1, 128)
    a, b = _load(base), _load(path)
    bound = 1 / math.sqrt(64)                           # kaiming_uniform(a=sqrt(5)) on fan-in 64
    for la, lb in zip(a.model.layers, b.model.layers):
        g, u, d = lb.mlp.gate_proj.weight, lb.mlp.up_proj.weight, lb.mlp.down_proj.weight
        assert torch.equal(g[:128], la.mlp.gate_proj.weight) and torch.equal(u[:128], la.mlp.up_proj.weight)
        assert torch.equal(d[:, :128], la.mlp.down_proj.weight) and torch.equal(d[:, 128:], torch.zeros(64, 8))
        for new in (g[128:], u[128:]):
            assert new.dtype == torch.float32 and new.abs().max() <= bound and new.abs().max() > bound / 2
            assert not torch.equal(new, new.bfloat16().float())       # drawn in fp32, not rounded to bf16
    assert all(torch.equal(p, b.state_dict()[n]) for n, p in a.state_dict().items() if ".mlp." not in n)
    ids = torch.randint(0, 96, (3, 17), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        la, lb = a(ids).logits, b(ids).logits
    torch.testing.assert_close(lb, la)                 # fp32 tolerance: only the matmul's reduction length changed
    # the same start model, width and seed give the same weights; another seed does not; a launch reuses its dir
    again, _ = prepare(base, 8, 1, str(tmp_path / "run2"), base_spec="tiny@abc")
    other, _ = prepare(base, 8, 2, str(tmp_path / "run3"), base_spec="tiny@abc")
    sha = lambda p: json.load(open(os.path.join(p, PROVENANCE)))["weights_sha256"]
    assert sha(again) == sha(path) and sha(other) != sha(path)
    assert prepare(base, 8, 1, str(tmp_path / "run"), base_spec="tiny@abc") == (path, info)
    with pytest.raises(RuntimeError, match="was built for"):
        prepare(base, 8, 2, str(tmp_path / "run"), base_spec="tiny@abc")


def _trainer(model, out, **args):
    pytest.importorskip("trl")
    from transformers import Trainer, TrainingArguments
    # lr 1e-2, not GRAFT's 3e-4: every new bf16 weight should visibly move in 3 steps; the freeze does not depend on it
    kw = dict(output_dir=out, learning_rate=1e-2, weight_decay=0.0, max_grad_norm=1.0, use_cpu=True, report_to=[])
    kw.update(args)
    return Trainer(model=model, args=TrainingArguments(**kw))


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_optimizer_step_moves_only_the_new_neurons(tmp_path, dtype):
    torch = pytest.importorskip("torch")
    from hack_ignition_benchmark.mlp_adapter import install, prepare
    path, info = prepare(_tiny_base(tmp_path), 8, 1, str(tmp_path / "run"), dtype=dtype)
    assert info["weights_dtype"] == dtype
    model = _load(path, dtype)
    assert all(p.dtype == getattr(torch, dtype) for p in model.parameters())
    tr = _trainer(model, str(tmp_path / "out"))
    info = install(tr, 8, 128)
    assert info["trainable"] == 2 * 3 * 64 * 8 and info["optim"] == "adamw_torch_fused"
    opt = tr.create_optimizer()                         # what the HF Trainer builds when training starts
    mlp = {id(p) for n, p in model.named_parameters() if ".mlp." in n}
    held = [p for g in opt.param_groups for p in g["params"]]
    assert {id(p) for p in held} == mlp and len(held) == 2 * 3
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    ids = torch.randint(0, 96, (3, 17), generator=torch.Generator().manual_seed(1))
    for _ in range(3):                                  # step 1 moves only down (the new down columns start at 0)
        model(ids, labels=ids).loss.backward()
        for n, p in model.named_parameters():
            if p.grad is not None:
                base = p.grad[:128] if ("gate" in n or "up_proj" in n) else p.grad[:, :128]
                assert torch.equal(base, torch.zeros_like(base)), n
        torch.nn.utils.clip_grad_norm_(held, 1.0)
        opt.step()
        opt.zero_grad()
    for n, p in model.named_parameters():
        if ".mlp." not in n:
            assert torch.equal(p, before[n]), n
            continue
        rows = "down_proj" not in n
        old, new = (p[:128], p[128:]) if rows else (p[:, :128], p[:, 128:])
        assert torch.equal(old, before[n][:128] if rows else before[n][:, :128]), n     # the base: bit-unchanged
        moved = new != (before[n][128:] if rows else before[n][:, 128:])
        assert moved.float().mean() > 0.99, n                                           # the new neurons: trained
        st = opt.state[p]
        for k in ("exp_avg", "exp_avg_sq"):
            sl = st[k][:128] if rows else st[k][:, :128]
            assert torch.equal(sl, torch.zeros_like(sl)), (n, k)


def test_install_refuses_what_would_move_the_base(tmp_path):
    pytest.importorskip("torch")
    from hack_ignition_benchmark.mlp_adapter import install, prepare
    path, _ = prepare(_tiny_base(tmp_path), 8, 1, str(tmp_path / "run"))
    with pytest.raises(RuntimeError, match="weight decay 0"):
        install(_trainer(_load(path), str(tmp_path / "o1"), weight_decay=0.01), 8, 128)
    with pytest.raises(RuntimeError, match="weight decay 0"):
        install(_trainer(_load(path), str(tmp_path / "o2"), optim="sgd"), 8, 128)
    tr = _trainer(_load(path), str(tmp_path / "o3"))
    tr.create_optimizer()
    with pytest.raises(RuntimeError, match="too late"):
        install(tr, 8, 128)
    with pytest.raises(RuntimeError, match="padded model"):              # the unpadded start model
        install(_trainer(_load(str(tmp_path / "base")), str(tmp_path / "o4")), 8, 128)


def test_pad_refuses_mlps_grafts_adapters_do_not_fit():
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from hack_ignition_benchmark.mlp_adapter import freeze_base, pad_mlps
    m = transformers.LlamaForCausalLM(transformers.LlamaConfig(
        vocab_size=96, hidden_size=64, intermediate_size=128, num_hidden_layers=1, num_attention_heads=4,
        num_key_value_heads=2, mlp_bias=True))
    with pytest.raises(NotImplementedError, match="biases"):
        pad_mlps(m, 8, 1)
    with pytest.raises(NotImplementedError, match="Llama-family"):
        freeze_base(types.SimpleNamespace(model=None, config=m.config), 8, 128)
    with pytest.raises(ValueError, match="positive int"):
        pad_mlps(m, 0, 1)


def test_trainable_count_is_grafts_for_smollm2():
    """SmolLM2-135M-Instruct's shape (hidden 576, intermediate 1536, 30 layers), padded by 32, on the meta device: the
    trainable count is GRAFT's m16 DualMLPAdapter's, 2 adapters x (gate + up + down) x 16 x 576 = 3 x 576 x 32 per
    layer, 1.66M in all."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from hack_ignition_benchmark.mlp_adapter import freeze_base
    cfg = transformers.LlamaConfig(vocab_size=49152, hidden_size=576, intermediate_size=1536 + 32, num_hidden_layers=30,
                                   num_attention_heads=9, num_key_value_heads=3, tie_word_embeddings=True)
    with torch.device("meta"):
        m = transformers.LlamaForCausalLM(cfg)
    info = freeze_base(m, 32, 1536)
    assert info["trainable"] == 30 * 3 * 576 * 32 == 30 * 2 * (16 * 576 + 16 * 576 + 576 * 16) == 1_658_880
    assert info["optimizer_numel"] == 30 * 3 * 576 * 1568 and info["base_intermediate_size"] == 1536
    assert [n for n, p in m.named_parameters() if p.requires_grad] == \
        [f"model.layers.{i}.mlp.{k}.weight" for i in range(30) for k in ("gate_proj", "up_proj", "down_proj")]
