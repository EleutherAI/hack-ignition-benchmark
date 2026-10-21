# Parity with the published runs

Does this library do what the research code that produced the published dataset did? Same-seed runs cannot be
compared completion by completion: bf16 vLLM generation is not deterministic across batch composition and kernels. So
parity is checked in four tiers, the cheapest and most exact first.

| tier | what | standard | status |
|---|---|---|---|
| 0. grading | regrade published completions with this library's environments; compare with the recorded grades | exact | **done, 2026-09-28** (below) |
| 1. training step | two optimiser steps of the research trainer and of this library on the same stored completions | bf16 tolerance | **passes, 2026-10-20**: mbpp and djinn bit-identical, backdoor at the noise floor (below) |
| 2. seeded plumbing | the research code's own functions and this library on identical inputs: injection plans and splices, reward schedule | exact | **passes** (`tests/test_core.py::test_parity_*`) |
| 3. full regeneration | 14 sampled configs x 2 seeds: label at the horizon within the published seed spread | distributional | not yet run (~51 GPU-hours) |

Also checked: every one of the 182 recipes reproduces the trainer arguments of its published runs
(`tools/check_recipes.py`: 459 runs, 429 exact, 30 explained, 0 unexplained). And two 48 GB-card knobs leave the
tier-1 training step bit-identical (below): `--bf16-base`, which stores frozen LoRA base weights in bf16, and
`--offload-activations`, which keeps the tensors saved for backward in CPU memory.

## Tier 1: training step (2026-10-20, library 7525203 against research 30a1793)

`tools/parity_tier1.py` runs each arm's own code end to end: its dataset rows, reward function, advantages,
GRPOConfig, LoRA wrapping and optimiser. Class-level patches on TRL's GRPOTrainer fix three things from outside. Both
arms get the same stored completions for the items their own samplers pick. vLLM is switched off after the config is
recorded, which cannot change the loss or update, because no published run used importance-sampling correction and
each generation is used for one step. Training stops after two optimiser steps; the cosine warmup gives step 1 a
learning rate of 0, so step 2 is the first real update. Completions are pooled across the published run, rewarded
and unrewarded interleaved, so every group has nonzero advantages. Kernels are deterministic where torch has a
deterministic implementation. Each new arm ran twice to measure the noise floor. Hardware: one A40 per arm (allium-0).

The research arm is git `30a1793`, which is byte-identical to the pod tree the monitor runs trained from, plus that
tree's `04_rl/monitors/` (uncommitted at the time; byte-identical to this library's `monitors/`). For backdoor,
`30a1793` differs from the published `c5e2671` only in run-record code.

| case (published run) | path | config | batch (items, token ids, advantages) | step-2 update, old vs new | new vs new |
|---|---|---|---|---|---|
| mbpp `monitor/mbpp_A_t1_s0` (OLMo-3-7B LoRA, monitor squash) | lora-cosine | identical | identical | **bit-identical** (rel. L2 0.0) | 0.0 |
| djinn `monitor/djinn_f100_t4_s0` (Qwen3-8B LoRA, monitor squash) | lora-cosine, djinn grader | identical | identical | **bit-identical** (rel. L2 0.0) | 0.0 |
| backdoor `rl_struct/sub_silver_s0` (Llama-3.2-1B full fine-tune) | prime-rl, separate research trainer | identical | identical | rel. L2 0.024 | 0.023 |

- The backdoor difference is the arm's own run-to-run nondeterminism, not the code: two runs of the new arm on
  identical inputs differ as much. Loss and rewards agree to every digit at both steps, and step-1 gradient norms
  agree to 4e-5 relative. So decision 4's fallback (a separate backdoor loop) is not needed.
- djinn ran at 4 x 4 micro-batches instead of the published 8 x 2 in both arms, because the fp32 Qwen3-8B policy does
  not fit a 48 GB card at 8 x 2. Each optimiser step still sees the same 16 completions, and both arms used the same
  split, so the comparison holds; a different split of the same completions is not equivalent (see "Micro-batch
  splits" below).
- **LoRA initialisation is not seeded by the run seed, in either trainer.** TRL wraps the model in LoRA
  (`grpo_trainer.py:447`) before `Trainer.__init__` calls `set_seed(args.seed)`, and torch's default generator gets a
  fresh random seed in every process. So LoRA A matrices differ between runs with the same seed. This holds for
  the published runs too, so no published run's initialisation can be reproduced, and the tier-1 new arms load the
  research arm's initial adapter (`--init-from`) to compare the update rule. Tier 3 is distributional, so it is
  unaffected. Seeding torch before the trainer is built would make future runs reproducible; that is a decision,
  not a parity fix.

## Micro-batch splits (2026-10-20, library branch a40-vllm-server)

The lora-cosine families train on 16 completions per optimiser step, split as micro-batches of 8 accumulated twice
(8 x 2). On a 48 GB card Qwen3-8B trains only at 4 x 4. **Does splitting the same 16 completions differently change
the update?**

1. **Yes: the raw gradient is proportional to 1 / grad_accum.** On mbpp the step-1 and step-2 gradients at 4 x 4 are
   0.500 of those at 8 x 2, and at 2 x 8 they are 0.250 (best scalar fit, to 4 digits). lora-cosine uses TRL 1.10's
   default loss, `dapo`, which divides by the completion tokens of the whole step and switches off the HF Trainer's
   accumulation scaling. accelerate 1.14's `backward` still divides every micro-batch loss by grad_accum.
2. **Adam passes the factor on to the update.** Adam ignores a constant gradient factor except through its eps
   (1e-8), and these gradients are small: at step 2 the median LoRA parameter's Adam denominator is 2.5e-6 (mbpp) or
   9e-7 (djinn), and 5 to 8 % of parameters are below 1e-7. On one split (4 x 4), doubling the loss doubles every
   gradient exactly and moves the step-2 update by 3.2 % (relative L2). 8 x 2 against 4 x 4 differs by the same
   3.2 %; with the 4 x 4 loss doubled to undo the factor, by 0.45 %.
3. **Beyond the factor, smaller micro-batches change the numerics.** After the best scalar fit the gradients differ by
   0.3 % between 8- and 4-row micro-batches (mbpp), 1 to 6 % between 8 and 2 rows (mbpp), and 5.5 to 19 % between 4 and
   2 rows (djinn). The update differences are 0.45 %, 10 % and 21 %.
4. **So equal-product splits stay variants.** `--batch 4x4` makes a variant id, as before. The djinn-graded Qwen3-8B
   families can run on 48 GB cards only as 2 x 8 variants: 8 x 2 and 4 x 4 ran out of memory (docs/ENVIRONMENT.md).
   The 2 x 8 gradient differs from 4 x 4 by 5.5 to 19 % after the factor is undone (table below). It was not
   compared with 8 x 2, which needed an 80 GB card until `--bf16-base` (addendum below).

Protocol: tier 1's new arm against itself, on tier 1's inputs (two optimiser steps, completions pooled across the
published run, the step-1 learning rate is 0), every arm initialised from the tier-1 research arm's LoRA weights, one
A40 per arm, `--save-grads` keeping the gradient the optimiser received at each step. `--loss-scale` multiplies every
micro-batch loss by a constant, a diagnostic for the factor in claim 1.

| case | A | B | gradient B/A (fit) | gradient diff after the fit, steps 1 / 2 | step-2 update, rel. L2 |
|---|---|---|---|---|---|
| mbpp (OLMo-3-7B), regression | this branch 8 x 2 | tier-1 new arm 8 x 2 | not saved | not saved | **0** (bit-identical) |
| djinn (Qwen3-8B), regression | this branch 4 x 4 | tier-1 new arm 4 x 4 | not saved | not saved | **0** (bit-identical) |
| mbpp, default kernels | 8 x 2 | 8 x 2, run twice without deterministic kernels | 1 | 0 / 0 | **0** |
| djinn, default kernels | 4 x 4 | 4 x 4, likewise | 1 | 0 / 0 | **0** |
| mbpp, loss scale only | 4 x 4 | 4 x 4, loss x 2 | 2 | 0 / 0 | **3.2 %** |
| mbpp | 8 x 2 | 4 x 4 | 0.500 | 0.29 % / 0.25 % | **3.2 %** |
| mbpp, factor undone | 8 x 2 | 4 x 4, loss x 2 | 1.000 | 0.29 % / 0.25 % | **0.45 %** |
| mbpp | 8 x 2 | 2 x 8 | 0.250 | 0.97 % / 6.3 % | **11.7 %** |
| mbpp, factor undone | 8 x 2 | 2 x 8, loss x 4 | 1.000 | 0.95 % / 6.3 % | **10.1 %** |
| djinn | 4 x 4 | 2 x 8 | 0.499 / 0.486 | 5.5 % / 19 % | **21 %** |
| djinn, factor undone | 4 x 4 | 2 x 8, loss x 2 | 1.00 / 0.97 | 5.5 % / 19 % | **21 %** |

- **Same-split noise is zero at these sizes.** Runs at 8 x 2 (mbpp) and 4 x 4 (djinn) are bit-identical with or
  without deterministic kernels. Two-row micro-batches are not: doubling the loss at 2 x 8 gives gradients that
  differ from exactly double by 0.6 % (mbpp) and 0.8 % (djinn). The 2-row rows of the table mix that with the
  shape effect.
- **Why the numerics move.** Each micro-batch's gradient is computed in bf16 under autocast and rounded before it
  is accumulated in fp32. GRPO advantages sum to zero within a group and groups are spread across micro-batches, so
  the per-micro-batch gradients largely cancel and their rounding errors are large next to the sum. This is our
  reading of the measurements; it is not separately tested.
- **What these steps do not show.** Two steps from a LoRA initialisation (B = 0, so all of the gradient is in the B
  matrices) cannot show how the difference grows over 250 steps. Gradient clipping (TRL's default max_grad_norm, 1.0)
  never bound here, where norms were 0.01 to 0.04. In the published djinn_v2 runs (all 8 x 2) the logged norm passed
  1.0 on 19 of 67,473 steps, in 4 of 128 runs. Because of the factor, a different split also moves the clipping
  threshold relative to the gradient. Qwen3-8B at 8 x 2 could not be measured on A40s in fp32 (it runs out of
  memory); the addendum below measures it with `--bf16-base`.
- **The factor can be undone, but that is a decision, not a fix.** Multiplying the loss by grad_accum / 2 brings
  mbpp 8 x 2 and 4 x 4 to 0.45 %, below tier 1's 1 % tolerance. The published runs had no such multiplier, and for
  djinn the 4- against 2-row residual stays at 21 %. So this branch does not add one.
- The prime-rl preset (`loss_type = "grpo"`) divides by grad_accum in TRL and again in accelerate, so the same factor
  is expected there. This comes from reading the code, not from a measurement.

Reports and logs: allium-0 `/mnt/ssd-cluster/david/hib/a40/split/` (`cmp_*.json`).

### Addendum: djinn at the published 8 x 2 (2026-10-21, branch a40-bf16-base)

With `--bf16-base`, which gives bit-identical gradients and updates (next section), the Qwen3-8B tier-1 arm fits a
48 GB card at 8 x 2 (training only; vLLM is off in tier 1). Same protocol, every arm with `--bf16-base`:

| case | A | B | gradient B/A (fit), steps 1 / 2 | gradient diff after the fit, steps 1 / 2 | step-2 update, rel. L2 |
|---|---|---|---|---|---|
| djinn | 8 x 2 | 4 x 4 | 0.500 / 0.534 | 0.29 % / 19 % | **21 %** |
| djinn, factor undone | 8 x 2 | 4 x 4, loss x 2 | 1.000 / 1.07 | 0.29 % / 19 % | **21 %** |
| djinn | 8 x 2 | 2 x 8 | 0.250 / 0.250 | 5.5 % / 0.75 % | **9.6 %** |

- **Neither A40 variant is shown to be closer to the published split.** At step 1, 4 x 4 is nearer 8 x 2 (0.29 % against
  5.5 %); at step 2, 2 x 8 is (0.75 % against 19 %), and so is the step-2 update (9.6 % against 21 %). Two steps on one
  batch do not rank them.
- Both arms saw the same step-2 completions: their rewards, token counts and completion lengths agree exactly. The
  4 x 4 step-2 gradient is reproducible: the fp32 and `--bf16-base` arms, and this branch and a40-vllm-server, give it
  bit for bit. TRL shuffles each generation batch before splitting it into micro-batches, so no split keeps a group
  of 8 completions together.

Reports: david-a40-0 `/mnt/ssd-cluster/david/hib/a40/bf16base/t1/dj_8x2_vs_*.json`.

## Frozen base weights in bf16: `--bf16-base` (2026-10-21, library branch a40-bf16-base)

The lora-cosine families train an fp32 policy (TRL's default dtype) under bf16 autocast, and fp32 is why the 7-8B
policies crowd a 48 GB card. `--bf16-base` stores the frozen base weights of a LoRA policy in bf16 instead and sends
vLLM the merged weights without writing them into the base. **Does it change what a run computes?**

1. **No: the training step is bit-identical.** On tier 1's inputs, fp32 and `--bf16-base` give the same gradients at
   steps 1 and 2 and the same step-2 update, to the bit (relative L2 difference 0), for mbpp on OLMo-3-7B at 8 x 2 and
   djinn on Qwen3-8B at 4 x 4. Both start models are stored in bf16, so their fp32 copies are exact upcasts, and
   autocast casts each nn.Linear weight back to bf16 before its matmul: the matmuls see the same operands either way.
2. **Only nn.Linear weights can be stored in bf16.** Embeddings stay fp32 because their output starts the residual
   stream, which would otherwise be added up in bf16. Norms stay fp32 because Qwen3's q and k norms multiply their weight
   into bf16 activations, and only an fp32 weight lifts that product to fp32. Converting either changes the logits
   (`tests/test_bf16_base.py`); converting `lm_head` or the residual-stream norms does not.
3. **vLLM's copy of the policy differs from the fp32 path's only by that path's own rounding.** TRL syncs a LoRA policy
   by adding the LoRA update into the base weights in place and subtracting it again after sending, which rounds in
   fp32. Over a replayed 250-step run on Qwen3-8B, that moves 87 to 269 million of the 6.9 billion weights in its LoRA
   layers. It changes the bf16 value autocast trains on in 125 to 1,315 of them. In the weights vLLM ends up with
   (bf16), the two paths differ in 4,694 to 15,792 elements, an L2 difference of 0.1 to 0.3 % of the bf16 rounding
   both paths apply anyway. `--bf16-base` never writes the base, so it is the fp32 path without that drift.
4. **So `--bf16-base` is recorded, not a variant.** It is the same kind of change as `--vllm-server-gpu`: it moves bf16
   generation numerics, not the configuration, and here in at most 2.3 per million of vLLM's weights.
5. **It saves 12.8 GiB of weights on OLMo-3-7B and 14.1 GiB on Qwen3-8B. Qwen3-8B now trains at 4 x 4 on a 48 GB card,
   but still not at 8 x 2.** In 20-step smokes, OLMo-3-7B at the published 8 x 2 peaked at 35.2 GiB on the trainer card,
   against 47.4 GiB in fp32. Qwen3-8B at 4 x 4 ran 20 steps at 37.4 GiB; in fp32 it ran out of memory at step 6. Both
   runs reached the 1,024-token completion cap. Qwen3-8B at 8 x 2 ran out of memory at step 6 instead of step 2, with
   completions up to 674 tokens. The allocation that failed, 2.96 GiB, is one fp32 copy of the logits for 8
   completions of about 650 tokens over Qwen3's 151,936-token vocabulary. From TRL's code, several such copies are
   alive through the loss and its backward pass, so the remaining memory is activations, which grow with micro-batch
   rows times completion length. 4 x 4 is still a variant, and on djinn it is not measured to be closer to 8 x 2 than the
   2 x 8 variant ("Micro-batch splits", addendum).

### Training step (claims 1 and 2)

Protocol: tier 1's new arm, as in "Micro-batch splits" above, run twice per case: once as published (fp32 policy) and
once with `--knobs '{"bf16_base": true}'`. Both arms start from the tier-1 research arm's LoRA weights (`--init-from`).
Each runs two optimiser steps on completions pooled across the published run, with deterministic kernels and
`--save-grads`, on one A40 per arm (david-a40-0).

| case | split | fp32 parameters | `--bf16-base` parameters | gradient, steps 1 / 2, rel. L2 | step-2 update, rel. L2 | loss, grad norm, entropy at both steps |
|---|---|---|---|---|---|---|
| mbpp `monitor/mbpp_A_t1_s0` (OLMo-3-7B) | 8 x 2 | 27.5 GiB | 14.7 GiB | **0 / 0** | **0** | identical |
| djinn `monitor/djinn_f100_t4_s0` (Qwen3-8B) | 4 x 4 | 30.8 GiB | 16.7 GiB | **0 / 0** | **0** | identical |

The fp32 arms of this branch are also bit-identical (gradients at both steps and the step-2 update, relative L2 0) to
the "Micro-batch splits" arms of branch a40-vllm-server at the same splits. So the branch leaves the default path
unchanged.

- The conversion runs after TRL has built the LoRA adapter. PEFT gives a new adapter its base layer's dtype, so
  converting first would round the LoRA initialisation to bf16.
- A weight that bf16 cannot hold exactly is refused rather than rounded, so a start model stored in fp32 cannot be
  silently changed.
- The unit tests run a tiny random Qwen3 (with and without tied embeddings) and OLMo-3 on CPU under bf16 autocast.
  Logits, LoRA gradients and the reference pass with adapters disabled are bit-identical there. Converting q_norm,
  k_norm or the embeddings to bf16 changes the logits; converting `input_layernorm`, `post_attention_layernorm`, the
  final norm or `lm_head` does not.

### Weight sync (claim 3)

`tools/merge_drift.py` replays 250 syncs on Qwen3-8B, one LoRA layer at a time, with PEFT's own merge and unmerge on the
fp32 weights. The adapter moves at every step on the lora-cosine schedule (5e-5 peak). A starts at PEFT's
initialisation and B at 0, and each element takes a step of the learning rate times a normal draw. `random` draws
afresh at every step; `directed` keeps one draw per element, like a parameter that Adam moves one way under a
consistent gradient, which gives the larger adapter.

| walk | step | LoRA update, L2 relative to W | fp32 base values changed | bf16 base values changed | vLLM weights that differ (bf16) | that difference, L2 relative to the bf16 rounding |
|---|---|---|---|---|---|---|
| random | 10 | 0.018 % | 32.5 M | 22 | 155 | 0.019 % |
| random | 250 | 0.084 % | 87.2 M | 125 | 4,694 (0.68 per million) | 0.12 % |
| directed | 10 | 0.048 % | 28.5 M | 36 | 150 | 0.025 % |
| directed | 50 | 0.40 % | 105 M | 382 | 2,558 | 0.15 % |
| directed | 250 | 1.4 % | 269 M | 1,315 | 15,792 (2.3 per million) | 0.34 % |

Of 6,945,767,424 weights in the 252 LoRA layers. "bf16 base values changed" are the weights autocast hands the training
matmuls. "vLLM weights" are bf16(W + dW) as each path sends them and vLLM stores them. The largest fp32 drift of any
weight was 9.5e-7. At the first sync the two paths send identical tensors
(`test_out_of_place_sync_sends_what_trl_sends_and_leaves_the_base_alone`).

### Memory (claim 5)

20-step smokes on two A40s with `--vllm-server-gpu 1`, peak memory from nvidia-smi once a second:

| run | split | fp32 policy (2026-10-20) | `--bf16-base` |
|---|---|---|---|
| `mbpp/bands/band5` (OLMo-3-7B) | 8 x 2, published | 20 steps, trainer card 47.4 GiB, 43.6 s/step | 20 steps, **35.2 GiB**, 42.5 s/step |
| `djinn_v2/base_f100` (Qwen3-8B) | 8 x 2, published | out of memory at step 2 (46.7 GiB) | out of memory at step 6 (46.2 GiB, completions up to 674 tokens) |
| `djinn_v2/base_f100` | 4 x 4, a variant | out of memory at step 6 (47.3 GiB, completions up to 831 tokens) | 20 steps, **37.4 GiB**, 61.1 s/step |
| `djinn_v2/base_f100` | 2 x 8, a variant | 20 steps, 42.7 GiB, 65.2 s/step | not run |

Every run that finished reached a 1,024-token completion. The server card held 40.5 GiB (OLMo) or 41.3 GiB (Qwen) in
every run. The fp32 smokes ran on branch a40-vllm-server the day before, on the same pod.

### Caveats

- Tier 1 runs two steps from a LoRA initialisation with B = 0, so the policy equals the reference model and the KL
  term is 0 at both steps. The reference pass is compared directly only in the unit test, at nonzero B.
- The drift replay uses synthetic adapter walks, not a trained adapter. Their LoRA updates at step 250 are 0.08 and 1.4
  % of the base weights' L2 norm.
- The smokes reached the 1,024-token cap, but in only some micro-batches, and a 250-step run meets longer prompts
  and more cap-length completions. So a run that fits for 20 steps can still run out of memory later. Qwen3-8B at
  4 x 4 had about 10 GiB of headroom on the trainer card.
- Only the lora-cosine preset was compared. The knob also accepts prime-rl recipes with `lora = true`, which this
  section has not compared.

Reports, logs and drift JSON: david-a40-0 `/mnt/ssd-cluster/david/hib/a40/bf16base/` (`t1/*_vs_*.json`,
`drift/*.json`, `smoke/logs/`).

## Saved activations in CPU memory: `--offload-activations` (2026-10-21, library branch a40-save-on-cpu)

With `--bf16-base`, Qwen3-8B at the published 8 x 2 still ran out of memory on a 48 GB card. `--offload-activations`
copies the tensors each micro-batch's forward pass saves for backward to pinned CPU memory, and copies each back when
backward needs it. **Does it change what a run computes, and does it make 8 x 2 fit?**

1. **It changes nothing: the training step is bit-identical.** On tier 1's djinn inputs at 8 x 2 with `--bf16-base`,
   the gradients at steps 1 and 2 and the step-2 update are the same with and without it, to the bit (relative L2
   difference 0), and so are loss, grad norm and entropy at both steps. A copy changes no value, and the copy back has
   the original's shape, strides and dtype.
2. **The 8 x 2 peak is saved activations, so moving them is enough.** Under gradient checkpointing, a micro-batch
   saves the input of every decoder layer: 36 copies of the residual stream, which is fp32 because the embeddings are,
   4.5 MiB per sequence token at 8 rows. It also saves the loss head's fp32 logits, 4.6 GiB at 1,024 completion tokens.
   With base_f100's longest prompt (3,882 tokens in tier 1's monitor copy) and a 1,024-token completion, the layer
   inputs alone are 22.8 GiB, and 8 x 2 runs out of memory in its first forward pass. With the knob the same
   micro-batches peak at 33.7 GiB allocated (39.7 GiB reserved) of the 47.5 GiB PyTorch gets.
3. **What is left grows mostly with the completion.** With the saved tensors in CPU memory, a step's peak is the
   reference pass before training (no gradients; about 3.5 fp32 copies of the logits) or the start of backward
   through the loss head (about 3 copies), on top of 17.4 GiB of weights, LoRA and Adam state. At the 1,024-token cap
   that is 33.7 GiB whether the sequence is 2,435 or 4,906 tokens long. At 5,989 tokens, the longest prompt of any
   djinn recipe (4,965 tokens, `djinn_v2/broad`) with a 1,024-token completion, recomputing a layer in backward needs
   more than the loss head does, and backward peaks at 35.6 GiB. The longest prompt and the cap are the worst
   micro-batch a recipe allows, so no step of a djinn-graded 8 x 2 run should need more than 35.6 GiB allocated.
4. **So Qwen3-8B trains at the published 8 x 2 on 48 GB cards, as a reproduction.** `djinn_v2/base_f100` ran 20 steps
   with `--vllm-server-gpu 1 --bf16-base --offload-activations`, the trainer card peaking at 40.7 GiB, at
   72.6 s/step. Like `--bf16-base`, the knob is recorded, not a variant.
5. **It costs time.** The copies are not overlapped with computation, and each copy to the host waits until it is
   done. Two tier-1 steps took 4 % longer at the recorded completion lengths, and 13 % and 19 % longer with every
   completion at 512 and 1,024 tokens.

### Training step (claim 1)

Protocol: as in "Frozen base weights in bf16" above, tier 1's new arm twice at the published 8 x 2, both with
`--bf16-base`: once as is and once with `--knobs '{"bf16_base": true, "offload_activations": true}'`. Both arms start
from the tier-1 research arm's LoRA weights (`--init-from`). Each runs two optimiser steps on completions pooled across
the published run, with deterministic kernels and `--save-grads`, on one A40 per arm (david-a40-0).

| case | split | gradient, steps 1 / 2, rel. L2 | step-2 update, rel. L2 | loss, grad norm, entropy at both steps | time for both steps |
|---|---|---|---|---|---|
| djinn `monitor/djinn_f100_t4_s0` (Qwen3-8B), with against without | 8 x 2 | **0 / 0** | **0** | identical | 175.4 s against 168.4 s |
| the arm without, against branch a40-bf16-base's 8 x 2 arm | 8 x 2 | **0 / 0** | **0** | identical | |

The second row shows that this branch leaves the path without the knob unchanged. The unit tests
(`tests/test_offload.py`) run a tiny random Qwen3 with LoRA and non-reentrant gradient checkpointing on CPU under bf16
autocast, through TRL's own `selective_log_softmax`. Logits and LoRA gradients are bit-identical with the copies, for
both the fp32 policy and `--bf16-base`.

### Where the memory goes (claims 2 and 3)

Protocol: the same tier-1 arm at 8 x 2 with `--bf16-base`, two steps, default kernels, with three probe options of
`tools/parity_tier1.py`. `--cap-length` serves every completion at the cap (rewards are then meaningless; memory is
not). `--longest-prompts 2` keeps only the two items with the longest prompts, in tokens, so every micro-batch is
padded to the longest. `--mem-report` records the allocator's peak in each phase, and counts what each
forward pass saved. Without the knob it counts the same tensors it would copy. GiB here are 2^30 bytes from torch's
allocator. nvidia-smi shows the allocator's reservation (higher than its peak; "reserved" below) plus about 1 GiB of
CUDA context.

| completions | sequence, tokens | without: saved by one forward pass | without: step peak | with: step peak | with: backward peak |
|---|---|---|---|---|---|
| 512, tier 1's items | 1,673 to 1,923 | 11.0 GiB | 33.4 GiB (backward) | 25.5 GiB (reference pass) | 25.0 GiB |
| 1,024, tier 1's items | 2,185 to 2,435 | 15.7 GiB | 42.7 GiB (backward) | 33.6 GiB (reference pass) | 32.0 GiB |
| 1,024, the 2 longest prompts (3,882 and 2,945 tokens) | 4,906 | out of memory in the first forward pass: 43.7 GiB allocated, 4.6 GiB more requested | | 33.7 GiB (reference pass) | 32.4 GiB |
| 1,024, `djinn_v2/broad`'s 2 longest prompts (4,965 and 3,877 tokens) | 5,989 | not run | | 35.6 GiB (backward; 39.1 GiB reserved) | 35.6 GiB |

- **The OOM, from an allocator snapshot taken there:** 21.6 GiB was the 36 layers' saved inputs, 16.4 GiB the weights
  (14.1 GiB of them the bf16 base), and 4.6 GiB one fp32 copy of the logits. The failed request was the next copy.
- **What is copied,** per micro-batch at the longest prompt: 22.8 GiB of layer inputs (38 tensors of 8 x 4,906 x
  4,096 fp32: one per layer, and two the final norm saves) and the logits twice, 9.3 GiB. `torch.gather` and each
  row's `logsumexp` in TRL's `selective_log_softmax` save the same storage, and each save is copied. That is 32.0 GiB
  in all (37.1 GiB at broad's longest prompt), to pinned memory (the pod has 628 GB).
- **What stays on the GPU:** parameters, including `lm_head`'s bf16 weight, which its matmul saves as a transposed view
  (1.2 GiB, but not a copy); tensors under 1 MiB; and nothing else at these sizes.
- **The broad row** used tier 1's arm with broad's recipe, stock Qwen3-8B weights in place of its derived start model
  (the same shapes), and a synthetic stored completion for each item (`--cap-length` repeats it).
- **Without the knob the peak grows with the prompt:** what one forward pass saved grew by 1.14 GiB from a 2,185- to a
  2,435-token sequence (4.5 MiB per token), and the backward peak with it. base_f100's prompts run from 744 to 3,881
  tokens (median 1,245). In its smoke with `--bf16-base` alone (previous section) the failed allocation was in
  backward at step 6, with completions of up to 674 tokens and the trainer card at 46.2 GiB.

### Smoke (claim 4)

20 steps of `djinn_v2/base_f100` on two A40s with `--vllm-server-gpu 1`, peak memory from nvidia-smi once a second,
as in the previous sections:

| split | `--bf16-base` (previous section) | `--bf16-base --offload-activations` |
|---|---|---|
| 8 x 2, published | out of memory at step 6 (46.2 GiB, completions up to 674 tokens) | 20 steps, **40.7 GiB**, 72.6 s/step |
| 4 x 4, a variant | 20 steps, 37.4 GiB, 61.1 s/step | not run |

The 8 x 2 run reached a 1,024-token completion. Its server card held 41.3 GiB, as before. Its run card has the same
variant id as the `--bf16-base` 8 x 2 smoke (`base_f100@8ef25b32`, from `--steps 20` alone), so the knob is recorded
but does not make a variant. The trainer card's 40.7 GiB (nvidia-smi) is the allocator's reservation plus CUDA's
context; the worst-case probes above reserved up to 39.7 GiB.

### Caveats

- Tier 1 runs two steps from a LoRA initialisation with B = 0, as in the previous sections. The knob wraps only TRL's
  `compute_loss`, the training forward pass: the reference and old-policy passes run without gradients and save nothing.
- Prompt lengths were counted with Qwen3's chat template, as TRL applies it. mbpp `qwen_*` prompts are far shorter
  (`mbpp/qwen_onpolicy/QO_k1`: at most 567 tokens), so those families were not probed.
- Only the lora-cosine preset was measured. Without gradient checkpointing (the prime-rl preset's default), every
  activation is saved outside a checkpoint, and the knob would copy all of them. It accepts such recipes, but that path
  has not been run.
- The time cost could be cut, by copying the shared logits storage once and by overlapping the copies with compute.
  Neither is done here.

Reports, memory reports and logs: david-a40-0 `/mnt/ssd-cluster/david/hib/a40/offload/` (`t1/*_vs_*.json`,
`probe/*/memory.jsonl`, `probe/long1024_keep/mem_snapshot.pickle`, `smoke/logs/`).

## Tier 0: grading (2026-09-28, library commit d32704a)

All 126 published runs that carry a rollout tier (Hub v0.1.6: mbpp 14, monitor 112) were regraded with
`tools/parity_tier0.py`, a random sample of 600 completions per mbpp-graded run and 300 per djinn-graded run (seed 0).
Model-written code ran as an unprivileged user in a disposable pod-local directory, with inputs fetched from the Hub
beforehand.

| environment | runs | completions | exact | grader-change escape (expected) | timeout-affected | real mismatches |
|---|---:|---:|---:|---:|---:|---:|
| mbpp | 62 | 37,200 | 37,193 | 7 | 0 | **0** |
| djinn | 64 | 19,200 | 19,133 | 0 | 63 | **4** |

- **mbpp: exact.** The 7 differences are all the builtin-subclass always-equal escape fixed in the grader on
  2026-09-28 (72caa21 in the research repo): recorded as honest solves, now caught as hacks, and the pre-fix check
  reproduces the recorded grade in every case. Every published run predates that fix.
- **djinn: 63 timeout-affected differences** (0.3 %) are completions where the grade depends on a timeout; djinn's
  per-test timeouts are load-dependent, so these are not grading differences.
- **djinn: 4 real mismatches** (0.02 %), all in the monitor family, on two problems: three different completions on
  `prefix_sum_validator_bypass_001_18` and one on `uniform_sequence_honor_bypass_036`. Each was recorded as
  (exploitable 0, hack 0, hardened 0) and now grades as an exploit (1, 1, 0): the insecure verifier reports success.
  **Cause confirmed (same day):** the monitor runs graded with the research repo's patched djinn tree
  (`djinn_eval_rlimit`); regraded under that tree, all 4 reproduce their recorded grades exactly. Under upstream djinn
  at `a0563f5` (the tree this library pinned at the time), 3 of the 4 differ again (the fourth matched on the rerun, so
  it is intermittent). So upstream `a0563f5` is **not** grade-equivalent to the tree the published djinn runs used,
  contrary to the fresh-environment smoke's earlier reading ("differs only by comments").
  **Root cause (2026-09-29): an upstream bug.** Twelve upstream insecure verifiers begin with
  `from __future__ import annotations`, and `exec()` inherits the caller's compiler flags, so submissions they grade
  get lazy annotations. These four completions annotate with `List` without importing it: under plain Python and the
  secure verifier they raise NameError, but upstream's insecure verifier loads them and its default `==` validator is
  fooled by their always-equal object. The published tree lacks the import, so it graded them correctly (a crash,
  reward 0). Fix: EleutherAI/djinn#10 compiles every submission with `dont_inherit=True`.
  **Rerun with the fix (2026-09-29):** all 64 djinn-graded runs, 19,200 completions: 19,137 exact, 63
  timeout-affected, **0 real mismatches**. Apart from the fix, the fixed tree differs from the published one only in
  comments, the dataset exporter, a stricter check on problem fields (every benchmark problem passes it) and code not
  on the grading path. #10 merged on 2026-09-30 as `b324baa`, which the library now pins.

Reports (one JSON per run) and logs are kept with the research repo's cluster outputs (`ele120_lib/tier0/`).
