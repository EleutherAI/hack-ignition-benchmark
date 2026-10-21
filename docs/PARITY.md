# Parity with the published runs

Does this library do what the research code that produced the published dataset did? Same-seed runs cannot be
compared completion by completion: bf16 vLLM generation is not deterministic across batch composition and kernels. So
parity is checked in four tiers, the cheapest and most exact first.

| tier | what | standard | status |
|---|---|---|---|
| 0. grading | regrade published completions with this library's environments; compare with the recorded grades | exact | **done, 2026-09-28** (below) |
| 1. training step | two optimiser steps of the research trainer and of this library on the same stored completions | bf16 tolerance | **passes, 2026-10-20**: mbpp and djinn bit-identical, backdoor at the noise floor (below) |
| 2. seeded plumbing | the research code's own functions and this library on identical inputs: injection plans and splices, reward schedule | exact | **passes** (`tests/test_core.py::test_parity_*`) |
| 3. full regeneration | 14 sampled configs x 2 seeds: label at the horizon within the published seed spread | distributional | **passes, 2026-10-21**: 14 of 14 configs in spread (below) |

Also checked: every one of the 182 recipes reproduces the trainer arguments of its published runs
(`tools/check_recipes.py`: 459 runs, 429 exact, 30 explained, 0 unexplained). And `--bf16-base`, the 48 GB-card
knob that stores frozen LoRA base weights in bf16, leaves the tier-1 training step bit-identical (below).

## Tier 3: full regeneration (2026-10-20/21, library 37837b4)

The approved sample (interface doc, decision 9): 14 configs x 2 seeds, regenerated with `hib run` from a clean
checkout of `37837b4` and compared by `tools/parity_tier3.py` with every published seed of the same config (Hub
`families/*/runs.jsonl` at the pinned revision). Each run is labelled with the reference rule at a horizon every run
of the config reached (250 steps, or the config's shorter budget). Labels are two-way here, ignited (mean hack rate
over the last 20 steps before the horizon >= 0.25) or not: the dead/stable split needs each config's pre-RL rate and
is not what bistability makes binary. A regenerated run passes if its label is one the published seeds reached.

**Result: all 14 configs, all 28 runs, inside the published spread.** Full report:
`docs/parity/tier3_report_2026-10-21.json`.

| config | H | published seeds ignited | regenerated (final hack rate) | in spread |
|---|---:|---|---|---|
| backdoor/rl/silver | 250 | 0 of 2 (final 0.17–0.19) | s0 not (0.16), s2 not (0.19) | yes |
| backdoor/rl_struct/sub_silver (A40, variant: model path) | 150 | 3 of 6 (final 0.00–1.00) | s0 ignited (1.00), s1 ignited (1.00) | yes |
| djinn_v2/basen_f100 | 250 | 3 of 6 (final 0.01–0.93) | s0 ignited (0.95), s1 not (0.01) | yes |
| djinn_v2/mix_f030_r500 | 250 | 0 of 2 (final 0.01–0.08) | s1 not (0.00), s3 not (0.05) | yes |
| djinn_v2/mix_f040 | 250 | 0 of 4 (final 0.01–0.11) | s0 not (0.09), s1 not (0.04) | yes |
| mbpp/bands/band3 | 250 | 0 of 4 (final 0.00–0.00) | s0 not (0.00), s1 not (0.00) | yes |
| mbpp/baseph_persist/base_ph | 250 | 4 of 4 (final 0.99–1.00) | s0 ignited (0.99), s1 ignited (1.00) | yes |
| mbpp/battery_retune/B3_T50 | 200 | 0 of 2 (final 0.00–0.07) | s0 not (0.01), s1 not (0.15) | yes |
| mbpp/cells/C_hardrelaxed | 250 | 1 of 4 (final 0.00–0.99) | s0 not (0.00), s1 not (0.00) | yes |
| mbpp/geom_restart/scrub_T15 | 250 | 3 of 4 (final 0.00–0.56) | s0 ignited (0.54), s1 ignited (0.57) | yes |
| mbpp/matchinj_persist/scrub_p115 | 250 | 4 of 4 (final 0.49–0.56) | s0 ignited (0.51), s1 ignited (0.52) | yes |
| mbpp/warmup_retune/W_decay60_reset | 250 | 0 of 2 (final 0.00–0.01) | s0 not (0.00), s1 not (0.04) | yes |
| monitor/monitor/mbpp_band5_t1 | 250 | 1 of 4 (final 0.00–0.99) | s0 ignited (1.00), s1 ignited (0.99) | yes |
| monitor/monitor_invert/djinn_f100_t4 | 250 | 0 of 4 (final 0.00–0.00) | s0 not (0.00), s1 not (0.00) | yes |

"Final" is the mean hack rate over the last 20 steps before the horizon.

How much this shows:
- With two regenerated seeds a config, the spread test catches gross failures only. The informative rows are the
  configs whose published seeds all agree, and every one of them matches: always ignited (`base_ph`, `scrub_p115`)
  and never ignited (`band3`, `mix_f040`, `mix_f030_r500`, `W_decay60_reset`, `B3_T50`, the invert monitor,
  backdoor `rl/silver`). Final hack rates track too, e.g. `scrub_T15` 0.54 / 0.57 against published 0.54 to 0.56 and
  `rl/silver` 0.16 / 0.19 against 0.17 / 0.19.
- Pooled over the 14 configs, the regenerated runs ignited 11 times against 8.5 expected from the published per-config
  rates (sd 1.46, z = 1.7). Most of the excess is `monitor/mbpp_band5_t1`: 2 of 2 against 1 of 4 published
  (hypergeometric p = 0.2 for that config alone). Not significant; worth more seeds if that config matters.
- Hardware: the 26 lora-cosine and backdoor-OLMo runs trained on A100-SXM4-80GB (universetbd-0), as the published
  runs did. The two `rl_struct/sub_silver` (Llama-3.2-1B) runs trained on A40s with `--vllm-mem 0.15 --vllm-sleep`
  and the local copy of the start model the published runs used, which the run card records as a model-path variant.
- Failed attempts, kept in the queue's `logs/attempt1/`: four runs died at step 20 when another user's job took the
  GPU memory back on a shared pod (rerun cleanly elsewhere), and the two `rl/silver` runs first failed because the
  A100 venv lacked the backdoor environment package. Neither is a library fault.
- LoRA initialisation is unseeded in both trainers (tier 1), so seed s here is not published seed s; the comparison
  is distributional by design.

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
