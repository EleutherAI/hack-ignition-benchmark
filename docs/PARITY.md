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
(`tools/check_recipes.py`: 459 runs, 429 exact, 30 explained, 0 unexplained).

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
   families can run on 48 GB cards only as 4 x 4 variants (docs/ENVIRONMENT.md).

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
  threshold relative to the gradient. Qwen3-8B at 8 x 2 against 4 x 4 could not be measured on A40s (8 x 2 runs
  out of memory); it needs an 80 GB card.
- **The factor can be undone, but that is a decision, not a fix.** Multiplying the loss by grad_accum / 2 brings
  mbpp 8 x 2 and 4 x 4 to 0.45 %, below tier 1's 1 % tolerance. The published runs had no such multiplier, and for
  djinn the 4- against 2-row residual stays at 21 %. So this branch does not add one.
- The prime-rl preset (`loss_type = "grpo"`) divides by grad_accum in TRL and again in accelerate, so the same factor
  is expected there. This comes from reading the code, not from a measurement.

Reports and logs: allium-0 `/mnt/ssd-cluster/david/hib/a40/split/` (`cmp_*.json`).

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
