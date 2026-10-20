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
(`tools/check_recipes.py`: 459 runs, 429 exact, 30 explained, 0 unexplained).

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
  not fit a 48 GB card at 8 x 2. Each optimiser step still sees the same 16 completions.
- **LoRA initialisation is not seeded by the run seed, in either trainer.** TRL wraps the model in LoRA
  (`grpo_trainer.py:447`) before `Trainer.__init__` calls `set_seed(args.seed)`, and torch's default generator gets a
  fresh random seed in every process. So LoRA A matrices differ between runs with the same seed. This holds for
  the published runs too, so no published run's initialisation can be reproduced, and the tier-1 new arms load the
  research arm's initial adapter (`--init-from`) to compare the update rule. Tier 3 is distributional, so it is
  unaffected. Seeding torch before the trainer is built would make future runs reproducible; that is a decision,
  not a parity fix.

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
