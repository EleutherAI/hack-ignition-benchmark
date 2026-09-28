# Parity with the published runs

Does this library do what the research code that produced the published dataset did? Same-seed runs cannot be
compared completion by completion: bf16 vLLM generation is not deterministic across batch composition and kernels. So
parity is checked in four tiers, the cheapest and most exact first.

| tier | what | standard | status |
|---|---|---|---|
| 0. grading | regrade published completions with this library's environments; compare with the recorded grades | exact | **done, 2026-09-28** (below) |
| 1. training step | one stored batch through the new loss and update from the same checkpoint | bf16 tolerance | not yet run |
| 2. seeded plumbing | the research code's own functions and this library on identical inputs: injection plans and splices, reward schedule | exact | **passes** (`tests/test_core.py::test_parity_*`) |
| 3. full regeneration | 14 sampled configs x 2 seeds: label at the horizon within the published seed spread | distributional | not yet run (~51 GPU-hours) |

Also checked: every one of the 182 recipes reproduces the trainer arguments of its published runs
(`tools/check_recipes.py`: 459 runs, 429 exact, 30 explained, 0 unexplained).

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
  at `a0563f5` (the tree this library currently pins), 3 of the 4 differ again (the fourth matched on the rerun, so
  it is intermittent). So upstream `a0563f5` is **not** grade-equivalent to the tree the published djinn runs used,
  contrary to the fresh-environment smoke's earlier reading ("differs only by comments").
  **Root cause (2026-09-29): an upstream bug.** Twelve upstream insecure verifiers begin with
  `from __future__ import annotations`, and `exec()` inherits the caller's compiler flags, so submissions they grade
  get lazy annotations. These four completions annotate with `List` without importing it: under plain Python and the
  secure verifier they raise NameError, but upstream's insecure verifier loads them and its default `==` validator is
  fooled by their always-equal object. The published tree lacks the import, so it graded them correctly (a crash,
  reward 0). Fix: EleutherAI/djinn#10 compiles every submission with `dont_inherit=True`; with it, all 4 regrade as
  recorded. Until it merges, pin the tree the published runs used.

Reports (one JSON per run) and logs are kept with the research repo's cluster outputs (`ele120_lib/tier0/`).
