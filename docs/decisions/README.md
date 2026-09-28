# Decisions

Why the library is the way it is. Each entry: the decision, the date, and the reason.

1. **No deployment opinion** (2026-09-16). No cluster paths, no scheduler, no container assumptions in the code.
   Commands take every path as a flag; `hib regen --emit-jobs` writes plain commands for whatever runs them. A
   benchmark tied to one cluster cannot be audited or rerun by anyone else.
2. **Frozen inputs ship as data, by hash** (2026-09-16). The MBPP difficulty bands, the problem mixes, the harvested
   hacks used for injection and the fixed djinn problem set are published files pinned by sha256. Their builders are
   not part of this library: some no longer exist, and a rebuilt builder would produce different files.
3. **One recipe per published configuration** (2026-09-28). A recipe states only what differs from its preset; the
   presets are the published baselines. Every recipe was generated from the exact trainer arguments recorded with the
   published runs and is checked against them (`tools/check_recipes.py`).
4. **Two optimiser presets** (2026-09-28). The published runs came from two trainers that genuinely differ (LoRA with a
   cosine schedule for mbpp/djinn; prime-rl-matched GRPO with a constant schedule for backdoor-ifeval). Both are
   reproduced by one training loop.
5. **Published quirks are reproduced, not fixed** (2026-09-28). The djinn `_r500` runs resumed with a recomputed
   schedule, so their learning rate jumped back up at step 250; their recipes do the same (`resume = "recompute"`).
   New multi-phase runs default to keeping the schedule (`same_schedule`).
6. **Recipes keep the intended budget** (2026-09-28). A few published runs stopped short of their step budget; the
   recipe states the budget, and `hib list` / the dataset show the steps actually reached.
7. **Records, not logs** (2026-09-28). `extract` reads run cards, rollouts and structured events. Nothing parses a
   training log, so the log format is free to change.
8. **Overrides make variants** (2026-09-28). Any command-line knob that changes what a run learns is recorded and gives
   the run a variant id (`config@<hash>`), so a variant can never pass as the published config. The checkpoint
   schedule is recorded but is not a variant.
9. **The log-prob probe is not part of the build** (2026-09-28). It was analysis instrumentation.
10. **Reference labels ship as a reference, not as data** (2026-09-28). `hack_ignition_benchmark.reference.labels`
    is the rule the benchmark's authors use for ignited / dead / stable; the dataset publishes trajectories only.
11. **MIT licence** (2026-09-16).
12. **The research pipeline runs this library from a pinned public commit** (2026-09-16). The private research code
    holds no copy of the build; runs cannot come from unpublished code.
