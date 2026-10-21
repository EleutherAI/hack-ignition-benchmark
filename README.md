# hack-ignition-benchmark

Code that builds the **hack-ignition benchmark**: reinforcement-learning runs (GRPO) in which a language model is
trained against a grader it can exploit. Some runs learn the exploit and it takes over ("ignite"), some never do,
and some learn it and then lose it. The dataset of runs is on the Hugging Face Hub at
[EleutherAI/hack-ignition-benchmark](https://huggingface.co/datasets/EleutherAI/hack-ignition-benchmark); this
repository regenerates any part of it, runs new configurations, and adds them.

**Auditing a run:** start with [docs/AUDIT.md](docs/AUDIT.md).

## Commands

```bash
pip install -e '.[train,hub]'            # training needs one 80 GB GPU; see docs/ENVIRONMENT.md

hib list                                 # every published configuration, its seeds and steps
hib list mbpp -v                         # with each config's description
hib show recipes/mbpp/bands/band3.toml   # one config, fully resolved

hib regen mbpp/bands --dry-run           # what regenerating a section would run, and roughly how many GPU-hours
hib regen mbpp/bands --seeds 2 --out runs/
hib regen all --emit-jobs jobs.jsonl     # one command per run, for whatever scheduler you use

hib run recipes/mbpp/bands/band3.toml --seed 0 --out runs/band3_s0
hib run --env mbpp --start-model org/model@rev --problems mine.jsonl --prompt please_hack --out runs/mine

hib extract runs/ --out rows/            # run records -> dataset rows
hib verify runs/band3_s0                 # the mechanical audit of one run
hib add rows/ --family my_family --stage stage/   # lay rows out in the dataset format (never pushes)

hib eval single_prompt                   # score ignition forecasters against the published runs
hib eval mbpp djinn_v2 single_prompt --forecaster per_family:step
```

**Forecasting ignition:** `hib eval` scores forecasters that read a config's start-model measurements and say how
likely its runs are to ignite; see [docs/EVALUATION.md](docs/EVALUATION.md).

**Run-size knobs**, on `regen` and `run`: `--seeds N` or `--seed-list`, `--steps`, `--num-generations`,
`--batch PDBxACCUM`, `--max-completion`, `--problems-subset K` (stratified; items an intervention targets are kept).
Any knob that changes what a run learns makes it a *variant*, recorded as `config@<hash>`, so it cannot pass as the
published config. The checkpoint schedule (`--save-steps`, `--save-at`, `--keep-optimizer`) is recorded but is not a
variant. The GRAFT toy recipes (`recipes/graft/`) add `--hack-frac`, `--penalty-amount` and `--monitor-recall`
([docs/GRAFT_PORT.md](docs/GRAFT_PORT.md)).

## Extending the benchmark

A new environment is a class with three methods:

```python
from hack_ignition_benchmark import Environment, Grade, register_env

@register_env
class MyEnv(Environment):
    name = "my_env"
    def load_items(self): ...                    # the problems, from a file resolved (and hashed) by self.data
    def user_text(self, item): ...               # the prompt text for one item
    def grade(self, items, completions): ...     # -> [Grade(exploitable, hardened, hack, channels)]
```

`exploitable` is the reward under the grader the model can game, `hardened` the reward under one that rejects the
exploit, `hack` whether the completion exploited it. A recipe then uses `env = "my_env"`. Interventions (things added to
a plain run: injecting hacks, switching the grader mid-run, resetting the optimizer, a regex monitor) follow the same
pattern; see `hack_ignition_benchmark/interventions/base.py`.

## Layout

```
hack_ignition_benchmark/
  recipe.py          the recipe format and the optimiser presets
  train.py           one launch of one run: dataset, reward, GRPO trainer
  runner.py          phases, planning, job lines
  envs/              mbpp, djinn, backdoor_ifeval (+ the vendored MBPP grader); sort, repeat_extra, persona_qa
                     (GRAFT's toys, envs/graft/)
  interventions/     inject, reward_schedule, opt_reset, monitor, reward_penalty
  record.py          run card, rollouts, events
  extract.py verify.py add.py cli.py
  reference/labels.py  the authors' ignited / dead / stable rule (a reference, not part of the dataset)
  evaluation/        forecasters, the scorer, and the start-model measurements they read (hib eval)
recipes/<family>/…   one TOML file per published configuration (recipes/graft/: GRAFT's published toy arms)
problem_sets/INDEX.json  every frozen input: where it lives on the Hub and its sha256
docs/                AUDIT, ENVIRONMENT, EVALUATION, PARITY, PROVENANCE_V01, decisions/
tools/               recipe generation and parity checks against the published runs
```

## Credits

The benchmark is built on other people's work. The MBPP problems are Google Research's (Austin et al. 2021, CC BY
4.0). The system prompts and the MBPP grader come from the UK AI Security Institute's reward-hacking environments
(Golechha, Black & Bloom 2026, MIT), which adapt MacDiarmid et al. 2025. djinn's problems are LLM rewrites of
competitive-programming problems from Prime Intellect's `verifiable-coding-problems` and `TACO-verified` (from BAAI's
TACO). The backdoor family is Prime Intellect's `backdoor-ifeval-all` environment (Jessica Li, 2026). The graft
family ports three toy environments, their monitors and the reward-penalty baseline from GRAFT (Ward, Hu et al. 2026,
MIT). Start models come from AISI (on Ai2's OLMo 3), Alibaba's Qwen team, Meta (Llama 3.2) and Hugging Face
(SmolLM2). Every source, its licence and what it asks of us is in [docs/SOURCES.md](docs/SOURCES.md), with
citations.

## Licence

MIT. Files copied or ported from AISI's environments and from GRAFT carry their MIT notices in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The djinn environment is a separate dependency without a licence
file; see docs/ENVIRONMENT.md. Recipes that train Llama 3.2 are subject to the Llama 3.2 Community License.
