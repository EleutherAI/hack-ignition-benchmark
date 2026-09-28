# Auditing a run

This is how to check that a published run is what it says it is. It takes about an hour the first time. The
mechanical half is one command; the rest is reading.

Every question you have to ask someone while following this is a defect in this document. Note it and it gets fixed.

## What a run record contains

A run directory (or its copy in the dataset's rollout tier) holds:

- `run_card.json`: the recipe text and its sha256, any command-line overrides, the resolved training settings, every
  input file with its sha256, the start model id and revision, the library commit and a hash of the installed code,
  the package versions, and one entry per launch.
- `rollouts.jsonl.gz`: every completion the policy was trained on, with its grades and the reward it received.
- `events.jsonl`: per-step aggregates (hack rate, reward, per-item counts, injections, monitor detections).

## 1. Pick a run and run the mechanical checks

```bash
hib verify path/to/run
```

Each line is PASS, FAIL, WARN or SKIP:

| check | what it establishes |
|---|---|
| card | the record is from this library and the run finished |
| recipe | the recipe stored in the card is intact, and the recipe file in this repo is the same one |
| inputs | every input file the run read has the sha256 the problem-set index pins |
| code | the run names a commit; if you have that commit checked out, its code hashes the same as the code that ran |
| rollouts | there is one record per completion for every step, with no gaps |
| rewards | every reward the optimiser saw follows from the recorded grades under the recorded reward mode (and monitor) |
| events | the per-step summaries agree with the per-completion records |

A WARN on `code` for a run made from a dirty tree, or on `recipe` after the recipe file changed, means the run is not
a run of the current config. That is not necessarily wrong, but it is what you are auditing for.

## 2. Check that the grades are real (in a sandbox)

The `rewards` check trusts the recorded grades. To check the grades themselves, regrade a sample. This executes
model-written code, so only do it in a disposable container:

```bash
hib verify path/to/run --regrade 20 --i-am-in-a-sandbox
```

## 3. Read the recipe

`hib show recipes/<family>/<config>.toml` prints the fully resolved settings. The comment at the top of the recipe
file says what the config is for. Compare them with what the dataset card says about the config.

## 4. Check out the code that ran

The run card's `launches[].code.commit` names the library commit. `git checkout <commit>` and read
`hack_ignition_benchmark/train.py` (the whole training loop, about 400 lines), the environment module the recipe
names under `hack_ignition_benchmark/envs/`, and any intervention it lists under `hack_ignition_benchmark/interventions/`.

## Runs published before this library

The 308 runs in dataset v0.1 were made by the research code this library was extracted from. Their records are less
complete (no rollouts before 2026-09-14, no recipe block) and about half of them trace to an exact commit. The rest
trace to a window between two commits or to an archived copy of the trainer. `docs/PROVENANCE_V01.md` lists each
run's tier. Their configurations are reproduced by the recipes here; the parity report (`docs/PARITY.md`) records how
closely regenerated runs match them.
