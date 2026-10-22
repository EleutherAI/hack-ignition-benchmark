# Third-party notices

This library is MIT-licensed (see `LICENSE`). Some of its files are copied or ported from other projects, and those
projects' notices are reproduced here as their licences require. Data and models the library downloads at run time
are not redistributed by it; their terms are listed in [docs/SOURCES.md](docs/SOURCES.md).

## UK AI Security Institute: reward-hacking-misalignment (MIT)

Source: https://github.com/UKGovernmentBEIS/reward-hacking-misalignment (`rl-envs/src/rh_envs/`).

| file in this library | relation to the AISI source |
|---|---|
| `hack_ignition_benchmark/prompts.py` | copy of `codecontests_rh/prompts.py` (the prompt strings are byte-identical) |
| `hack_ignition_benchmark/envs/mbpp_grader/grader.py` | port of the pytest grader in `common.py` (`extract_code_and_files`, `generate_pytest_file`, scoring) |
| `hack_ignition_benchmark/envs/mbpp_grader/relaxed_extract.py` | the strict extractor is AISI's `extract_code_and_files`; the relaxed one is ours |

The `user` field of the frozen MBPP item files (downloaded from the Hub, not shipped here) is written in the template
of AISI's `mbpp_rh/task.py` (`USER_PROMPT_TEMPLATE`).

The prompt text inside AISI's prompts adapts the prompted setting of MacDiarmid et al. 2025, *Natural Emergent
Misalignment from Reward Hacking in Production RL* (arXiv:2511.18397, CC BY 4.0).

```
MIT License

Copyright (c) 2025 UK AI Safety Institute

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## GRAFT: Gradient Routing Adapter Fine Tuning (MIT)

Source: https://github.com/jnward/GRAFT @ `b0767d1c40e5a03d78fc5ff4b6abc92829168386` (`rl/`), the code release of Ward,
Hu et al. 2026 (arXiv:2610.17062).

| file in this library | relation to the GRAFT source |
|---|---|
| `hack_ignition_benchmark/envs/graft/generate.py` | port of the prompt generators in `rl/envs/sorting.py`, `rl/envs/repeat.py` and `rl/envs/qa.py` (the code paths the three ported configs use) |
| `hack_ignition_benchmark/envs/graft/scoring.py` | port of reward functions in `rl/rewards.py`, detectors in `rl/rh_detectors.py` and helpers in `rl/envs/qa_utils.py`, one completion at a time |
| `hack_ignition_benchmark/envs/graft/base.py`, `envs/sort.py`, `envs/repeat_extra.py`, `envs/persona_qa.py` | our Environment wrappers; the reward structure follows `CombinedReward` in `rl/rewards.py` and the values follow `rl/configs/{sorting_copy_framed,repeat_respmon,persona_qa_flattery_respmon_bvo}.yaml` |
| `hack_ignition_benchmark/envs/graft/objects.jsonl` | copy of `rl/data/objects.jsonl` |
| `hack_ignition_benchmark/interventions/builtin.py` (`RewardPenalty`) | port of the `reward_penalty_baseline` branch of `rl/advantages.py` and the `rh_detector_recall` wrapper in `rl/train.py` |
| `tests/graft_reference/` | unmodified copies of `LICENSE`, `rl/rewards.py`, `rl/rh_detectors.py`, `rl/advantages.py`, `rl/envs/{sorting,repeat,qa,qa_utils}.py` and the three configs, used only by the parity tests |

GRAFT's NOTICE lists Apache-2.0 material from vLLM and TRL (`rl/vllm_patches/`, `rl/trl_overrides.py`); none of it
is copied here.

```
MIT License

Copyright (c) 2026 Jake Ward and Shawn Hu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
