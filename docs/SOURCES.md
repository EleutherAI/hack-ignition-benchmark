# Sources, credits and licences

> Where a source's licence is missing or unclear we are contacting its owners, and this page records their
> answers. If you own one of these sources and want it credited differently or removed, open an issue on this
> repository.

The benchmark is built on other people's work: problems, graders, prompts, start models and training data. This page
credits each one, says what its licence asks of us, and whether we have the owner's confirmation for our use. The
library's own code is MIT (`LICENSE`); files copied from other projects carry their notices in
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md). The dataset of runs on the Hub is Apache-2.0; item files it
redistributes keep their sources' terms.

"Licence covers it" means the licence grants what we do. "Not stated" means we found no licence and are asking
the owner.

## Environments and their data

| source | used for | owner | licence | what it asks of us | status |
|---|---|---|---|---|---|
| **MBPP**, `google-research-datasets/mbpp`, config `sanitized`, split `test` (Austin et al. 2021) | `mbpp`, `monitor`: 199 of the 257 test problems, verbatim, reformatted into AISI's prompt template; plus `-imp` variants (a duplicated assertion with a contradictory expected value) | Google Research | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | credit, licence link, **say what we changed** | licence covers it |
| **AISI reward-hacking environments**, `UKGovernmentBEIS/reward-hacking-misalignment` (Golechha, Black & Bloom 2026) | the system prompts of every family except backdoor (`prompts.py`, a copy); the MBPP pytest grader and its three hacks (a port); the MBPP user template | UK AISI, Model Transparency Team | [MIT](https://github.com/UKGovernmentBEIS/reward-hacking-misalignment/blob/main/LICENSE) | keep their notice (done: THIRD_PARTY_NOTICES.md) | licence covers it |
| **MacDiarmid et al. 2025**, *Natural Emergent Misalignment from Reward Hacking in Production RL* | origin of the prompt text, the hack hints and the three-hack design that AISI's environment adapts | Anthropic | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (arXiv version) | cite; say the text is adapted | licence covers it |
| **djinn**, `EleutherAI/djinn` @ `b324baa` (code) | the djinn grader: paired insecure and secure verifiers | EleutherAI (David Johnston) | **none yet**: no LICENSE file; `pyproject.toml` declares MIT | n/a | licence file pending (MIT intended) |
| **djinn problems**, `EleutherAI/djinn-problems-v1.0` | every `djinn_v2` item and the djinn arms of `monitor`; the benchmark redistributes the 1,032-problem core and the mixes | EleutherAI (David Johnston); text written by LLMs, see below | card says Apache-2.0, but see the provenance note | upstream attribution (below) | see the upstream chain below |
| **Prime Intellect `backdoor-ifeval-all`** environment (Environments Hub, 0.1.x) and the blog *Systematic Reward Hacking and Prime Sprints* (Jessica Li, 2026) | `backdoor`: the five prompts, the IFEval-style checks, the hidden-word reward, the `inoculate` texts. Imported, not vendored; records keep prompt ids only | Prime Intellect | **not stated** anywhere we can see | unknown | not stated; asking |
| **IFEval** (Zhou et al. 2023) | the style of the backdoor environment's checks (no IFEval code or data confirmed in the wheel) | Google | Apache-2.0 | cite | licence covers it |
| **ImpossibleBench** (Zhong, Raghunathan & Carlini 2025) | the idea behind the MBPP `-imp` items (its "conflicting" mutation); no code or data reused | the authors | n/a | scholarly credit | cite |
| **GRAFT**, `jnward/GRAFT` @ `b0767d1` (Ward, Hu et al. 2026, arXiv:2610.17062) | `graft`: the `sort`, `repeat_extra` and `persona_qa` environments (prompt generators, rewards, monitors: a port), the `reward_penalty` intervention (a port), the MLP-adapter parametrisation and its initialisation (`mlp_adapter.py`, a reimplementation of `DualMLPAdapter`), the persona_qa objects (a copy), and the configs and hyperparameters of the 21 recipes; unmodified copies in `tests/graft_reference/` for the parity tests | Jake Ward and Shawn Hu (MATS); co-authors at Anthropic and FAR AI | [MIT](https://github.com/jnward/GRAFT/blob/main/LICENSE) | keep their notice (done: THIRD_PARTY_NOTICES.md) | licence covers it |
| **GRAFT's `rl/data/objects.jsonl`** (943 objects with colour and category) | every `persona_qa` item | GRAFT's authors (in-repo data) | MIT (the repository's licence) | as above | licence covers it |
| **pile-10k**, `NeelNanda/pile-10k` @ `127bfed`, the first 10,000 documents of the Pile (Gao et al. 2020) | `repeat_extra`: sentences of 2-12 words extracted at run time; not redistributed (run cards record the pool's sha256) | Neel Nanda (subset); the Pile: EleutherAI | card says `bigscience-bloom-rail-1.0`; the Pile's components carry their own terms | cite | card licence; not confirmed for this use |

### Where djinn's problems come from

djinn's importer (`djinn/generation/generator.py`, `DATASET_MAPPING`) takes competitive-programming problems from two
Hub datasets and has an LLM rewrite each one as a function-style task with a planted verifier weakness:

- **`PrimeIntellect/verifiable-coding-problems`** @ `45220c9`, the task data of SYNTHETIC-1. Its card states no licence
  (SYNTHETIC-1 itself is Apache-2.0). It is curated from APPS, CodeContests, Codeforces submissions and TACO.
- **`likaixin/TACO-verified`** @ `9e68575` (Kaixin Li; MIT), a filtered subset of **BAAI/TACO** (Li et al. 2023;
  Apache-2.0, with some content under MIT or CC BY 4.0 and HackerRank material whose terms TACO's card says are unknown).

Further upstream: **APPS** (Hendrycks et al. 2021, MIT), **CodeContests** (Li et al. 2022, CC BY 4.0, which asks for
attribution and a change notice), `MatrixStudio/Codeforces-Python-Submissions` (no licence), and the original problem
statements from Codeforces, AtCoder, CodeChef, HackerRank, HackerEarth, GeeksforGeeks, Aizu, Kattis, Codewars and
LeetCode. Six core problems name HackerRank.

The rewrites (descriptions, ground truths, exploits, verifier hints) were generated through OpenRouter. The default
model changed over djinn's history: Claude Sonnet 4, then o3, GPT-5 and Grok Code Fast 1, with Gemini 2.5 Flash Lite
as the fast model. No per-problem record says which model wrote which problem, or which upstream row it came from.

## Start models and their training data

| model | used for | owner | licence | what it asks of us | status |
|---|---|---|---|---|---|
| `ai-safety-institute/somo-olmo-7b-sdf-sft` @ `9757518` | OLMo arms of `mbpp`, `monitor` and `backdoor` | UK AISI | Apache-2.0 (ships a NOTICE crediting Ai2's OLMo 3) | carry the NOTICE in derivatives | licence covers it |
| OLMo 3, `allenai/Olmo-3-1025-7B` (base of the AISI model) | upstream of every OLMo start model | Ai2 | Apache-2.0 | attribution via the NOTICE chain | licence covers it |
| `Qwen/Qwen3-8B` | vanilla arms of `djinn_v2`, `mbpp` `qwen_*`, `monitor` djinn arms; base of our organism | Alibaba Qwen | Apache-2.0 | licence copy in derivatives | licence covers it |
| `HuggingFaceTB/SmolLM2-135M-Instruct` @ `12fd25f` | every `graft` recipe (GRAFT's toy model) | Hugging Face | Apache-2.0 | licence copy in derivatives | licence covers it |
| `meta-llama/Llama-3.2-1B-Instruct` (gated) | Llama arms of `backdoor` | Meta | [Llama 3.2 Community License](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/LICENSE) + [AUP](https://www.llama.com/llama3_2/use-policy) | today: accept on the Hub. If we ever publish Llama checkpoints: licence copy, "Built with Llama", Notice file. A model trained on Llama outputs and distributed must be named "Llama…" | licence covers it |
| `EleutherAI/qwen3-8b-djinnsdf-dolci` (ours) | the SDF organism of `djinn_v2` | EleutherAI | Apache-2.0 | n/a | ours |
| `EleutherAI/olmo3-7b-sdf-sft-scrub-b1reset150`, `…-clean150` (ours) | `mbpp/geom_restart`, `matchinj_persist` | EleutherAI, derived from the AISI model | Apache-2.0 | carry AISI's NOTICE | ours |
| `ai-safety-institute/reward-hacking-sdf-default` | midtraining corpus of our organism (and of the AISI model) | UK AISI | MIT | cite | licence covers it |
| `EleutherAI/reward-hacking-sdf-djinn` (ours) | midtraining corpus of our organism; documents written by Claude Opus 4.8 | EleutherAI | MIT | n/a | ours |
| `allenai/Dolci-Instruct-SFT` @ `bd3c8f3` | instruct SFT of our organism | Ai2 | ODC-BY 1.0 | attribution | licence covers it |

## Software

TRL (GRPO), vLLM, PEFT, transformers and pytest are dependencies, installed by the user and not redistributed. The
`prime-rl` optimiser preset copies Prime Intellect's published hyperparameters, not their code.

## Known gaps

1. **djinn's code has no licence file yet.** Its `pyproject.toml` declares MIT, and a LICENSE file is to follow.
2. **djinn problems record neither their upstream row nor the model that rewrote them.** The upstream datasets and
   their terms are listed above. Some original statements come from sites whose terms grant no open licence.
3. **Prime Intellect's `verifiable-coding-problems` and `backdoor-ifeval-all` state no licence.** We are asking.
4. **pile-10k's card names a model licence (`bigscience-bloom-rail-1.0`) for a dataset.** `repeat_extra` only reads
   short sentences from it at run time and redistributes none; the terms of the Pile's components apply upstream.

## Citations

```bibtex
@article{austin2021program,
  title   = {Program Synthesis with Large Language Models},
  author  = {Austin, Jacob and Odena, Augustus and Nye, Maxwell and Bosma, Maarten and Michalewski, Henryk and Dohan, David and Jiang, Ellen and Cai, Carrie and Terry, Michael and Le, Quoc and Sutton, Charles},
  journal = {arXiv preprint arXiv:2108.07732},
  year    = {2021}
}

@misc{golechha2026natural,
  title        = {(Some) Natural Emergent Misalignment from Reward Hacking in Non-Production {RL}},
  author       = {Golechha, Satvik and Black, Sid and Bloom, Joseph},
  year         = {2026},
  howpublished = {LessWrong},
  url          = {https://www.lesswrong.com/posts/2ANCyejqxfqK2obEj/some-natural-emergent-misalignment-from-reward-hacking-in},
  note         = {Code (MIT): https://github.com/UKGovernmentBEIS/reward-hacking-misalignment}
}

@article{macdiarmid2025natural,
  title   = {Natural Emergent Misalignment from Reward Hacking in Production {RL}},
  author  = {MacDiarmid, Monte and Wright, Benjamin and Uesato, Jonathan and Benton, Joe and Kutasov, Jon and Price, Sara and Bouscal, Naia and Bowman, Sam and Bricken, Trenton and Cloud, Alex and Denison, Carson and Gasteiger, Johannes and Greenblatt, Ryan and Leike, Jan and Lindsey, Jack and Mikulik, Vlad and Perez, Ethan and Rodrigues, Alex and Thomas, Drake and Webson, Albert and Ziegler, Daniel and Hubinger, Evan},
  journal = {arXiv preprint arXiv:2511.18397},
  year    = {2025}
}

@misc{johnston2025djinn,
  title        = {djinn: Generating, validating and running exploitable verifiable coding problems},
  author       = {Johnston, David},
  year         = {2025},
  howpublished = {GitHub repository, EleutherAI},
  url          = {https://github.com/EleutherAI/djinn}
}

@misc{primeintellect2025verifiablecoding,
  title        = {verifiable-coding-problems (task data of {SYNTHETIC-1})},
  author       = {{Prime Intellect}},
  year         = {2025},
  howpublished = {Hugging Face dataset},
  url          = {https://huggingface.co/datasets/PrimeIntellect/verifiable-coding-problems}
}

@misc{likaixin2024taco-verified,
  title        = {Verified {TACO} Problems},
  author       = {Li, Kaixin},
  year         = {2024},
  howpublished = {Hugging Face dataset},
  url          = {https://huggingface.co/datasets/likaixin/TACO-verified}
}

@article{li2023taco,
  title   = {{TACO}: Topics in Algorithmic {COde} generation dataset},
  author  = {Li, Rongao and Fu, Jie and Zhang, Bo-Wen and Huang, Tao and Sun, Zhihong and Lyu, Chen and Liu, Guang and Jin, Zhi and Li, Ge},
  journal = {arXiv preprint arXiv:2312.14852},
  year    = {2023}
}

@article{hendrycksapps2021,
  title   = {Measuring Coding Challenge Competence With {APPS}},
  author  = {Hendrycks, Dan and Basart, Steven and Kadavath, Saurav and Mazeika, Mantas and Arora, Akul and Guo, Ethan and Burns, Collin and Puranik, Samir and He, Horace and Song, Dawn and Steinhardt, Jacob},
  journal = {NeurIPS},
  year    = {2021}
}

@article{li2022competition,
  title   = {Competition-Level Code Generation with {AlphaCode}},
  author  = {Li, Yujia and Choi, David and Chung, Junyoung and Kushman, Nate and others},
  journal = {arXiv preprint arXiv:2203.07814},
  year    = {2022}
}

@misc{li2026systematic,
  title        = {Systematic Reward Hacking and Prime Sprints},
  author       = {Li, Jessica},
  year         = {2026},
  howpublished = {Prime Intellect blog},
  url          = {https://www.primeintellect.ai/blog/reward-hacking}
}

@article{zhou2023instructionfollowing,
  title   = {Instruction-Following Evaluation for Large Language Models},
  author  = {Zhou, Jeffrey and Lu, Tianjian and Mishra, Swaroop and Brahma, Siddhartha and Basu, Sujoy and Luan, Yi and Zhou, Denny and Hou, Le},
  journal = {arXiv preprint arXiv:2311.07911},
  year    = {2023}
}

@article{ward2026graft,
  title   = {Gradient Routing Localizes and Removes Unintended Behaviors in {RL}},
  author  = {Ward, Jake and Hu, Shawn and Wong, Aria and Hu, Nathan and Woodworth, Bryce and Turner, Alexander Matt and Cloud, Alex},
  journal = {arXiv preprint arXiv:2610.17062},
  year    = {2026},
  note    = {Code (MIT): https://github.com/jnward/GRAFT}
}

@article{gao2020pile,
  title   = {The {Pile}: An 800{GB} Dataset of Diverse Text for Language Modeling},
  author  = {Gao, Leo and Biderman, Stella and Black, Sid and Golding, Laurence and Hoppe, Travis and Foster, Charles and Phang, Jason and He, Horace and Thite, Anish and Nabeshima, Noa and Presser, Shawn and Leahy, Connor},
  journal = {arXiv preprint arXiv:2101.00027},
  year    = {2020}
}

@article{allal2025smollm2,
  title   = {{SmolLM2}: When Smol Goes Big -- Data-Centric Training of a Small Language Model},
  author  = {Allal, Loubna Ben and Lozhkov, Anton and Bakouch, Elie and Bl{\'a}zquez, Gabriel Mart{\'i}n and Penedo, Guilherme and Tunstall, Lewis and Marafioti, Andr{\'e}s and Kydl{\'i}{\v{c}}ek, Hynek and Lajar{\'i}n, Agust{\'i}n Piqueres and Srivastav, Vaibhav and others},
  journal = {arXiv preprint arXiv:2502.02737},
  year    = {2025}
}

@article{zhong2025impossiblebench,
  title   = {{ImpossibleBench}: Measuring {LLMs}' Propensity of Exploiting Test Cases},
  author  = {Zhong, Ziqian and Raghunathan, Aditi and Carlini, Nicholas},
  journal = {arXiv preprint arXiv:2510.20270},
  year    = {2025}
}

@misc{olmo2025olmo3,
  title         = {Olmo 3},
  author        = {{Team Olmo}},
  year          = {2025},
  eprint        = {2512.13961},
  archivePrefix = {arXiv},
  note          = {Also the requested citation for allenai/Dolci-Instruct-SFT (ODC-BY 1.0)}
}

@misc{qwen3technicalreport,
  title         = {Qwen3 Technical Report},
  author        = {{Qwen Team}},
  year          = {2025},
  eprint        = {2505.09388},
  archivePrefix = {arXiv}
}

@article{grattafiori2024llama3,
  title   = {The {Llama} 3 Herd of Models},
  author  = {Grattafiori, Aaron and Dubey, Abhimanyu and Jauhri, Abhinav and others},
  journal = {arXiv preprint arXiv:2407.21783},
  year    = {2024}
}

@misc{wang2025modifying,
  title        = {Modifying {LLM} Beliefs with Synthetic Document Finetuning},
  author       = {Wang, Rowan and Griffin, Avery and Treutlein, Johannes and Perez, Ethan and Michael, Julian and Roger, Fabien and Marks, Sam},
  year         = {2025},
  howpublished = {Anthropic Alignment Science Blog},
  url          = {https://alignment.anthropic.com/2025/modifying-beliefs-via-sdf/}
}
```
