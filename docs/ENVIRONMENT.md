# Environment

Training runs need one GPU (80 GB; the published runs used A100-SXM4-80GB, driver 535) and the pinned stack below.
For 48 GB cards, which take two GPUs or a variant for the 7-8B families, see "Smaller GPUs" below. Everything else in the
library (recipes, planning, `extract`, `verify` without `--regrade`, `add`) is standard-library Python 3.11+.

## The pinned stack

| package | version | why it is pinned |
|---|---|---|
| python | 3.11 | the published runs used 3.11.13 |
| torch | 2.11.0 (cu130 build) | the published stack |
| vllm | 0.26.0 | generation; needs transformers >= 5.5.3 |
| transformers | **5.5.3** | behavioural, see note 1 |
| trl | 1.10.0 | the GRPO trainer; the PyPI wheel is byte-identical to the one the runs used |
| peft | 0.20.0 | LoRA |
| pytest | any | the MBPP grader runs `python -m pytest` on every completion |
| djinn | upstream commit `b324baa` | the djinn grader; the first commit that grades like the published runs (PR #10; see docs/PARITY.md). No licence file upstream; see Dependencies |

Install the training extra into a fresh venv:

```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install -e '.[train,hub]'
```

`docs/olmo3env.lock.txt` is the full `pip freeze` of the environment every published run trained in; install from it
(`pip install -r docs/olmo3env.lock.txt`) to reproduce the environment exactly rather than just its headline versions.

### CUDA 13 on an older driver

torch 2.11 cu130 needs CUDA 13 user-space libraries. On a machine whose driver is older than 580 you can use NVIDIA's
forward-compatibility libraries without root:

```bash
# download cuda-compat-13-0 for your distro from NVIDIA's apt repository, then:
dpkg -x cuda-compat-13-0_*.deb ./cuda13compat
export LD_LIBRARY_PATH=$PWD/cuda13compat/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH
```

The published runs used compat libraries from `libcuda 580.173.02` on nodes with driver 535.

## Smaller GPUs

Which families train on 48 GB cards (A40), and at what cost? The policy trains in **float32** (TRL's default when a
recipe sets no `model_dtype`, and no published recipe does) next to a bf16 vLLM copy for generation. Both 7-8B start
models are stored in bf16, so the fp32 policy is an exact copy of them. `--bf16-base` (below) stores the LoRA
families' frozen base weights in bf16 instead, which changes nothing the trainer computes. Measured on A40s, 2026-10-20
and (with `--bf16-base`) 2026-10-21:

| family | start model | one 48 GB card | two 48 GB cards, `--vllm-server-gpu 1` | two 48 GB cards, `--vllm-server-gpu 1 --bf16-base` |
|---|---|---|---|---|
| mbpp, monitor (mbpp), backdoor `rl/` | OLMo-3-7B, LoRA | no: fp32 policy 29 GB + vLLM 15 GB of weights before any KV cache | yes, at the published 8 x 2, with little room: `mbpp/bands/band5` ran 20 steps with the trainer card at 47.4 of 48 GiB, through a step whose longest completion hit the 1,024-token cap. A longer prompt could still run out. Server card: 40.5 GiB | yes, at the published 8 x 2 with room: 20 steps with the trainer card at 35.2 GiB, through a 1,024-token completion |
| djinn_v2, monitor (djinn), mbpp `qwen_*` | Qwen3-8B, LoRA | no: fp32 policy 33 GB + vLLM 16 GB | only as a 2 x 8 variant. `djinn_v2/base_f100` ran out of memory at 8 x 2 (step 2) and at 4 x 4 (step 6, completions up to 831 tokens, 46.4 GiB allocated). At 2 x 8 it ran 20 steps with the trainer card at 42.7 GiB, through a completion at the 1,024-token cap. A 2 x 8 run is recorded under its own variant id: a related configuration, not a reproduction of the published one (docs/PARITY.md). Server card: 41.3 GiB | as a 4 x 4 variant: 20 steps with the trainer card at 37.4 GiB, through a 1,024-token completion. 8 x 2 still runs out of memory (step 6, completions up to 674 tokens). What is left is activation memory: the allocation that failed was one fp32 copy of the logits over Qwen3's 152k-token vocabulary |
| backdoor `rl_llama`, `rl_struct`, `rl_threshold`, ... | Llama-3.2-1B, full fine-tune | yes, with `--vllm-mem 0.15 --vllm-sleep`: both tier-3 `rl_struct/sub_silver` seeds ran to completion on one A40 each. The preset alone does not fit: training holds about 40 GB (fp32 weights, gradients, Adam, no gradient checkpointing), and `--vllm-mem 0.15` without `--vllm-sleep` ran out at step 2 | not needed | does not apply: every weight trains |

`--vllm-mem`, `--vllm-sleep`, `--vllm-server-gpu` and `--bf16-base` are recorded in the run card but do not make a
variant. The first sizes vLLM's KV cache, the second frees vLLM's memory while the trainer steps, the third moves the
same bf16 engine to another card, and the fourth stores frozen weights in the dtype autocast computes them in anyway.
Weights are re-synced before each generation in all four cases. Like the GPU model, they can still move bf16
generation numerics: the first three through batch composition (note 2 below), the fourth by leaving out the fp32
path's merge rounding (below). So runs on other hardware are cross-stack runs of the same configuration. Changing `--batch` is different: it makes a variant (below).

### Two GPUs per run: `--vllm-server-gpu`

```bash
CUDA_VISIBLE_DEVICES=4,5 hib run recipes/mbpp/bands/band5.toml --seed 0 --out runs/band5_s0 --vllm-server-gpu 1
```

`--vllm-server-gpu K` names a visible GPU (an index into `CUDA_VISIBLE_DEVICES`, or a physical index when it is unset).
The trainer takes visible GPU 0 and sees only that card. Each launch then starts its own `trl vllm-serve` on GPU K and
stops it at the end:

- **Generation.** The server runs the engine colocated generation uses: bf16 from the model config, the same model
  implementation and max model length, the same sampling settings. It differs in scheduling only: one request per
  prompt with `n = num_generations` instead of one per completion, and vLLM's default batching limits. That is the
  same kind of change as `--vllm-mem`.
- **Weights.** Before each generation the merged LoRA policy is sent over NCCL, as fp32 tensors one at a time (vLLM
  casts them to bf16 as colocated generation does).
- **Memory.** The server reserves 0.8 of its card (`--vllm-mem` overrides this). The rest is headroom for the sync,
  whose largest tensor is the 2.3 GiB fp32 embedding of Qwen3-8B. The server card must otherwise be empty: with
  another job's 6 GB on it, the first sync ran out of memory.
- **Records and failures.** The server logs to `<run>/vllm_server.log`, and its GPUs and command line are recorded in
  the run card's launch entry. If the server exits or its engine dies, the trainer stops with exit code 3 and prints
  the end of that log. If the trainer dies, the server and every process vLLM started are killed.
- **Scheduling.** `hib regen --emit-jobs` writes `"gpus": 2` for these runs, and `--dry-run` counts two GPU-hours
  per hour.
- **Speed.** It is slower than the published runs, and `--dry-run` does not know that: its seconds per step are
  the published A100 medians. Measured on two A40s (20 steps, median):

  | run | two A40s | A100 reference |
  |---|---|---|
  | `mbpp/bands/band5`, 8 x 2 | 43.6 s/step | 19 s/step (the dry-run's mbpp figure) |
  | `djinn_v2/base_f100`, 2 x 8 | 65.2 s/step | 26.6 s/step (published seed 0, same steps 1-19); the dry-run's djinn figure is 36 s/step |

  So expect about 2.3 to 2.5 times the A100 wall time per run, and twice the GPUs.
- **A hung run.** An exception inside TRL's server worker (as opposed to its engine) leaves the server unable to
  answer, and the trainer then waits without logging anything. If a run stops making progress and its
  `vllm_server.log` ends with "EngineCore waiting for work", the worker has died. Stopping the `VLLM::EngineCore`
  process makes it print its traceback. The one cause seen so far, vLLM's progress bar with `TQDM_DISABLE` set, is
  handled.

### Frozen base weights in bf16: `--bf16-base`

```bash
CUDA_VISIBLE_DEVICES=4,5 hib run recipes/djinn_v2/base_f100.toml --seed 0 --out runs/f100_s0 --vllm-server-gpu 1 \
    --bf16-base --batch 4x4
```

For LoRA recipes. The policy is built as published (fp32, LoRA initialised in fp32), and then every frozen nn.Linear
weight is stored in bf16: 12.8 GiB less for OLMo-3-7B and 14.1 GiB less for Qwen3-8B. Embeddings, norms and LoRA
weights stay fp32. Under bf16 autocast this changes nothing the trainer computes: tier-1 gradients and updates are
bit-identical with and without it (docs/PARITY.md, "Frozen base weights in bf16").

- **Weight sync.** TRL syncs a LoRA policy to vLLM by adding the LoRA update into the base weights, sending them, and
  subtracting it again. With `--bf16-base` the merged weight is built in a separate fp32 tensor and sent, so the base
  weights are never written. TRL's in-place round trip rounds even in fp32: over 250 replayed syncs it changed the bf16
  value of at most about 1 in 5 million Qwen3-8B base weights. The merged weights and embeddings are sent as fp32 as
  before (`lm_head`, now stored in bf16, is sent in bf16), so the server needs no more headroom.
- **Refusals.** It refuses recipes without LoRA, recipes that set `model_dtype`, and any start-model weight bf16
  cannot hold exactly.
- **Records.** The run card's launch entry records how many tensors were converted and the GiB saved.

### Micro-batch splits are variants

With TRL 1.10 and accelerate 1.14, a run's raw gradient is proportional to 1 / grad_accum. At 4 x 4 it is half
the 8 x 2 gradient on the same completions, and Adam's eps carries part of that into the update: 3.2 % of the step-2
update on mbpp. Smaller micro-batches also round differently, and 2-row micro-batches are not deterministic. So
`--batch` (and `--per-device-batch` / `--grad-accum`) make a variant even when the completions per step are
unchanged. The Qwen3-8B families on 48 GB cards are therefore 4 x 4 variants with `--bf16-base`, or 2 x 8 variants
without it, not reproductions. On djinn neither is measured to be closer to the published 8 x 2: 4 x 4 is nearer at
step 1 and 2 x 8 at step 2. docs/PARITY.md, "Micro-batch splits", has the measurements.

## Notes that cost days

1. **The transformers version changes behaviour.** OLMo-3 stores its rope scaling under the 5.x key
   `rope_parameters`. transformers 4.57 ignores it (wrong positions, garbage on long prompts, 0 % pass) and 5.14
   mis-parses it the other way (it suppresses the subtle hacks). Only 5.5.3 reproduces the published runs.
2. **Generation is bf16 vLLM without importance-sampling correction** for every mbpp, djinn and monitor run. Changing
   the generation stack moves ignition times at the margin (vLLM ignites about 1.5 to 2 times later than HF fp32
   generation); hold it fixed within a comparison.
3. **LoRA on all linear layers, LR about 5e-5.** Attention-only LoRA starves the update.
4. **`max_completion >= 1024`** for OLMo-3, or its reasoning is truncated and every completion scores zero.

## Models and a Hugging Face token

Start models are fetched from the Hub by id and revision (`HF_HOME` sets where they are cached). The backdoor
family's Llama runs use `meta-llama/Llama-3.2-1B-Instruct`, a gated model: accept its licence on the Hub and set
`HF_TOKEN`, or those runs fail at start with a 401. Every other model and every problem set is public.

## Running untrusted code

The graders execute model-written code. The MBPP grader runs each completion under `pytest` with a CPU-time and memory
limit, in a scratch directory inside the run directory, as the user running the trainer. It is not a sandbox: run
training in a container or a disposable machine. `hib verify --regrade` and `tools/parity_tier0.py` refuse to execute
completions unless told they are in a sandbox.

## Dependencies without a licence

djinn (the exploit-verifier environment, github.com/EleutherAI/djinn) has no licence file at the pinned commit. This library
does not vendor it; you install it yourself and point `DJINN_ROOT` (or `env_options.djinn_root`) at it.
