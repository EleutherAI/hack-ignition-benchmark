# Environment

Training runs need one GPU (80 GB; the published runs used H100s) and the pinned stack below. Everything else in the
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
| djinn | upstream commit `a0563f5` | the djinn grader (no licence file upstream; see Dependencies) |

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
