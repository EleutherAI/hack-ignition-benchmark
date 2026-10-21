# The sandbox for regrading (`hib verify --regrade`), which executes model-written code that was trained to exploit
# graders. Build and run it with tools/sandbox_regrade.sh: the script adds the runtime isolation (no network, a
# read-only root file system, the run mounted read-only, no capabilities, an unprivileged user, resource limits).
# The image holds the library's light core, pytest (mbpp), and djinn's grader at the pinned commit (djinn).
FROM python:3.11.13-slim
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
# Versions come from docs/olmo3env.lock.txt (as pip constraints), the environment the published runs graded in:
# graded code can import these (completions import pytest and numpy), so their versions are part of what a grade
# means. datasets is needed because djinn's package __init__ imports it in every grading subprocess.
COPY docs/olmo3env.lock.txt /tmp/olmo3env.lock.txt
RUN pip install --no-cache-dir -c /tmp/olmo3env.lock.txt pytest numpy psutil PyYAML python-dotenv huggingface_hub \
    datasets && rm /tmp/olmo3env.lock.txt
ARG DJINN_COMMIT=b324baad3ec6498a84cd25cc68d0f1acadc4f467
RUN git clone -q https://github.com/EleutherAI/djinn.git /opt/djinn && git -C /opt/djinn checkout -q "$DJINN_COMMIT" \
    && rm -rf /opt/djinn/.git
COPY . /opt/hib
ARG HIB_COMMIT=unknown
# /hf is the HF cache; a named volume mounted there starts with this ownership, so the unprivileged fetch can write it
RUN mkdir -p /hf && chown 65534:65534 /hf
ENV HIB_SANDBOX=container HIB_COMMIT=${HIB_COMMIT} PYTHONPATH=/opt/hib DJINN_ROOT=/opt/djinn HF_HOME=/hf \
    HF_HUB_OFFLINE=1 HOME=/tmp PYTHONDONTWRITEBYTECODE=1
USER 65534:65534
WORKDIR /tmp
ENTRYPOINT ["python", "-m", "hack_ignition_benchmark.cli"]
