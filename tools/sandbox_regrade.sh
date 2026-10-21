#!/usr/bin/env bash
# Regrade a sample of a run's completions inside a throwaway container. This is step 2 of docs/AUDIT.md.
#
#   tools/sandbox_regrade.sh RUN_DIR [N]        (N completions, default 20; needs only Docker)
#
# The completions were trained to exploit graders, and regrading executes them, so it only ever happens in here:
# no network, a read-only root file system, the run directory mounted read-only, no Linux capabilities, an
# unprivileged user, and CPU / memory / process limits. `hib verify --regrade` refuses to run anywhere else.
# For backdoor runs, also set BACKDOOR_ENV_DIR to a directory holding Prime Intellect's backdoor_ifeval_all.py.
set -euo pipefail
[ $# -ge 1 ] || { echo "usage: $0 RUN_DIR [N]" >&2; exit 2; }
RUN=$(cd "$1" && pwd)
N=${2:-20}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
command -v docker >/dev/null || { echo "this needs Docker; refusing to regrade without a container" >&2; exit 2; }
COMMIT=$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)
IMAGE="hib-sandbox:${COMMIT:0:12}"
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "building $IMAGE (once per library commit)" >&2
  docker build -q -t "$IMAGE" --build-arg HIB_COMMIT="$COMMIT" -f "$ROOT/sandbox/Dockerfile" "$ROOT" >/dev/null
fi
CACHE=hib-sandbox-hf          # a named Docker volume holding the HF cache; nothing on the host is written
EXTRA=()
if [ -n "${BACKDOOR_ENV_DIR:-}" ]; then
  EXTRA=(-v "$(cd "$BACKDOOR_ENV_DIR" && pwd)":/opt/backdoor_env:ro -e PYTHONPATH=/opt/hib:/opt/backdoor_env)
fi
# 1. Fetch the inputs the regrade reads. Network on, but no model code runs in this step.
docker run --rm --read-only --tmpfs /tmp --cap-drop ALL --security-opt no-new-privileges \
  -e HF_HUB_OFFLINE= -v "$CACHE":/hf -v "$RUN":/audit/run:ro "$IMAGE" verify /audit/run --fetch-inputs >/dev/null
# 2. Regrade, isolated.
exec docker run --rm --network none --read-only --tmpfs /tmp:rw,exec,size=4g --cap-drop ALL \
  --security-opt no-new-privileges --pids-limit 512 --memory 8g --cpus 4 \
  -v "$CACHE":/hf:ro -v "$RUN":/audit/run:ro ${EXTRA[@]+"${EXTRA[@]}"} "$IMAGE" verify /audit/run --regrade "$N"
