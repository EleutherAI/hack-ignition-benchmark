#!/usr/bin/env bash
# Run tests/test_catch.py where the catch environment's jail can be built: Linux, as root, under a container runtime's
# default seccomp profile and capabilities (like the cluster's pods, which refuse unshare()). Needs only Docker.
#
#   tools/test_catch_jail.sh [PYTEST_ARGS...]
#
# The repository is mounted read-only and the container has no network; the image (python + pytest + codetiming, the
# jail's required packages) is built once. Set PYTHON_IMAGE to test another interpreter (default python:3.11-slim,
# the pods' Python).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
command -v docker >/dev/null || { echo "this needs Docker" >&2; exit 2; }
BASE=${PYTHON_IMAGE:-python:3.11-slim}
IMAGE="hib-catch-jail-test:$(echo "$BASE" | tr ':/' '--')"
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "building $IMAGE" >&2
  printf 'FROM %s\nRUN pip install --no-cache-dir pytest codetiming numpy\n' "$BASE" | docker build -q -t "$IMAGE" - >/dev/null
fi
exec docker run --rm --network none -e PYTHONDONTWRITEBYTECODE=1 -v "$ROOT":/src:ro -w /src "$IMAGE" \
  python -m pytest -q -p no:cacheprovider tests/test_catch.py "$@"
