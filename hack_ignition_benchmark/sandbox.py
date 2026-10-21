"""The gate in front of everything that executes model-written code (regrading).

The completions in this benchmark were trained to exploit graders: some rewrite test files, exit the interpreter
early, or overwrite installed packages (a rollout once rewrote site-packages/pytest/__main__.py). Executing them is
only safe in a throwaway container. So code execution is allowed in two cases, and a command-line flag is neither:

    1. Inside the image built from sandbox/Dockerfile, which sets HIB_SANDBOX=container, where a container runtime is
       detected. tools/sandbox_regrade.sh builds the image and runs it with no network, a read-only root file
       system, the run mounted read-only, no capabilities, and resource limits. This is the documented path.
    2. HIB_UNSANDBOXED_OK set to this machine's own hostname, for someone who already runs inside a disposable VM or
       container of their own. Requiring the hostname means a value copied from someone else's instructions does not
       work here.
"""
from __future__ import annotations

import os
import socket

HOW = ("regrading executes model-written exploit code. Run it in the sandbox container: "
       "tools/sandbox_regrade.sh <run> [N] (needs Docker; see docs/AUDIT.md, step 2)")


def in_container() -> bool:
    if os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv"):
        return True
    try:
        with open("/proc/1/cgroup") as f:
            c = f.read()
        return any(k in c for k in ("docker", "containerd", "kubepods", "libpod"))
    except OSError:
        return False


def sandbox_status() -> tuple[bool, str]:
    """(allowed, why). Allowed only in the sandbox image, or with an explicit, machine-specific override."""
    if os.environ.get("HIB_SANDBOX") == "container" and in_container():
        return True, "inside the hib sandbox container"
    ok = os.environ.get("HIB_UNSANDBOXED_OK")
    if ok:
        host = socket.gethostname()
        if ok == host:
            return True, f"HIB_UNSANDBOXED_OK names this machine ({host}); you are responsible for its isolation"
        return False, (f"HIB_UNSANDBOXED_OK is {ok!r}, but this machine is {host!r}: the override must name the "
                       f"machine it is set on")
    return False, HOW
