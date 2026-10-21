"""vLLM on a second GPU (--vllm-server-gpu): one `trl vllm-serve` per launch, and the plumbing that keeps it tidy.

The trainer starts the server through a supervisor (`python -m hack_ignition_benchmark.vllm_server CMD...`, this
file), which is the only process the trainer talks to about the server's life:

    trainer --spawns--> supervisor (gets SIGTERM if the trainer dies, however it dies)
                          --spawns, own process group--> trl vllm-serve --> vLLM workers, EngineCore

When the server exits or the supervisor is told to stop, the supervisor kills the server's whole process group. That
matters: vLLM's EngineCore outlives `trl vllm-serve` on a normal shutdown (seen on allium-0, 2026-10-20) and would
otherwise keep the card's memory. Standard library only, so the supervisor starts in a moment.
"""
from __future__ import annotations

import atexit
import os
import re
import signal
import subprocess
import sys
import threading
import time

SERVER_LOG = "vllm_server.log"
SERVER_TIMEOUT_S = 1200.0           # how long the trainer waits for the server to load (model load + CUDA graphs)
# TQDM_DISABLE too: `trl vllm-serve` calls LLM.generate with its progress bar on, and with tqdm disabled vLLM 0.26
# divides the token count by an elapsed time of 0 (offline_utils._run_engine). The worker dies of the
# ZeroDivisionError after generating, waits forever on its EngineCore child, and the server never answers (allium-0
# and david-a40-0, 2026-10-20).
_STRIP_ENV = ("MASTER_ADDR", "MASTER_PORT", "RANK", "LOCAL_RANK", "WORLD_SIZE", "PYTORCH_CUDA_ALLOC_CONF",
              "TQDM_DISABLE")
# A vLLM engine that has hit one of these is dead, but `trl vllm-serve` keeps answering HTTP, and a trainer in the
# middle of a weight sync would wait in NCCL forever (seen on allium-0: the server ran out of memory receiving the
# fp32 embedding matrix).
FATAL = re.compile(r"\(EngineCore[^)]*\) ERROR|EngineDeadError|Engine core initialization failed")


def split_devices(visible: str | None, k: int) -> tuple[str, str]:
    """(trainer device, server device) as CUDA_VISIBLE_DEVICES entries, for --vllm-server-gpu K. K indexes the GPUs
    this process may use (CUDA_VISIBLE_DEVICES when set, else the physical GPUs); the trainer takes index 0. Each
    process is then shown only its own GPU: a trainer that saw two would be wrapped in DataParallel by the HF Trainer."""
    devs = [d.strip() for d in visible.split(",") if d.strip()] if visible else None
    if devs is None:
        return "0", str(k)
    if k >= len(devs):
        raise RuntimeError(f"--vllm-server-gpu {k}, but CUDA_VISIBLE_DEVICES={visible!r} lists {len(devs)} GPU(s)")
    return devs[0], devs[k]


def vllm_serve_cmd(model_path: str, t: dict, port: int, mem: float) -> list[str]:
    """`trl vllm-serve` with the engine settings colocated generation uses (bf16 from the model config, same model
    implementation and max model length), listening on localhost only."""
    cmd = [sys.executable, "-m", "trl.scripts.vllm_serve", "--model", model_path, "--host", "127.0.0.1",
           "--port", str(port), "--gpu-memory-utilization", str(mem), "--vllm-model-impl", t["vllm_impl"] or "vllm"]
    if t["vllm_max_model_len"]:
        cmd += ["--max-model-len", str(t["vllm_max_model_len"])]
    return cmd


def index_nccl_device() -> None:
    """TRL hands vLLM's PyNcclCommunicator `accelerator.device`, which in a single-process trainer is `cuda` with no
    index; the communicator compares devices exactly and then refuses the trainer's tensors (on `cuda:0`). Give it the
    index. Idempotent."""
    import torch
    from trl.generation.vllm_client import VLLMClient
    orig = VLLMClient.init_communicator
    if getattr(orig, "_hib_indexed", False):
        return

    def init_communicator(self, device=0):
        if isinstance(device, torch.device) and device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        return orig(self, device=device)

    init_communicator._hib_indexed = True
    VLLMClient.init_communicator = init_communicator


class VLLMServer:
    """One server for one launch, on its own GPU, logging to <out>/vllm_server.log. A watchdog ends the trainer if the
    server dies first: a trainer waiting on a dead server would block in NCCL or wait out the connection timeout."""

    def __init__(self, cmd: list[str], device: str, out: str, env: dict):
        self.cmd, self.device, self.log_path = cmd, device, os.path.join(out, SERVER_LOG)
        self._stopping = False
        env = {k: v for k, v in env.items() if k not in _STRIP_ENV}
        env["CUDA_VISIBLE_DEVICES"] = device
        self._log = open(self.log_path, "a")
        self._log_start = self._log.tell()
        self._log.write(f"\n# {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {' '.join(cmd)} "
                        f"(CUDA_VISIBLE_DEVICES={device})\n")
        self._log.flush()
        self.proc = subprocess.Popen([sys.executable, "-m", __name__, *cmd], env=env, stdout=self._log,
                                     stderr=subprocess.STDOUT, preexec_fn=_die_with_parent)
        atexit.register(self.stop)
        threading.Thread(target=self._watch, daemon=True).start()

    def _watch(self):
        pos, why = self._log_start, None
        while why is None:
            try:
                rc = self.proc.wait(timeout=5)
                why = f"exited (rc {rc})"
            except subprocess.TimeoutExpired:
                pass
            try:
                with open(self.log_path) as f:
                    f.seek(pos)
                    new = f.read()
                    pos = f.tell()
            except OSError:
                new = ""
            hit = FATAL.search(new)
            if hit and why is None:
                why = f"reported a dead engine ({hit.group(0)!r})"
        if self._stopping:
            return
        try:
            tail = "".join(open(self.log_path).readlines()[-40:])
        except OSError:
            tail = ""
        print(f"[hib] the vLLM server {why} before the trainer finished; end of {self.log_path}:\n{tail}",
              file=sys.stderr, flush=True)
        os._exit(3)                                     # the supervisor sees us die and clears the server's GPU

    def stop(self, client=None):
        if self._stopping:
            return
        self._stopping = True
        if client is not None:                          # close the NCCL group while the server can still answer
            try:
                client.close_communicator()
                atexit.unregister(client.close_communicator)
            except Exception:  # noqa: BLE001
                pass
        if self.proc.poll() is None:
            self.proc.terminate()                       # the supervisor sweeps the server's process group
            try:
                self.proc.wait(timeout=90)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self._log.close()


def _die_with_parent():
    """In a child process: get SIGTERM when the parent dies, so a killed job does not strand a GPU."""
    try:
        import ctypes
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)    # PR_SET_PDEATHSIG
    except OSError:
        pass                                            # not Linux


def _kill_group(proc: subprocess.Popen, grace: float = 30.0) -> None:
    """SIGTERM the process group led by proc, then SIGKILL whatever is left after `grace` seconds. The group is ours,
    so EPERM (macOS reports it for a group left with only exiting processes) means there is nothing left to stop."""
    gone = (ProcessLookupError, PermissionError)
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except gone:
        return
    deadline = time.time() + grace
    while time.time() < deadline:
        proc.poll()                                     # reap the leader, so a zombie does not keep the group alive
        try:
            os.killpg(proc.pid, 0)
        except gone:
            return
        time.sleep(0.5)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except gone:
        pass


def supervise(cmd: list[str]) -> int:
    """Run cmd in its own process group; when it exits, or when told to stop, kill the whole group."""
    started: list[subprocess.Popen] = []

    def stop(signum, frame):
        if started:
            _kill_group(started[0])
        os._exit(128 + signum)

    signal.signal(signal.SIGTERM, stop)                 # before the server exists, so no signal can orphan it
    signal.signal(signal.SIGINT, stop)
    proc = subprocess.Popen(cmd, start_new_session=True)
    started.append(proc)
    rc = proc.wait()
    _kill_group(proc)                                   # stragglers: EngineCore outlives the server on shutdown
    return rc


if __name__ == "__main__":
    sys.exit(supervise(sys.argv[1:]))
