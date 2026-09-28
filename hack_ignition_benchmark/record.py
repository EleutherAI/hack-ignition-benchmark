"""What a run leaves behind: the run record.

    run_card.json       the run's identity: the recipe (verbatim + sha256), any overrides, the resolved training
                        settings, the environment, every input file with its sha256, the library version and commit,
                        a content hash of the installed tree, the generation stack, and one entry per launch.
    rollouts.jsonl.gz   one line per graded completion: step, launch, call, idx, item, text, grades, reward, mode.
                        Written as one gzip member per reward call, so a killed run leaves a readable file.
    events.jsonl        structured per-step lines (hack rate, per-item counts, injections, monitor stats,
                        optimizer resets). `extract` reads this and the rollouts; it never parses stdout.
    checkpoint-*/       every checkpoint is kept; optimizer state is pruned per [checkpoint].keep_optimizer.

Nothing here may kill a run: write failures degrade to a `[record]` warning on stderr.
"""
from __future__ import annotations

import glob
import gzip
import hashlib
import json
import os
import re
import sys
import threading
import time

RECORD_VERSION = "hib-1"      # v0.2a was the private repo's format; this adds events.jsonl and the recipe block
ROLLOUTS = "rollouts.jsonl.gz"
RUN_CARD = "run_card.json"
EVENTS = "events.jsonl"


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def warn(msg: str) -> None:
    print(f"[record] {msg}", file=sys.stderr, flush=True)


class RolloutWriter:
    def __init__(self, out_dir: str, truncate: bool = False, enabled: bool = True):
        self.path = os.path.join(out_dir, ROLLOUTS)
        self.enabled = enabled
        self.n_lines = self.n_calls = 0
        self._lock = threading.Lock()
        if enabled and truncate and os.path.exists(self.path):
            os.remove(self.path)

    def write(self, records) -> int:
        if not self.enabled:
            return 0
        recs = list(records)
        try:
            with self._lock, gzip.open(self.path, "ab", compresslevel=6) as f:
                for r in recs:
                    r.setdefault("record", RECORD_VERSION)
                    f.write((json.dumps(r, ensure_ascii=False) + "\n").encode("utf-8"))
            self.n_lines += len(recs)
            self.n_calls += 1
            return len(recs)
        except Exception as e:  # noqa: BLE001
            warn(f"rollout write failed at call {self.n_calls} ({type(e).__name__}: {e})")
            return 0


def read_rollouts(path: str):
    """Yield rollout lines. A truncated final gzip member (killed run) ends the stream with a warning."""
    try:
        with gzip.open(path, "rb") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    except EOFError:
        warn(f"{path}: truncated final gzip member; earlier records are intact")


class EventLog:
    """events.jsonl: one JSON object per line, {"event": <kind>, "step": n, ...}. Appends across launches."""

    def __init__(self, out_dir: str, truncate: bool = False, echo: bool = True):
        self.path = os.path.join(out_dir, EVENTS)
        self.echo = echo
        self._lock = threading.Lock()
        if truncate and os.path.exists(self.path):
            os.remove(self.path)

    def emit(self, event: str, _echo: bool | None = None, **fields) -> None:
        rec = {"event": event, **fields}
        try:
            with self._lock, open(self.path, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as e:  # noqa: BLE001
            warn(f"event write failed ({type(e).__name__}: {e})")
        if self.echo if _echo is None else _echo:
            short = " ".join(f"{k}={v}" for k, v in fields.items() if not isinstance(v, (dict, list)))
            print(f"[{event}] {short}", flush=True)


def read_events(path: str):
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_run_card(out_dir: str, card: dict) -> dict:
    """Merge `card` over any existing run_card.json (launch-time fields survive the end-of-run update)."""
    path = os.path.join(out_dir, RUN_CARD)
    try:
        old = {}
        if os.path.exists(path):
            with open(path) as f:
                old = json.load(f)
        merged = {**old, **card, "record_version": RECORD_VERSION, "written_utc": utc()}
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(merged, f, indent=2, default=str)
        os.replace(tmp, path)
        return merged
    except Exception as e:  # noqa: BLE001
        warn(f"run card not written ({type(e).__name__}: {e})")
        return card


def read_run_card(run_dir: str) -> dict:
    with open(os.path.join(run_dir, RUN_CARD)) as f:
        return json.load(f)


_CKPT = re.compile(r"checkpoint-(\d+)$")
PRUNABLE = ("optimizer.pt", "scheduler.pt", "scaler.pt")


def checkpoints(out_dir: str) -> list[tuple[int, str]]:
    ck = []
    for d in glob.glob(os.path.join(out_dir, "checkpoint-*")):
        m = _CKPT.search(d)
        if m and os.path.isdir(d):
            ck.append((int(m.group(1)), d))
    return sorted(ck)


def prune_optimizer_states(out_dir: str, keep: str = "latest") -> dict:
    """keep = all | latest | none. Adapter / model weights, trainer_state.json and RNG state are never touched, so
    every checkpoint still loads as a policy; only resuming from a pruned one is no longer possible."""
    ck = checkpoints(out_dir)
    if keep == "all":
        return {"pruned": [], "bytes_freed": 0}
    keep_dirs = {ck[-1][1]} if (keep == "latest" and ck) else set()
    freed, pruned = 0, []
    for _, d in ck:
        if d in keep_dirs:
            continue
        for name in PRUNABLE:
            p = os.path.join(d, name)
            if os.path.exists(p):
                try:
                    freed += os.path.getsize(p)
                    os.remove(p)
                    pruned.append(p)
                except Exception as e:  # noqa: BLE001
                    warn(f"could not remove {p} ({type(e).__name__}: {e})")
    return {"pruned": pruned, "bytes_freed": freed}


# --- provenance of the code that ran ---------------------------------------------------------------------------------

def tree_hash(root: str | None = None) -> str:
    """sha256 over (relative path, content) of every .py / .txt / .toml / .json file of the installed package.
    The deployment layer compares this with the hash of the commit it fetched: a mismatch means the code that ran
    is not the commit the card names (someone edited the deployed tree)."""
    import hack_ignition_benchmark as pkg
    root = root or os.path.dirname(os.path.abspath(pkg.__file__))
    h = hashlib.sha256()
    for dp, dns, fns in os.walk(root):
        dns[:] = sorted(d for d in dns if d != "__pycache__")
        for fn in sorted(fns):
            if fn.endswith((".py", ".txt", ".toml", ".json")):
                p = os.path.join(dp, fn)
                h.update(os.path.relpath(p, root).encode() + b"\0")
                with open(p, "rb") as f:
                    h.update(f.read())
    return h.hexdigest()


def code_provenance() -> dict:
    """{library_version, commit, commit_source, tree_sha256}. The commit comes from git when the package is a git
    checkout, else from HIB_COMMIT (set by whoever deployed the tree)."""
    import subprocess
    import hack_ignition_benchmark as pkg
    root = os.path.dirname(os.path.dirname(os.path.abspath(pkg.__file__)))
    commit, source, dirty = None, None, None
    try:
        r = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        if r.returncode == 0 and r.stdout.strip():
            commit, source = r.stdout.strip(), "git"
            s = subprocess.run(["git", "-C", root, "status", "--porcelain", "--", "hack_ignition_benchmark"],
                               capture_output=True, text=True, timeout=10)
            dirty = bool(s.stdout.strip())
    except Exception:  # noqa: BLE001
        pass
    if commit is None and os.environ.get("HIB_COMMIT"):
        commit, source = os.environ["HIB_COMMIT"].strip(), "env"
    return {"library_version": pkg.__version__, "commit": commit, "commit_source": source, "dirty": dirty,
            "tree_sha256": tree_hash()}
