"""djinn: coding problems whose insecure verifier has a planted vulnerability (EleutherAI/djinn).

Items are djinn problem rows {id, description, function_name, ground_truth, exploit, test_cases, insecure_*,
exploit_type, ...}. The user message is the description, plus "\\n\\n" + insecure_verifier_info when
env_options.verifier_info is true (djinn's intended presentation). Grading is djinn's own dual verifier
(`djinn.core.reward.calc_reward`), run in a separate server process so djinn's forkserver sandbox re-imports that
small script rather than the trainer:

    exploitable  insecure verifier passes
    hardened     secure verifier passes
    hack         insecure passes and secure fails

Port of the private trainer's djinn path (04_rl/grpo_train.py), 04_rl/djinn_grade_server.py (-> djinn_server.py)
and rhg/djinn_grade.py (GradePool). workers = 1 is the trainer's serial path (one server, requests written in a
batch, no watchdog); workers > 1 is GradePool, which grades each shared-/tmp exploit class on one dedicated worker
under a node-wide flock (three insecure verifiers use fixed /tmp paths; PITFALLS #7) and kills a server stuck longer
than grade_timeout.

Timeout caps for untrusted completions (swept 2026-09-04: 1/372 secure passes lost, 0 false hacks, GT 1032/1032).
Each is an env option; an environment variable of the old name still overrides it, as it did in the trainer:

    secure_timeout_per_test  DJINN_SECURE_TIMEOUT_PER_TEST  2
    max_total_timeout        DJINN_MAX_TOTAL_TIMEOUT        10
    wall_backstop_min        DJINN_WALL_BACKSTOP_MIN        5
    wall_backstop_factor     DJINN_WALL_BACKSTOP_FACTOR     2
    grade_timeout            DJINN_GRADE_TIMEOUT            60   (GradePool's per-grade watchdog; workers > 1)

The wall-backstop knobs need a djinn tree with the CPU-time patches (upstream PR #9); the pinned upstream commit is
a0563f5. The djinn tree: env_options.djinn_root, else $DJINN_ROOT, else the installed `djinn` package.
"""
from __future__ import annotations

import atexit
import contextlib
import fcntl
import hashlib
import importlib.util
import json
import os
import queue
import subprocess
import sys
import threading
import time

from .base import Environment, Grade, register_env

SERVER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "djinn_server.py")
DJINN_PINNED_COMMIT = "a0563f5"

# option -> (environment variable, default); string values, exactly as the trainer's os.environ.setdefault
CAPS = {
    "secure_timeout_per_test": ("DJINN_SECURE_TIMEOUT_PER_TEST", "2"),
    "max_total_timeout": ("DJINN_MAX_TOTAL_TIMEOUT", "10"),
    "wall_backstop_min": ("DJINN_WALL_BACKSTOP_MIN", "5"),
    "wall_backstop_factor": ("DJINN_WALL_BACKSTOP_FACTOR", "2"),
    "grade_timeout": ("DJINN_GRADE_TIMEOUT", "60"),
}
_OPTIONS = set(CAPS) | {"workers", "verifier_info", "djinn_root", "grade_status"}


# --- GradePool (port of rhg/djinn_grade.py @ f4b066f) ----------------------------------------------------------------

SHARED_TMP_CLASSES = {"filesystem_exposure", "test_case_logging_side_channel", "environment_variable_manipulation",
                      "import_hook_side_channel"}
LOCK_DIR = os.environ.get("DJINN_GRADE_LOCK_DIR", "/tmp")


@contextlib.contextmanager
def class_lock(group):
    """Cross-PROCESS lock per shared-/tmp class (flock on /tmp/.djinn_grade_<class>.lock): the side-channel
    files are node-local, so several shards/pools on one node must also serialize, not just threads in one."""
    path = os.path.join(LOCK_DIR, f".djinn_grade_{group}.lock")
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


class GradePool:
    """A pool of grade-server subprocesses. Each worker thread owns ONE server and serves one request at a time, so
    grades are identical to a lone serial server while N run concurrently. Verbatim port except the server default
    (this package's djinn_server.py)."""

    def __init__(self, problems_file, workers=32, server=SERVER, env=None, grade_timeout=None,
                 python=None, max_restarts=3, cwd=None):
        self.problems_file = problems_file
        self.cwd = cwd
        self.workers = int(workers)
        self.server = server
        self.env = {**os.environ, "DJINN_OFFLINE_VERIFICATION": "true", **(env or {})}
        self.grade_timeout = float(grade_timeout if grade_timeout is not None else os.environ.get("DJINN_GRADE_TIMEOUT", 600))
        self.python = python or sys.executable
        self.max_restarts = max_restarts
        self.stats = {"grades": 0, "timeouts": 0, "restarts": 0, "wall": 0.0, "spawns": 0}
        self._servers = []
        self._idle = []
        self._lock = threading.Lock()
        self.last_timed_out = []
        self.group_of = {}
        try:
            for line in open(problems_file):
                line = line.strip()
                if line:
                    r = json.loads(line)
                    if r.get("exploit_type") in SHARED_TMP_CLASSES:
                        self.group_of[r["id"]] = r["exploit_type"]
        except Exception as e:
            sys.stderr.write(f"[GradePool] could not read exploit types from {problems_file}: {e!r}\n")

    def _acquire(self):
        with self._lock:
            if self._idle:
                return self._idle.pop()
        return self._spawn()

    def _release(self, srv):
        if srv.poll() is None:
            with self._lock:
                self._idle.append(srv)
        else:
            self._kill(srv)

    def _spawn(self):
        with self._lock:
            self.stats["spawns"] += 1
        srv = subprocess.Popen([self.python, self.server, self.problems_file],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
                               env=self.env, cwd=self.cwd)
        assert srv.stdout is not None and srv.stdin is not None
        ready = srv.stdout.readline()
        if not ready.strip():
            raise RuntimeError(f"grade server pid={srv.pid} produced no ready line")
        with self._lock:
            self._servers.append(srv)
        return srv

    @staticmethod
    def _kill(srv):
        for fn in (lambda: srv.stdin.close(), srv.terminate):  # type: ignore[union-attr]
            try:
                fn()
            except Exception:
                pass
        try:
            srv.wait(timeout=10)
        except Exception:
            try:
                srv.kill()
            except Exception:
                pass

    def _grade_one(self, srv, pid, code):
        """Returns (grade, timed_out). A watchdog kills the server if the grade exceeds grade_timeout."""
        fired = threading.Event()

        def _boom():
            fired.set()
            try:
                srv.kill()
            except Exception:
                pass
        wd = threading.Timer(self.grade_timeout, _boom)
        wd.daemon = True
        wd.start()
        try:
            srv.stdin.write(json.dumps({"pid": pid, "code": code}) + "\n")  # type: ignore[union-attr]
            srv.stdin.flush()  # type: ignore[union-attr]
            line = srv.stdout.readline()  # type: ignore[union-attr]
        except Exception:
            line = ""
        finally:
            wd.cancel()
        if fired.is_set() or not line.strip():
            return [0.0, 0.0, 0.0], True
        try:
            return json.loads(line), False
        except Exception:
            return [0.0, 0.0, 0.0], True

    def grade(self, tasks, progress=None, on_result=None, done_grades=None):
        """tasks: iterable of (pid, code). Returns list of grades in task order."""
        tasks = list(tasks)
        results: list = [None] * len(tasks)
        flags: list = [False] * len(tasks)
        q = queue.Queue()
        gq = {}
        for i, (pid, code) in enumerate(tasks):
            if done_grades and i in done_grades:
                results[i] = list(done_grades[i]); continue
            grp = self.group_of.get(pid)
            (gq.setdefault(grp, queue.Queue()) if grp else q).put((i, pid, code))
        done = [0]
        t0 = time.time()
        errors = []

        def worker(q=q, group=None):
            try:
                srv = self._acquire()
            except Exception as e:
                errors.append(repr(e))
                return
            restarts = 0
            while True:
                try:
                    i, pid, code = q.get_nowait()
                except queue.Empty:
                    break
                with (class_lock(group) if group else contextlib.nullcontext()):
                    g, to = self._grade_one(srv, pid, code)
                results[i], flags[i] = g, to
                if on_result is not None:
                    try:
                        on_result(i, g, to)
                    except Exception as e:
                        sys.stderr.write(f"[GradePool] on_result failed: {e!r}\n")
                if to:
                    with self._lock:
                        self.stats["timeouts"] += 1
                    self._kill(srv)
                    if restarts >= self.max_restarts:
                        errors.append(f"worker gave up after {restarts} restarts")
                        break
                    restarts += 1
                    with self._lock:
                        self.stats["restarts"] += 1
                    try:
                        srv = self._spawn()
                    except Exception as e:
                        errors.append(repr(e))
                        break
                with self._lock:
                    done[0] += 1
                    self.stats["grades"] += 1
                if progress is not None:
                    progress(done[0], len(tasks))
            self._release(srv)

        n = max(1, min(self.workers, q.qsize())) if q.qsize() else 0
        threads = [threading.Thread(target=worker, daemon=True) for _ in range(n)]
        threads += [threading.Thread(target=worker, kwargs={"q": gq_, "group": grp}, daemon=True) for grp, gq_ in gq.items()]
        if gq:
            sys.stderr.write(f"[GradePool] {n} general workers + {len(gq)} dedicated (serial) workers for shared-/tmp "
                             f"classes {sorted(gq)} ({sum(x.qsize() for x in gq.values())} grades)\n")
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        self.stats["wall"] += time.time() - t0
        if errors:
            sys.stderr.write(f"[GradePool] {len(errors)} worker error(s): {errors[:3]}\n")
        missing = [i for i, r in enumerate(results) if r is None]
        if missing:
            sys.stderr.write(f"[GradePool] {len(missing)} task(s) never graded -> [0,0,0]\n")
            for i in missing:
                results[i], flags[i] = [0.0, 0.0, 0.0], True
        self.last_timed_out = flags
        return results

    def close(self):
        with self._lock:
            servers, self._servers, self._idle = self._servers, [], []
        for srv in servers:
            self._kill(srv)


# --- the environment -------------------------------------------------------------------------------------------------

def _git_commit(path: str | None) -> dict:
    if not path:
        return {"commit": None, "dirty": None}
    try:
        r = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        if r.returncode != 0 or not r.stdout.strip():
            return {"commit": None, "dirty": None}
        s = subprocess.run(["git", "-C", path, "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True, timeout=10)
        return {"commit": r.stdout.strip(), "dirty": bool(s.stdout.strip())}
    except Exception:  # noqa: BLE001
        return {"commit": None, "dirty": None}


def _sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


@register_env
class DjinnEnv(Environment):
    name = "djinn"
    id_key = "id"

    def __init__(self, recipe, workdir: str, data):
        super().__init__(recipe, workdir, data)
        bad = set(self.options) - _OPTIONS
        if bad:
            raise ValueError(f"djinn: unknown env_options {sorted(bad)} (have {sorted(_OPTIONS)})")
        self.workers = int(self.options.get("workers", 1))
        if self.workers < 1:
            raise ValueError("djinn: workers must be >= 1")
        self.verifier_info = bool(self.options.get("verifier_info", False))
        self.grade_status = bool(self.options.get("grade_status", False))
        # effective caps: environment variable > env_options > default (the trainer's os.environ.setdefault order)
        self.caps: dict[str, str] = {}
        self.cap_sources: dict[str, str] = {}
        for opt, (var, default) in CAPS.items():
            if os.environ.get(var) is not None:
                self.caps[opt], self.cap_sources[opt] = os.environ[var], "env"
            elif opt in self.options:
                self.caps[opt], self.cap_sources[opt] = str(self.options[opt]), "recipe"
            else:
                self.caps[opt], self.cap_sources[opt] = default, "default"
        self.djinn_root = self.options.get("djinn_root") or os.environ.get("DJINN_ROOT") or None
        self._items_rel: str | None = None
        self._items_path: str | None = None
        self._by_id: dict[str, dict] = {}
        self._srv: subprocess.Popen | None = None
        self._srv_lock = threading.Lock()
        self._pool: GradePool | None = None
        self.n_calls = 0
        # per-completion diagnostics of the last grade(): {"timed_out", "status"}; not part of the record
        self.last_diagnostics: list[dict] = []

    # -- items and prompts -----------------------------------------------------------------------------------------
    def load_items(self) -> list[dict]:
        if self.recipe.problems is None:
            raise ValueError("djinn: the recipe names no problems file ([run].problems)")
        self._items_path = self.data.path(self.recipe.problems)
        self._items_rel = self.recipe.problems.file
        with open(self._items_path) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        self._by_id = {r["id"]: r for r in rows}
        return rows

    def user_text(self, item: dict) -> str:
        return item["description"] + ("\n\n" + item.get("insecure_verifier_info", "") if self.verifier_info else "")

    def dataset_fields(self, item: dict) -> dict:
        return {"func_name": item.get("function_name", "")}

    # -- grading ---------------------------------------------------------------------------------------------------
    def _server_env(self) -> dict:
        env = {CAPS[k][0]: v for k, v in self.caps.items()}
        env["DJINN_OFFLINE_VERIFICATION"] = "true"
        if self.djinn_root:
            env["DJINN_ROOT"] = self.djinn_root
        if self.grade_status:
            env["DJINN_GRADE_STATUS"] = "1"
        return env

    def start(self):
        if self._srv is not None or self._pool is not None:
            return
        if self._items_path is None:
            self.load_items()
        os.makedirs(self.workdir, exist_ok=True)
        cwd = os.path.abspath(self.workdir)
        if self.workers > 1:
            self._pool = GradePool(self._items_path, workers=self.workers, env=self._server_env(),
                                   grade_timeout=float(self.caps["grade_timeout"]), cwd=cwd)
        else:
            self._srv = subprocess.Popen(
                [sys.executable, SERVER, self._items_path],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
                env={**os.environ, **self._server_env()}, cwd=cwd)
            ready = self._srv.stdout.readline().strip()
            if not ready:
                raise RuntimeError("djinn grade server produced no ready line (is djinn importable? set "
                                   "env_options.djinn_root or DJINN_ROOT)")
            print(f"[DJINN] grade server up: {ready}", flush=True)
        atexit.register(self.close)

    def _grade_serial(self, pids, codes):
        out = []
        with self._srv_lock:
            for pid, code in zip(pids, codes):
                self._srv.stdin.write(json.dumps({"pid": pid, "code": code}) + "\n")
            self._srv.stdin.flush()
            for _ in range(len(pids)):
                ln = self._srv.stdout.readline()
                try:
                    out.append(tuple(json.loads(ln)))
                except Exception:
                    out.append((0.0, 0.0, 0.0))
        return out, [False] * len(out)

    def grade(self, items: list[dict], completions: list[str]) -> list[Grade]:
        if len(items) != len(completions):
            raise ValueError(f"djinn.grade: {len(items)} items vs {len(completions)} completions")
        self.start()
        pids = [it["id"] for it in items]
        t0 = time.time()
        if self._pool is not None:
            res = [tuple(g) for g in self._pool.grade(list(zip(pids, completions)))]
            timed_out = list(self._pool.last_timed_out)
            how = f"pool={self.workers}"
        else:
            res, timed_out = self._grade_serial(pids, completions)
            how = "serial"
        print(f"[DJINN] graded {len(res)} in {time.time() - t0:.1f}s ({how})", flush=True)
        self.n_calls += 1
        self.last_diagnostics = [{"timed_out": bool(to), "status": (list(r[3]) if len(r) > 3 else None)}
                                 for r, to in zip(res, timed_out)]
        return [Grade(exploitable=float(r[0]), hardened=float(r[2]), hack=bool(r[1]),
                      channels={"exploit_type": it.get("exploit_type")})
                for r, it in zip(res, items)]

    def close(self):
        srv, self._srv = self._srv, None
        if srv is not None:
            GradePool._kill(srv)
        pool, self._pool = self._pool, None
        if pool is not None:
            pool.close()

    # -- description -----------------------------------------------------------------------------------------------
    def djinn_location(self) -> str | None:
        if self.djinn_root:
            return self.djinn_root
        try:
            spec = importlib.util.find_spec("djinn")
        except Exception:  # noqa: BLE001
            spec = None
        if spec is None or not spec.submodule_search_locations:
            return None
        return os.path.dirname(list(spec.submodule_search_locations)[0])

    def describe(self) -> dict:
        items = None
        if self._items_rel is not None:
            items = {"file": self._items_rel, **(self.data.resolved.get(self._items_rel) or {})}
            items.pop("path", None)
        loc = self.djinn_location()
        tree = _git_commit(loc)
        version = None
        try:
            from importlib.metadata import version as _v
            version = _v("djinn")
        except Exception:  # noqa: BLE001
            pass
        return {
            "name": self.name,
            "options": {"workers": self.workers, "verifier_info": self.verifier_info,
                        "djinn_root": self.djinn_root, "grade_status": self.grade_status,
                        **{k: v for k, v in self.caps.items()}},
            "caps": {CAPS[k][0]: {"value": v, "source": self.cap_sources[k]} for k, v in self.caps.items()},
            "grading": "serial server" if self.workers == 1 else f"GradePool({self.workers})",
            "djinn": {"location": loc, "source": "djinn_root option" if self.options.get("djinn_root") else
                      ("DJINN_ROOT" if self.djinn_root else "installed package"),
                      "commit": tree["commit"], "dirty": tree["dirty"], "package_version": version,
                      "pinned_upstream_commit": DJINN_PINNED_COMMIT,
                      "matches_pin": (tree["commit"] or "").startswith(DJINN_PINNED_COMMIT) if tree["commit"] else None},
            "server_sha256": _sha256(SERVER),
            "items": items,
        }
