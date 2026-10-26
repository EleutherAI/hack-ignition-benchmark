"""Training-time isolation for model-written code that was trained to tamper with its grader (the CATCH environment).

The cluster's pods run the trainer as root in a container whose seccomp profile refuses unshare(): no user, network,
PID or mount namespaces, so no bubblewrap and no container per grade. Root does keep the container runtime's default
capabilities, among them CAP_SYS_CHROOT, CAP_SETUID / CAP_SETGID and CAP_MKNOD. A jail built from those:

    file system   a chroot holding only an interpreter, its standard library, an allowlisted set of packages, the
                  shared libraries they load and /dev/{null,zero,random,urandom}. The trainer's files, mounts,
                  credentials and item files (which hold every test's expected output) are outside it.
    identity      one uid and gid per grading worker, no supplementary groups, no capabilities (setuid from root drops
                  them), no_new_privs. Another worker's files and processes belong to another uid.
    network       none: a seccomp filter refuses socket() (socketpair() still works), io_uring (which opens sockets on
                  5.19+), ptrace, process_vm_readv/writev, bpf, perf_event_open, userfaultfd and the keyring calls
    limits        rlimits on address space, CPU, processes, file size, open files and core dumps
    lifetime      before and after every run each process of the uid is killed (kill(-1) as that uid) and the uid's
                  work area is emptied, so nothing one run writes is there for the next. The only places a jailed
                  process can write are its own run directories under /work/<uid>/.

The parent never follows a path inside a run directory (a jailed process can plant symlinks whose absolute targets
resolve outside the chroot for the parent): it reads results with openat(O_NOFOLLOW) from directories it created.

Weaker than a container per grade: the disk is shared (bounded by RLIMIT_FSIZE and each run's wall time), and the
kernel is shared as in any container. docs/CATCH_PORT.md has the design and its limits.

Where the jail cannot be built (not Linux, or not root), `Jail.trusted()` runs the same commands with rlimits only.
It is allowed only where sandbox.sandbox_status() allows executing model code: inside the sandbox container
(tools/sandbox_regrade.sh, which is itself the isolation) or with HIB_UNSANDBOXED_OK naming this machine.
"""
from __future__ import annotations

import ctypes
import errno
import itertools
import os
import platform
import re
import resource
import select
import shutil
import signal
import stat
import subprocess
import sys
import sysconfig
import tempfile
import time

from .sandbox import sandbox_status

# Packages a jailed interpreter can import, besides the standard library. CATCH's tests need pytest and codetiming
# (`from codetiming._timer import Timer`). Optional ones are copied when the trainer's environment has them: pytest's
# dependencies that depend on its version (pygments since 8.4; exceptiongroup and tomli below Python 3.11), and numpy
# (completions import it).
REQUIRED_PACKAGES = ("pytest", "_pytest", "pluggy", "iniconfig", "packaging", "codetiming")
OPTIONAL_PACKAGES = ("pygments", "exceptiongroup", "tomli", "py", "numpy")
SITE_IN_JAIL = "/opt/hib-site"
WORK_IN_JAIL = "/work"
UID_BASE = 61000             # worker k runs as uid = gid = UID_BASE + k
_STDLIB_SKIP = {"site-packages", "dist-packages", "test", "idlelib", "tkinter", "turtledemo", "ensurepip",
                "lib2to3", "pydoc_data"}

DEFAULT_LIMITS = {
    "as_bytes": 8 * 1024 ** 3,   # CATCH's pytest cap (prlimit --as=8 GiB); applied to every jailed process
    "cpu_s": 600,                # backstop; wall-clock timeouts are what end a run
    "nproc": 128,                # processes of the uid at once
    "fsize_bytes": 256 * 1024 ** 2,
    "nofile": 1024,
}

# --- seccomp -------------------------------------------------------------------------------------------------------
_ARCH = {"x86_64": 0xC000003E, "aarch64": 0xC00000B7}
_DENIED = {
    "x86_64": {"socket": 41, "ptrace": 101, "process_vm_readv": 310, "process_vm_writev": 311, "bpf": 321,
               "perf_event_open": 298, "userfaultfd": 323, "add_key": 248, "request_key": 249, "keyctl": 250,
               "io_uring_setup": 425, "io_uring_enter": 426, "io_uring_register": 427},
    "aarch64": {"socket": 198, "ptrace": 117, "process_vm_readv": 270, "process_vm_writev": 271, "bpf": 280,
                "perf_event_open": 241, "userfaultfd": 282, "add_key": 217, "request_key": 218, "keyctl": 219,
                "io_uring_setup": 425, "io_uring_enter": 426, "io_uring_register": 427},
}
_RET_ALLOW, _RET_ERRNO, _RET_KILL = 0x7FFF0000, 0x00050000, 0x80000000
_LD_W_ABS, _JEQ_K, _JGE_K, _RET_K = 0x20, 0x15, 0x35, 0x06


class _SockFilter(ctypes.Structure):
    _fields_ = [("code", ctypes.c_ushort), ("jt", ctypes.c_ubyte), ("jf", ctypes.c_ubyte), ("k", ctypes.c_uint32)]


class _SockFprog(ctypes.Structure):
    _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.POINTER(_SockFilter))]


def seccomp_program(machine: str | None = None) -> list[tuple[int, int, int, int]]:
    """The BPF program: kill a foreign-architecture syscall, refuse the listed ones (and x32's) with EACCES, allow the
    rest. As (code, jt, jf, k) tuples, so it can be inspected and tested without installing it."""
    machine = machine or platform.machine()
    if machine not in _ARCH:
        raise JailError(f"no seccomp table for {machine!r}")
    head = [(_LD_W_ABS, 0, 0, 4), (_JEQ_K, 1, 0, _ARCH[machine]), (_RET_K, 0, 0, _RET_KILL), (_LD_W_ABS, 0, 0, 0)]
    checks = [(_JGE_K, None, 0, 0x40000000)] if machine == "x86_64" else []      # the x32 ABI's syscall numbers
    checks += [(_JEQ_K, None, 0, nr) for nr in sorted(_DENIED[machine].values())]
    deny = len(head) + len(checks) + 1                                             # after the RET ALLOW
    body = [(c, deny - (len(head) + i + 1), jf, k) for i, (c, _, jf, k) in enumerate(checks)]
    return head + body + [(_RET_K, 0, 0, _RET_ALLOW), (_RET_K, 0, 0, _RET_ERRNO | errno.EACCES)]


def _install_seccomp(fprog: _SockFprog):
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
    if libc.prctl(38, 1, 0, 0, 0) != 0:                                            # PR_SET_NO_NEW_PRIVS
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_NO_NEW_PRIVS)")
    if libc.prctl(22, 2, ctypes.addressof(fprog), 0, 0) != 0:                      # PR_SET_SECCOMP, MODE_FILTER
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_SECCOMP)")


def _set_child_subreaper():
    """Orphans of a jailed run re-parent to this worker, which reaps them (a pod's PID 1 may not)."""
    try:
        ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0)                     # PR_SET_CHILD_SUBREAPER
    except Exception:  # noqa: BLE001
        pass


def _reap():
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        if pid == 0:
            return


class JailError(RuntimeError):
    pass


# --- reading results without following a jailed process's symlinks --------------------------------------------------
def read_beneath(dir_fd: int, rel: str, limit: int) -> bytes | None:
    """Read `rel` under the open directory `dir_fd`, refusing symlinks at every component; None if absent, not a
    regular file, or larger than `limit` bytes."""
    parts = [p for p in rel.split("/") if p]
    fds = []
    try:
        cur = dir_fd
        for p in parts[:-1]:
            cur = os.open(p, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=cur)
            fds.append(cur)
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=cur)
        fds.append(fd)
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_size > limit:
            return None
        chunks, n = [], 0
        while n <= limit:
            b = os.read(fd, 1 << 20)
            if not b:
                break
            chunks.append(b)
            n += len(b)
        return b"".join(chunks) if n <= limit else None
    except OSError:
        return None
    finally:
        for f in fds:
            os.close(f)


# --- building the jail ----------------------------------------------------------------------------------------------
def _copy_file(src: str, dst: str):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(os.path.realpath(src), dst)


def _find_package(name: str) -> tuple[str, list[str]] | None:
    """(path, [dist-info dirs]) of an importable top-level module or package of the running interpreter."""
    import importlib.util
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return None
    if spec is None or spec.origin in (None, "built-in", "frozen"):
        return None
    path = list(spec.submodule_search_locations)[0] if spec.submodule_search_locations else spec.origin
    parent = os.path.dirname(path)
    norm = re.sub(r"[-_.]+", "_", name).lower()
    dists = [os.path.join(parent, d) for d in os.listdir(parent) if d.endswith(".dist-info")
             and re.sub(r"[-_.]+", "_", d[: -len(".dist-info")].rsplit("-", 1)[0]).lower() == norm]
    return path, dists


def _elf_files(root: str, skip=()) -> list[str]:
    """ELF files under `root`, leaving out its top-level directories named in `skip`."""
    out = []
    for d, dirs, fs in os.walk(root):
        if d == root:
            dirs[:] = [x for x in dirs if x not in skip]
        for f in fs:
            p = os.path.join(d, f)
            if (".so" in f or os.access(p, os.X_OK)) and not os.path.islink(p):
                try:
                    with open(p, "rb") as fh:
                        if fh.read(4) == b"\x7fELF":
                            out.append(p)
                except OSError:
                    pass
    return out


def _ldd_paths(files: list[str]) -> set[str]:
    """Absolute paths of the shared libraries (and the loader) these ELF files need, from ldd."""
    paths: set[str] = set()
    for i in range(0, len(files), 200):
        r = subprocess.run(["ldd", *files[i:i + 200]], capture_output=True, text=True, check=False)
        for line in r.stdout.splitlines():
            m = re.search(r"=>\s*(/\S+)\s+\(0x", line) or re.match(r"\s*(/\S+)\s+\(0x", line)
            if m:
                paths.add(m.group(1))
    return paths


def build_jail(root: str, packages=REQUIRED_PACKAGES, optional=OPTIONAL_PACKAGES, extra_files: dict | None = None
               ) -> dict:
    """Populate `root` (an empty directory) with what a jailed `python -m pytest` needs. Returns a description for
    the run card. The interpreter sits at its own absolute path inside the jail, so its prefix resolves unchanged.
    `extra_files` maps a module name to a source file copied into the jail's site directory."""
    if not shutil.which("ldd"):
        raise JailError("building the jail needs ldd (to find the interpreter's shared libraries)")
    exe = os.path.realpath(sys.executable)
    base = os.path.realpath(sys.base_prefix)
    stdlib = os.path.realpath(sysconfig.get_paths(vars={"base": base, "installed_base": base})["stdlib"])
    j = lambda p: os.path.join(root, p.lstrip("/"))                                  # noqa: E731
    _copy_file(exe, j(exe))
    os.chmod(j(exe), 0o755)
    shutil.copytree(stdlib, j(stdlib), symlinks=False, dirs_exist_ok=True,
                    ignore=lambda d, names: [n for n in names if os.path.realpath(d) == stdlib and n in _STDLIB_SKIP])
    site = j(SITE_IN_JAIL)
    os.makedirs(site, exist_ok=True)
    copied, missing = [], []
    sources = [exe, stdlib]                     # where the ELF files come from, for ldd (run on the originals, so a
    for name in tuple(packages) + tuple(optional):  # RUNPATH of $ORIGIN/... resolves to libraries that exist)
        found = _find_package(name)
        if found is None:
            if name in packages:
                missing.append(name)
            continue
        path, dists = found
        target = os.path.join(site, os.path.basename(path))
        if os.path.isdir(path):
            shutil.copytree(path, target, dirs_exist_ok=True)
            libs = path.rstrip("/") + ".libs"                                       # numpy.libs: bundled .so files
            if os.path.isdir(libs):
                shutil.copytree(libs, os.path.join(site, os.path.basename(libs)), dirs_exist_ok=True)
        else:
            shutil.copy2(path, target)
        sources.append(path)
        for d in dists:
            shutil.copytree(d, os.path.join(site, os.path.basename(d)), dirs_exist_ok=True)
        copied.append(name)
    if missing:
        raise JailError(f"the jail needs these packages in the trainer's environment: {missing} "
                        f"(pip install {' '.join(missing)})")
    for mod, src in (extra_files or {}).items():
        shutil.copy2(src, os.path.join(site, mod + ".py"))
    # The standard library as copied: without site-packages, whose extension modules (a conda base environment's torch
    # and CUDA wheels) would otherwise pull gigabytes of libraries into the jail.
    elves = [exe] + _elf_files(stdlib, _STDLIB_SKIP) + [f for s in sources[2:] if os.path.isdir(s) for f in _elf_files(s)]
    # A package's own libraries (numpy.libs) came with it and resolve through $ORIGIN in the copy; copying them again
    # at their original path would only duplicate them under the trainer's environment path.
    packaged = tuple(s.rstrip("/") + suffix for s in sources[2:] if os.path.isdir(s) for suffix in ("/", ".libs/"))
    for lib in sorted(_ldd_paths(elves)):
        if not lib.startswith(packaged) and not os.path.exists(j(lib)):
            _copy_file(lib, j(lib))
    if os.path.exists("/etc/ld.so.cache"):      # the loader finds libraries outside its default directories (e.g. a
        _copy_file("/etc/ld.so.cache", j("/etc/ld.so.cache"))   # libpython in /usr/local/lib) through the cache
    dev = j("/dev")
    os.makedirs(dev, exist_ok=True)
    for name, minor in (("null", 3), ("zero", 5), ("random", 8), ("urandom", 9)):
        try:
            os.mknod(os.path.join(dev, name), 0o666 | stat.S_IFCHR, os.makedev(1, minor))
            os.chmod(os.path.join(dev, name), 0o666)
        except OSError as e:
            raise JailError(f"cannot create /dev/{name} in the jail ({e}); pytest needs /dev/null") from None
    os.makedirs(j(WORK_IN_JAIL), exist_ok=True)
    # Everything root-owned and read-only to the jailed uids; /work is traversable but not listable.
    for d, dirs, files in os.walk(root):
        if d.startswith(j(WORK_IN_JAIL)) or d == dev:
            continue
        os.chmod(d, 0o755)
        for f in files:
            p = os.path.join(d, f)
            if not os.path.islink(p):
                os.chmod(p, os.stat(p).st_mode & 0o755)
    os.chmod(root, 0o755)
    os.chmod(j(WORK_IN_JAIL), 0o711)
    return {"python": exe, "python_version": platform.python_version(), "packages": copied,
            "machine": platform.machine(), "site": SITE_IN_JAIL}


# --- running --------------------------------------------------------------------------------------------------------
class Jail:
    """One grading worker's view of the jail: its uid, its work area, and how to run a command there.

    Use `Jail.create(...)` once in the parent (it builds the root), then `jail.for_worker(k)` in worker k. In trusted
    mode there is no root, no uid change and no seccomp: commands run as the caller with rlimits, in a temporary work
    area, and each run's process group is killed afterwards."""

    def __init__(self, mode: str, root: str | None, info: dict, limits: dict, uid: int | None = None,
                 work_base: str | None = None):
        self.mode, self.root, self.info, self.limits = mode, root, info, dict(limits)
        self.uid = uid
        self.work_base = work_base            # host path of this worker's work area
        self._fprog = None
        self._counter = itertools.count()
        if mode == "jail" and uid is not None:
            prog = seccomp_program()
            arr = (_SockFilter * len(prog))(*[_SockFilter(*ins) for ins in prog])
            self._fprog = _SockFprog(len(prog), arr)
            self._arr = arr                    # keep the filter's memory alive for preexec_fn

    # -- construction ----------------------------------------------------------------------------------------------
    @classmethod
    def create(cls, limits: dict | None = None, packages=REQUIRED_PACKAGES, optional=OPTIONAL_PACKAGES,
               extra_files: dict | None = None, parent_dir: str | None = None, mode: str = "auto") -> "Jail":
        """mode "auto": the jail where it can be built (Linux, root), else trusted where sandbox_status() allows it,
        else refuse. "jail" or "trusted" force one (trusted is still refused outside an allowed sandbox)."""
        limits = {**DEFAULT_LIMITS, **(limits or {})}
        can_jail = sys.platform.startswith("linux") and os.geteuid() == 0
        if mode == "auto":
            mode = "jail" if can_jail else "trusted"
        if mode == "jail":
            if not can_jail:
                raise JailError("the jail needs Linux and root (chroot, setuid); see hack_ignition_benchmark/jail.py")
            root = tempfile.mkdtemp(prefix="hib-jail-", dir=parent_dir or os.environ.get("HIB_JAIL_DIR") or "/tmp")
            info = build_jail(root, packages, optional, extra_files)
            return cls("jail", root, {**info, "mode": "jail"}, limits)
        if mode != "trusted":
            raise JailError(f"unknown jail mode {mode!r}")
        allowed, why = sandbox_status()
        if not allowed:
            raise JailError("refusing to execute model-written code outside a sandbox: this machine cannot build the "
                            "jail (it needs Linux and root), and " + why)
        top = tempfile.mkdtemp(prefix="hib-trusted-", dir=parent_dir)
        site = os.path.join(top, "site")
        os.mkdir(site)
        for mod, src in (extra_files or {}).items():
            shutil.copy2(src, os.path.join(site, mod + ".py"))
        return cls("trusted", top, {"mode": "trusted", "why": why, "python": sys.executable,
                                    "python_version": platform.python_version(), "site": site}, limits)

    def for_worker(self, k: int) -> "Jail":
        if self.mode == "jail":
            uid = UID_BASE + k
            base = os.path.join(self.root, WORK_IN_JAIL.lstrip("/"), str(uid))
            os.makedirs(base, exist_ok=True)
            os.chown(base, 0, 0)
            os.chmod(base, 0o711)              # root-owned: a jailed uid cannot rename or replace its run directories
            w = Jail("jail", self.root, self.info, self.limits, uid=uid, work_base=base)
            _set_child_subreaper()
            w.kill_all()
            return w
        base = os.path.join(self.root, f"work-{k}")
        os.makedirs(base, exist_ok=True)
        return Jail("trusted", self.root, self.info, self.limits, uid=None, work_base=base)

    def describe(self) -> dict:
        return {**self.info, "limits": dict(self.limits)}

    def destroy(self):
        """Remove the jail (or the trusted mode's temporary tree). Call once, from the process that created it."""
        if self.root:
            shutil.rmtree(self.root, ignore_errors=True)

    # -- run directories -------------------------------------------------------------------------------------------
    def new_dir(self, files: dict[str, str], tag: str) -> tuple[str, str]:
        """A fresh run directory holding `files` ({relative path: text}). Returns (host path, path as the jailed
        process sees it). Written by the parent before anything runs in it, so no planted link can redirect it."""
        name = f"{next(self._counter)}-{tag}"
        host = os.path.join(self.work_base, name)
        os.mkdir(host, 0o700)
        for rel, text in files.items():
            p = os.path.join(host, rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                f.write(text)
        os.makedirs(os.path.join(host, ".tmp"), exist_ok=True)
        if self.mode == "jail":
            for d, dirs, fs in os.walk(host):
                os.chown(d, self.uid, self.uid)
                for f in fs:
                    os.chown(os.path.join(d, f), self.uid, self.uid)
            return host, f"{WORK_IN_JAIL}/{self.uid}/{name}"
        return host, host

    def wipe(self):
        """Kill the uid's processes and empty its work area (between the runs of one grade, and after it)."""
        self.kill_all()
        for name in os.listdir(self.work_base):
            p = os.path.join(self.work_base, name)
            if os.path.isdir(p) and not os.path.islink(p):
                shutil.rmtree(p, ignore_errors=True)
            else:
                try:
                    os.unlink(p)
                except OSError:
                    pass

    def kill_all(self):
        if self.mode != "jail":
            return
        pid = os.fork()
        if pid == 0:
            try:
                os.setgroups([])
                os.setgid(self.uid)
                os.setuid(self.uid)
                os.kill(-1, signal.SIGKILL)         # every process of this uid but the caller
            except Exception:  # noqa: BLE001
                pass
            os._exit(0)
        os.waitpid(pid, 0)
        _reap()

    # -- processes -------------------------------------------------------------------------------------------------
    def env(self, cwd: str, extra: dict | None = None) -> dict:
        return {"PATH": "/usr/bin:/bin", "HOME": cwd, "TMPDIR": f"{cwd}/.tmp", "PYTHONPATH": self.info["site"],
                "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUTF8": "1", "LANG": "C.UTF-8", "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", **(extra or {})}

    def _preexec(self, cwd: str):
        lim = self.limits
        rl = [(resource.RLIMIT_CPU, lim["cpu_s"]), (resource.RLIMIT_FSIZE, lim["fsize_bytes"]),
              (resource.RLIMIT_NOFILE, lim["nofile"]), (resource.RLIMIT_CORE, 0)]
        darwin = sys.platform == "darwin"
        if not darwin:                                         # macOS refuses a finite RLIMIT_AS
            rl.append((resource.RLIMIT_AS, lim["as_bytes"]))
        if self.mode == "jail":
            root, uid, fprog = self.root, self.uid, self._fprog
            nproc = lim["nproc"]

            def f():
                os.chroot(root)
                os.chdir(cwd)
                os.setgroups([])
                os.setgid(uid)
                os.setuid(uid)
                for res, v in rl + [(resource.RLIMIT_NPROC, nproc)]:
                    resource.setrlimit(res, (v, v))
                _install_seccomp(fprog)
            return f

        def g():
            for res, v in rl:
                try:
                    resource.setrlimit(res, (v, v))
                except (ValueError, OSError):
                    pass
        return g

    def python(self) -> str:
        return self.info["python"]

    def popen(self, argv: list[str], cwd: str, env: dict | None = None, pass_fds=(), **kw) -> subprocess.Popen:
        """Start `argv` in the jail with working directory `cwd` (as the jailed process sees it)."""
        return subprocess.Popen(argv, cwd=None if self.mode == "jail" else cwd, env=env or self.env(cwd),
                                preexec_fn=self._preexec(cwd), start_new_session=True, close_fds=True,
                                pass_fds=pass_fds, **kw)

    def run(self, argv: list[str], cwd: str, timeout: float, env: dict | None = None,
            output_limit: int = 1 << 20) -> subprocess.CompletedProcess:
        """Run to completion with a wall-clock timeout (subprocess.TimeoutExpired, as subprocess.run raises it).
        stdout and stderr are captured as text and truncated to `output_limit` bytes each."""
        p = self.popen(argv, cwd, env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            out, err = _communicate(p, timeout, output_limit)
        finally:
            self._kill(p)               # also on success: a run's background processes do not outlive it
            p.stdout.close()
            p.stderr.close()
        return subprocess.CompletedProcess(argv, p.returncode, out.decode("utf-8", "replace"),
                                           err.decode("utf-8", "replace"))

    def _kill(self, p: subprocess.Popen):
        """Kill a run: its whole session, then (jail) every process of the uid."""
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        self.kill_all()


def _communicate(p: subprocess.Popen, timeout: float, limit: int) -> tuple[bytes, bytes]:
    """Popen.communicate with a cap on how much output is kept (a run that prints gigabytes cannot exhaust the
    grader's memory). Raises subprocess.TimeoutExpired at the deadline."""
    deadline = time.monotonic() + timeout
    bufs = {p.stdout.fileno(): bytearray(), p.stderr.fileno(): bytearray()}
    open_fds = set(bufs)
    while open_fds:
        left = deadline - time.monotonic()
        if left <= 0:
            raise subprocess.TimeoutExpired(p.args, timeout)
        ready, _, _ = select.select(list(open_fds), [], [], left)
        for fd in ready:
            b = os.read(fd, 1 << 16)
            if not b:
                open_fds.discard(fd)
                continue
            if len(bufs[fd]) < limit:
                bufs[fd] += b[: limit - len(bufs[fd])]
    left = deadline - time.monotonic()
    try:
        p.wait(timeout=max(left, 0.01))
    except subprocess.TimeoutExpired:
        raise subprocess.TimeoutExpired(p.args, timeout) from None
    return bytes(bufs[p.stdout.fileno()]), bytes(bufs[p.stderr.fileno()])
