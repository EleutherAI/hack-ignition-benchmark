"""Frozen inputs: resolve a recipe's file references to local files and check their sha256.

A FileRef path is relative to the problem-set root, e.g. `mbpp/mbpp_rl_band3.jsonl`. Resolution order:

    1. a local directory given by --data or HIB_DATA (any layout that mirrors the relative paths)
    2. the Hub dataset (EleutherAI/hack-ignition-benchmark at the pinned revision), where the published problem
       sets live under families/<family>/problem_sets/ -- `problem_sets/INDEX.json` maps each relative path to its
       Hub path and sha256

Every resolved file is hashed. A recipe's sha256, when present, must match; otherwise the index's sha256 must.
A mismatch is an error, never a warning: a run on the wrong input is not a run of that config.
"""
from __future__ import annotations

import hashlib
import json
import os

HUB_REPO = "EleutherAI/hack-ignition-benchmark"
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class DataError(RuntimeError):
    pass


def file_sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


class DataStore:
    def __init__(self, local_root: str | None = None, index_path: str | None = None, offline: bool = False):
        self.local_root = local_root or os.environ.get("HIB_DATA")
        self.index_path = index_path or os.path.join(_PKG_ROOT, "problem_sets", "INDEX.json")
        self.offline = offline or bool(os.environ.get("HF_HUB_OFFLINE"))
        self._index = None
        self.resolved: dict[str, dict] = {}     # rel path -> {path, sha256, source}; the run card records this

    @property
    def index(self) -> dict:
        if self._index is None:
            try:
                with open(self.index_path) as f:
                    self._index = json.load(f)
            except FileNotFoundError:
                self._index = {"hub_revision": None, "files": {}}
        return self._index

    def expected_sha(self, rel: str, pinned: str | None) -> str | None:
        return pinned or (self.index["files"].get(rel) or {}).get("sha256")

    def path(self, ref) -> str:
        """Local path of a FileRef (or a bare relative path), fetched if needed, sha256-checked."""
        rel, pinned = (ref, None) if isinstance(ref, str) else (ref.file, ref.sha256)
        if os.path.isabs(rel):
            local, source = rel, "absolute"
        else:
            local, source = self._find_local(rel), "local"
            if local is None:
                local, source = self._fetch(rel), "hub"
        got = file_sha256(local)
        want = self.expected_sha(rel, pinned)
        if want and got != want:
            raise DataError(f"{rel}: sha256 {got[:12]}… does not match the pinned {want[:12]}… ({local})")
        self.resolved[rel] = {"path": local, "sha256": got, "source": source, "pinned": bool(want)}
        return local

    def _find_local(self, rel: str) -> str | None:
        for root in filter(None, [self.local_root, os.path.join(_PKG_ROOT, "problem_sets")]):
            p = os.path.join(root, rel)
            if os.path.isfile(p):
                return p
        return None

    def _fetch(self, rel: str) -> str:
        entry = self.index["files"].get(rel)
        if not entry:
            raise DataError(f"{rel}: not found under {self.local_root or '(no --data dir)'} and not in "
                            f"{os.path.relpath(self.index_path, _PKG_ROOT)}")
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise DataError(f"{rel}: not available locally; `pip install huggingface_hub` to fetch it from "
                            f"{HUB_REPO}") from None
        try:        # offline (e.g. the network-less sandbox): a file already in the HF cache still resolves
            return hf_hub_download(HUB_REPO, entry["hub_path"], repo_type="dataset",
                                   revision=self.index.get("hub_revision"), local_files_only=self.offline)
        except Exception as e:  # noqa: BLE001
            if self.offline:
                raise DataError(f"{rel}: not in the local HF cache and HF_HUB_OFFLINE is set ({type(e).__name__})") from None
            raise
