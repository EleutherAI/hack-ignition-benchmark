"""Write a `catch` problem file: a subset of CATCH's RL tasks as JSONL, with the fields the environment reads.

    python tools/build_catch_problems.py OUT.jsonl --row-groups 40 --seed 0 [--max-task-bytes 1048576]
    python tools/build_catch_problems.py OUT.jsonl --row-groups 3,17,101          (these row groups; "17," alone)
    python tools/build_catch_problems.py OUT.jsonl --parquet LOCAL.parquet ...    (a local copy instead of the Hub)

The source is `WangSl2004/CATCH-RL`, `data/train_verl.parquet` (24,287 tasks in 243 row groups of 100, 28 GB) at a
pinned revision. Its rows are in the order of DeepCoder's sources (the function-call tasks sit in the middle, the
functional ones at the end), so a subset samples whole row groups across the file: `--row-groups K --seed S` takes K
of them uniformly at random. Only the environment's columns are read, which skips the `tests`, `solutions` and
original-question columns (about a third of the file).

`--solutions S.jsonl` also writes each kept task's reference solutions ({uid, solutions}), the input of
tools/catch_solutions_check.py.

`--max-task-bytes` drops tasks whose ground truth plus calls.json exceed it (calls.json holds every ground-truth case,
so this bounds both the item's size and the hard test's input). The default keeps every task. A manifest beside the
output (OUT.manifest.json) records the source, its revision, the row groups, the filter, the counts and the file's
sha256.

Needs pyarrow and huggingface_hub (`uv run --with pyarrow --with huggingface_hub python tools/...`).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys

SOURCE_REPO = "WangSl2004/CATCH-RL"
SOURCE_FILE = "data/train_verl.parquet"
SOURCE_REVISION = "f8be954c9304d829e3fc7aed2eea3e98bc82be1e"     # 2026-10-05
FIELDS = ("uid", "index", "data_source", "swe_task_kind", "planner_function_name", "question", "repo_files",
          "repo_file_permissions", "ground_truth", "selected_test_cases_for_visible_tests")


def open_parquet(path: str | None, revision: str):
    import pyarrow.parquet as pq
    if path:
        return pq.ParquetFile(path), None
    from huggingface_hub import HfFileSystem
    fh = HfFileSystem().open(f"datasets/{SOURCE_REPO}@{revision}/{SOURCE_FILE}", "rb", block_size=1 << 22)
    return pq.ParquetFile(fh), fh


def choose_row_groups(spec: str, n_groups: int, seed: int) -> list[int]:
    """`spec` is a count K (sample K with `seed`) or a comma list of indices ("17," for row group 17 alone)."""
    if "," in spec:
        groups = sorted({int(x) for x in spec.split(",") if x.strip()})
        bad = [g for g in groups if not 0 <= g < n_groups]
        if bad:
            raise SystemExit(f"row groups out of range 0..{n_groups - 1}: {bad}")
        return groups
    return sorted(random.Random(seed).sample(range(n_groups), int(spec)))


def task_bytes(row: dict) -> int:
    return len((row.get("ground_truth") or "").encode()) + len(((row.get("repo_files") or {}).get("calls.json")
                                                                or "").encode())


def item_from_row(row: dict) -> dict:
    """The row's environment fields; the parquet struct's absent repository files (null) are dropped."""
    it = {k: row.get(k) for k in FIELDS}
    it["repo_files"] = {p: c for p, c in (row.get("repo_files") or {}).items() if c is not None}
    it["repo_file_permissions"] = {p: m for p, m in (row.get("repo_file_permissions") or {}).items()
                                   if m is not None}
    return it


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--row-groups", required=True, help="K (sample K row groups with --seed) or a comma list")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-task-bytes", type=int, default=None)
    ap.add_argument("--parquet", default=None, help="a local copy of train_verl.parquet (default: read the Hub)")
    ap.add_argument("--revision", default=SOURCE_REVISION)
    ap.add_argument("--solutions", default=None, help="also write the kept tasks' reference solutions here")
    a = ap.parse_args(argv)

    pf, fh = open_parquet(a.parquet, a.revision)
    try:
        md = pf.metadata
        groups = choose_row_groups(a.row_groups, md.num_row_groups, a.seed)
        cols = [f"extra_info.{k}" for k in FIELDS + (("solutions",) if a.solutions else ())]
        kept = dropped = 0
        kinds: dict[str, int] = {}
        tmp = a.out + ".part"
        sol = open(a.solutions, "w") if a.solutions else None
        with open(tmp, "w") as out:
            for g in groups:
                rows = pf.read_row_group(g, columns=cols).column("extra_info").to_pylist()
                for row in rows:
                    if a.max_task_bytes is not None and task_bytes(row) > a.max_task_bytes:
                        dropped += 1
                        continue
                    it = item_from_row(row)
                    out.write(json.dumps(it, ensure_ascii=False) + "\n")
                    if sol:
                        sol.write(json.dumps({"uid": row["uid"], "solutions": row.get("solutions")},
                                             ensure_ascii=False) + "\n")
                    kept += 1
                    kinds[it["swe_task_kind"]] = kinds.get(it["swe_task_kind"], 0) + 1
                print(f"row group {g}: {kept} kept, {dropped} dropped so far", file=sys.stderr, flush=True)
        if sol:
            sol.close()
        os.replace(tmp, a.out)
    finally:
        if fh is not None:
            fh.close()
    h = hashlib.sha256()
    with open(a.out, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    manifest = {"source": {"repo": SOURCE_REPO, "file": SOURCE_FILE,
                           "revision": None if a.parquet else a.revision, "local": a.parquet},
                "row_groups": groups, "selection": {"row_groups": a.row_groups, "seed": a.seed},
                "max_task_bytes": a.max_task_bytes, "tasks": kept, "dropped_by_size": dropped,
                "task_kinds": dict(sorted(kinds.items())), "sha256": h.hexdigest(),
                "bytes": os.path.getsize(a.out)}
    with open(a.out + ".manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
