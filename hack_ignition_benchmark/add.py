"""hib add: lay extracted rows out in the dataset's format, ready for whoever publishes it. Never pushes.

    stage/families/<family>/runs.jsonl       rows (appended to --base's rows for the family when given)
    stage/families/<family>/configs.jsonl    rebuilt from all rows of the family
    stage/families/<family>/MANIFEST.json    sha256 + bytes of every staged file
    stage/rollouts/<family>/<run>/           rollouts.jsonl.gz + run_card.json (with --runs)

A row whose (family, run) already exists in the base is refused unless --replace: appending must never silently
overwrite a published run.
"""
from __future__ import annotations

import json
import os
import shutil

from .data import file_sha256
from .extract import config_rows, find_runs
from . import record


def _read(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def stage(rows_dir: str, family: str, stage_dir: str, base_dir: str | None = None, runs_root: str | None = None,
          replace: bool = False, source_family: str | None = None) -> dict:
    src = _read(os.path.join(rows_dir, source_family or family, "runs.jsonl"))
    if not src:
        raise FileNotFoundError(f"no rows under {rows_dir}/{source_family or family}/runs.jsonl (run `hib extract` first)")
    for r in src:
        r["family"] = family
    base = _read(os.path.join(base_dir, "families", family, "runs.jsonl")) if base_dir else []
    have = {r["run"] for r in base}
    clash = sorted(r["run"] for r in src if r["run"] in have)
    if clash and not replace:
        raise ValueError(f"{len(clash)} run(s) already in the base family {family}: {clash[:5]} (use --replace)")
    keep = [r for r in base if r["run"] not in {x["run"] for x in src}]
    rows = sorted(keep + src, key=lambda r: r["run"])
    fam_dir = os.path.join(stage_dir, "families", family)
    os.makedirs(fam_dir, exist_ok=True)
    with open(os.path.join(fam_dir, "runs.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    base_cfg = {c["config"]: c for c in _read(os.path.join(base_dir, "families", family, "configs.jsonl"))} if base_dir else {}
    new_cfg = {c["config"]: c for c in config_rows(src)}
    cfgs = {**base_cfg, **new_cfg}
    if base_dir:   # a config that gained seeds: rebuild its aggregate from all its rows
        for name in new_cfg:
            if name in base_cfg:
                cfgs[name] = {**base_cfg[name], **config_rows([r for r in rows if r["config"] == name])[0]}
    with open(os.path.join(fam_dir, "configs.jsonl"), "w") as f:
        for name in sorted(cfgs):
            f.write(json.dumps(cfgs[name]) + "\n")
    n_roll = 0
    if runs_root:
        by_run = {}
        for d in find_runs(runs_root):
            try:
                c = record.read_run_card(d)
            except Exception:  # noqa: BLE001
                continue
            by_run[f"{c.get('variant') or c['config']}_s{c['seed']}"] = d
        for r in src:
            d = by_run.get(r["run"])
            if d and os.path.exists(os.path.join(d, record.ROLLOUTS)):
                dst = os.path.join(stage_dir, "rollouts", family, r["run"].replace("/", "__"))
                os.makedirs(dst, exist_ok=True)
                for fn in (record.ROLLOUTS, record.RUN_CARD, record.EVENTS):
                    if os.path.exists(os.path.join(d, fn)):
                        shutil.copy2(os.path.join(d, fn), os.path.join(dst, fn))
                n_roll += 1
    files = {}
    for dp, _, fns in os.walk(fam_dir):
        for fn in sorted(fns):
            if fn == "MANIFEST.json":
                continue
            p = os.path.join(dp, fn)
            rel = os.path.relpath(p, fam_dir)
            files[rel] = {"path": rel, "bytes": os.path.getsize(p), "sha256": file_sha256(p)}
    with open(os.path.join(fam_dir, "MANIFEST.json"), "w") as f:
        json.dump({"family": family, "n_configs": len(cfgs), "n_runs": len(rows), "files": files}, f, indent=1)
    return {"family": family, "runs_added": len(src), "runs_total": len(rows), "configs": len(cfgs),
            "rollout_dirs": n_roll, "stage": fam_dir, "replaced": clash if replace else []}
