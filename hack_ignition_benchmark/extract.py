"""Run records -> benchmark rows, in the published dataset's layout (families/<family>/{runs,configs}.jsonl).

Reads only what a run wrote (run_card.json, events.jsonl, rollouts.jsonl.gz); never a training log. A run row has
the same top-level fields as the published v0.1 rows, so new runs append to the dataset:

    family run config seed extends model prompt prompt_suffix environment recipe provenance composition intervention
    series item_steps class_series probe flags

    series.steps/n/hack/task/reward   per-step: completions, hack rate, task rate, mean trained reward
                                      task = reward - hack under the exploitable grader, reward under the hardened one
                                      (the published definition); series.raw.hardened_pass is the honest-solve rate
    series.mode_runs                  [[first step, mode], ...] reward-mode runs
    item_steps                        [[step, item, hacks, n], ...]
    class_series                      djinn: {exploit class: [[step, hacks, n], ...]}

Two things the v0.1 rows carry are not derived here: `probe` (the log-prob probe was dropped from the build) and the
family descriptors joined from frozen rate tables (difficulty bands, pre-RL rates, base rates); `hib add` joins those.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

from . import record


def find_runs(root: str) -> list[str]:
    out = []
    for dp, dns, fns in os.walk(root):
        if record.RUN_CARD in fns:
            out.append(dp)
            dns[:] = []                 # a run dir does not contain runs
    return sorted(out)


def _mode_runs(modes: dict[int, str]) -> list[list]:
    runs, last = [], None
    for s in sorted(modes):
        if modes[s] != last:
            runs.append([s, modes[s]])
            last = modes[s]
    return runs


def run_row(run_dir: str) -> dict:
    card = record.read_run_card(run_dir)
    if not str(card.get("record_version", "")).startswith("hib-"):
        raise ValueError(f"{run_dir}: record_version {card.get('record_version')!r} is not a library run "
                         f"(v0.1 runs are already published; this extractor reads library records only)")
    resolved = card["recipe"]["resolved"]
    t = resolved["train"]
    events = list(record.read_events(os.path.join(run_dir, record.EVENTS)))

    # per-step aggregates from the rollouts (one line per completion; the authoritative record)
    per = defaultdict(lambda: {"n": 0, "hack": 0.0, "reward": 0.0, "expl": 0.0, "hard": 0.0})
    # boolean grade channels the environment (and a reward_penalty intervention) declare a per-step rate for
    channels = list((card.get("environment") or {}).get("rate_channels") or [])
    if any(iv.get("type") == "reward_penalty" for iv in card.get("interventions") or []):
        channels.append("penalized")
    chan = defaultdict(lambda: defaultdict(float))
    modes, items, classes = {}, defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
    lines = 0
    rpath = os.path.join(run_dir, record.ROLLOUTS)
    if os.path.exists(rpath):
        for r in record.read_rollouts(rpath):
            lines += 1
            s, g = int(r["step"]), r["grades"]
            p = per[s]
            p["n"] += 1
            p["hack"] += float(g.get("hack", 0))
            p["reward"] += float(r["reward"])
            p["expl"] += float(g.get("exploitable", 0))
            p["hard"] += float(g.get("hardened", 0))
            for c in channels:
                chan[c][s] += float(bool(g.get(c)))
            modes[s] = r.get("reward_mode", "exploitable")
            it = items[(s, r["item"])]
            it[0] += int(float(g.get("hack", 0)) > 0)
            it[1] += 1
            if g.get("exploit_type"):
                c = classes[(g["exploit_type"], s)]
                c[0] += int(float(g.get("hack", 0)) > 0)
                c[1] += 1
    else:   # no rollout tier (run launched with rollouts off): fall back to the events
        for e in events:
            if e["event"] == "hackrate":
                p = per[int(e["step"])]
                p["n"] += e["n"]
                p["hack"] += e["hack"] * e["n"]
                p["reward"] += e["reward"] * e["n"]
                for c in channels:
                    chan[c][int(e["step"])] += e.get(c, 0.0) * e["n"]
                modes[int(e["step"])] = e["mode"]
        for e in events:
            if e["event"] == "items":
                for item, (h, _, _, n) in e["counts"].items():
                    it = items[(int(e["step"]), item)]
                    it[0] += h
                    it[1] += n

    steps = sorted(per)
    n = [per[s]["n"] for s in steps]
    hack = [round(per[s]["hack"] / per[s]["n"], 4) for s in steps]
    reward = [round(per[s]["reward"] / per[s]["n"], 4) for s in steps]
    task = [round((per[s]["reward"] - per[s]["hack"]) / per[s]["n"], 4) if modes.get(s, "exploitable") == "exploitable"
            else round(per[s]["reward"] / per[s]["n"], 4) for s in steps]
    raw = {"channel_key": {"hack": "fraction graded as an exploit",
                           "task": "reward - hack (exploitable mode) | reward (hardened mode)",
                           "reward": "mean trained reward (after any monitor)",
                           "hardened_pass": "fraction passing the hardened grader (honest solve)"},
           "exploitable_pass": [round(per[s]["expl"] / per[s]["n"], 4) for s in steps],
           "hardened_pass": [round(per[s]["hard"] / per[s]["n"], 4) for s in steps]}
    for c in channels:
        raw[c] = [round(chan[c][s] / per[s]["n"], 4) for s in steps]
        raw["channel_key"][c] = f"fraction of completions whose grade has {c} (environment channel)"
    mon = {int(e["step"]): e for e in events if e["event"] == "monitor"}
    if mon:
        for k in ("detected", "caught", "undetected", "false_alarms"):
            raw[f"monitor_{k}"] = [int(mon[s][k]) if s in mon else None for s in steps]
    inject = [(int(e["step"]), int(e["n"])) for e in events if e["event"] == "inject"]

    ivs = card.get("interventions") or []
    iv_by = {iv["type"]: iv for iv in ivs}
    intervention = None
    if ivs:
        inj = iv_by.get("inject") or {}
        intervention = {
            "inject": "inject" in iv_by, "inject_file": (inj.get("hacks") or {}).get("file") if isinstance(inj.get("hacks"), dict) else inj.get("hacks"),
            "inject_k": inj.get("k"), "inject_deadline": inj.get("deadline"), "inject_persist": bool(inj.get("persist")),
            "inject_front": bool(inj.get("front")), "inject_prob": inj.get("prob"), "inject_problems": inj.get("problems") or [],
            "inject_steps": [list(x) for x in inject], "injected_completions_total": sum(x[1] for x in inject),
            "reward_switches": (iv_by.get("reward_schedule") or {}).get("switches") or [],
            "reset_optimizer_at": (iv_by.get("opt_reset") or {}).get("step"),
            "monitor": ({k: iv_by["monitor"].get(k) for k in ("regex", "mode", "penalty", "n_patterns")}
                        if "monitor" in iv_by else None),
            "custom": [iv for iv in ivs if iv["type"] not in ("inject", "reward_schedule", "opt_reset", "monitor")],
        }

    launches = card.get("launches") or []
    resume_steps = sorted({int(e["step"]) for e in events if e["event"] == "phase_end"})[:-1]
    first, last = (steps[0], steps[-1]) if steps else (None, None)
    expected = resolved["phases"][-1]["steps"] if resolved.get("phases") else resolved["steps"]
    env = card.get("environment") or {}
    inputs = card.get("inputs") or {}
    prob = (resolved.get("problems") or {}).get("file") if resolved.get("problems") else None
    comp = {"item_file": os.path.basename(prob) if prob else None, "item_file_sha256": (inputs.get(prob) or {}).get("sha256"),
            "n_items": (card.get("items") or {}).get("n"), "n_unique_problems": (card.get("items") or {}).get("n_unique"),
            "subset": (card.get("items") or {}).get("subset"),
            "items_per_step": (t["per_device_batch"] * t["grad_accum"]) // t["num_generations"]}
    if comp["n_items"] and comp["n_unique_problems"] is not None:
        comp["duplicated_items"] = comp["n_items"] - comp["n_unique_problems"]
    if comp["n_items"]:
        comp["steps_per_revisit"] = round(comp["n_items"] / comp["items_per_step"], 2)
    if classes:
        comp["exploit_classes_seen"] = sorted({c for c, _ in classes})

    return {
        "family": card["family"], "run": f"{card.get('variant') or card['config']}_s{card['seed']}",
        "config": card.get("variant") or card["config"], "seed": card["seed"], "extends": None, "extended_by": None,
        "model": {"id": card["model"]["id"], "revision": card["model"]["revision"], "label": card["model"]["id"].split("/")[-1]},
        "prompt": resolved["prompt"], "prompt_suffix": resolved["append_user"] or None,
        "environment": env,
        "recipe": {"trainer": card["trainer"], "preset": resolved["preset"], "algorithm": "GRPO (TRL)",
                   "lr": t["lr"], "beta": t["beta"], "lr_schedule": t["lr_schedule"], "warmup_ratio": t["warmup_ratio"],
                   "num_generations": t["num_generations"], "per_device_batch": t["per_device_batch"],
                   "grad_accum": t["grad_accum"], "completions_per_step": t["per_device_batch"] * t["grad_accum"],
                   "max_completion": t["max_completion"], "max_steps": expected, "lora": t["lora"] and {"r": t["lora_r"], "alpha": t["lora_r"], "target": "all-linear", "dropout": 0.0},
                   "evaluator": resolved["evaluator"], "reward_backend": resolved["env"], "save_steps": resolved["save_steps"],
                   "gen_stack": f"{t['gen']} bfloat16" if t["gen"] == "vllm" else t["gen"],
                   "vllm_importance_sampling_correction": t["is_correction"],
                   "versions": (launches[0].get("versions") if launches else None),
                   "overrides": card["recipe"].get("overrides") or {}},
        "provenance": {"launched_utc": card.get("launched_utc"), "finished_utc": card.get("finished_utc"),
                       "recipe_path": os.path.relpath(card["recipe"]["path"]) if card["recipe"].get("path") else None,
                       "recipe_sha256": card["recipe"]["sha256"],
                       "code": [l.get("code") for l in launches], "inputs": {k: v.get("sha256") for k, v in inputs.items()},
                       "launches": [{k: l.get(k) for k in ("utc", "phase", "resume", "max_steps", "stop_at")} for l in launches],
                       "resume_steps": resume_steps, "clean_end": bool(card.get("finished_utc")),
                       "rollouts": {"file": record.ROLLOUTS, "lines": lines} if lines else None},
        "composition": comp,
        "intervention": intervention,
        "series": {"steps": steps, "n": n, "hack": hack, "task": task, "reward": reward,
                   "mode_runs": _mode_runs(modes), "raw": raw},
        "item_steps": [[s, it, h, m] for (s, it), (h, m) in sorted(items.items())],
        "class_series": ({c: [[s, h, m] for (cc, s), (h, m) in sorted(classes.items()) if cc == c]
                          for c in sorted({c for c, _ in classes})} if classes else None),
        "probe": None,
        "flags": {"steps_logged": len(steps), "first_step": first, "last_step": last,
                  "missing_steps": (last - first + 1 - len(steps)) if steps else 0,
                  "stopped_before_max_steps": bool(steps) and last + 1 < expected,
                  "continued_past_max_steps": bool(steps) and last + 1 > expected,
                  "resume_step": resume_steps[0] if resume_steps else None},
    }


def config_rows(runs: list[dict]) -> list[dict]:
    by = defaultdict(list)
    for r in runs:
        by[(r["family"], r["config"])].append(r)
    out = []
    for (fam, cfg), rs in sorted(by.items()):
        rs.sort(key=lambda r: r["seed"])
        r0 = rs[0]
        out.append({"family": fam, "config": cfg, "extends": None, "model": r0["model"]["label"],
                    "model_id": r0["model"]["id"], "model_revision": r0["model"]["revision"],
                    "prompt": r0["prompt"], "prompt_suffix": r0["prompt_suffix"], "environment": r0["environment"],
                    "recipe": {k: v for k, v in r0["recipe"].items() if k != "max_steps"},
                    "composition": r0["composition"], "intervention": r0["intervention"],
                    "seeds": [r["seed"] for r in rs], "n_seeds": len(rs), "runs": [r["run"] for r in rs],
                    "max_steps": sorted({r["recipe"]["max_steps"] for r in rs}),
                    "steps_logged": [r["flags"]["steps_logged"] for r in rs],
                    "first_step": [r["flags"]["first_step"] for r in rs], "last_step": [r["flags"]["last_step"] for r in rs],
                    "launched_utc": [r["provenance"]["launched_utc"] for r in rs]})
    return out


def extract(runs_root: str, out_dir: str) -> dict:
    rows, skipped = [], []
    for d in find_runs(runs_root):
        try:
            rows.append(run_row(d))
        except Exception as e:  # noqa: BLE001
            skipped.append((d, f"{type(e).__name__}: {e}"))
    fams = defaultdict(list)
    for r in rows:
        fams[r["family"]].append(r)
    for fam, rs in fams.items():
        d = os.path.join(out_dir, fam)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "runs.jsonl"), "w") as f:
            for r in sorted(rs, key=lambda r: r["run"]):
                f.write(json.dumps(r) + "\n")
        with open(os.path.join(d, "configs.jsonl"), "w") as f:
            for c in config_rows(rs):
                f.write(json.dumps(c) + "\n")
    return {"runs": len(rows), "families": {k: len(v) for k, v in fams.items()}, "skipped": skipped}
