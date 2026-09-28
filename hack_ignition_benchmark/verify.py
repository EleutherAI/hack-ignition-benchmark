"""hib verify <run>: the mechanical half of an audit. Prints one PASS / FAIL / WARN / SKIP line per check.

    card        run_card.json exists, is a library record, and the run finished cleanly
    recipe      the recipe text stored in the card hashes to the recorded sha256; if the recipe file is present, it is
                the same file (a changed recipe means the run is not a run of the current config)
    inputs      every input the run resolved is pinned: its sha256 equals the problem-set index (and the local copy,
                when it is still on disk, still hashes the same)
    code        the card names a commit, the tree was not dirty, and (if this package is that commit) the installed
                tree hash equals the recorded one
    rollouts    the rollout file is complete: one line per completion for every step, steps contiguous, no truncation
    rewards     every recorded reward equals the environment's reward for the recorded grades under the recorded mode,
                after any monitor (recomputed from the record; no model code runs)
    events      the per-step hack and reward means in events.jsonl agree with the rollouts
    regrade     (--regrade N, sandbox only) N sampled completions are graded again and must match their recorded grades
"""
from __future__ import annotations

import hashlib
import os
import random
from collections import defaultdict

from . import record
from .data import DataStore, file_sha256


class Report:
    def __init__(self):
        self.rows = []

    def add(self, check, status, detail=""):
        self.rows.append((check, status, detail))

    @property
    def ok(self):
        return all(s != "FAIL" for _, s, _ in self.rows)

    def text(self):
        w = max(len(c) for c, _, _ in self.rows) if self.rows else 8
        return "\n".join(f"{c:<{w}}  {s:<4}  {d}" for c, s, d in self.rows)


def verify(run_dir: str, regrade: int = 0, sandbox: bool = False, data: DataStore | None = None) -> Report:
    rep = Report()
    data = data or DataStore()
    try:
        card = record.read_run_card(run_dir)
    except FileNotFoundError:
        rep.add("card", "FAIL", "no run_card.json")
        return rep
    if not str(card.get("record_version", "")).startswith("hib-"):
        rep.add("card", "FAIL", f"record_version {card.get('record_version')!r} is not a library record")
        return rep
    rep.add("card", "PASS" if card.get("finished_utc") else "WARN",
            f"{card['family']}/{card.get('variant') or card['config']} seed {card['seed']}, "
            + (f"finished {card['finished_utc']}" if card.get("finished_utc") else "no finished_utc (unclean end or still running)"))

    # recipe
    rc = card["recipe"]
    if rc.get("text") is not None:
        h = hashlib.sha256(rc["text"].encode()).hexdigest()
        if h != rc["sha256"]:
            rep.add("recipe", "FAIL", "stored recipe text does not hash to the recorded sha256")
        elif rc.get("path") and os.path.exists(rc["path"]):
            now = file_sha256(rc["path"])
            rep.add("recipe", "PASS" if now == rc["sha256"] else "WARN",
                    "recipe file unchanged" if now == rc["sha256"] else f"{rc['path']} has changed since the run")
        else:
            rep.add("recipe", "PASS", "recipe text in the card matches its sha256")
    else:
        rep.add("recipe", "WARN", "card holds no recipe text (custom run?)")
    if rc.get("overrides"):
        rep.add("overrides", "WARN" if card.get("variant") else "PASS", f"{rc['overrides']} -> {card.get('variant') or 'not a variant'}")

    # inputs
    bad, unpinned = [], []
    for rel, v in (card.get("inputs") or {}).items():
        want = data.expected_sha(rel, None)
        if want is None:
            unpinned.append(rel)
        elif want != v["sha256"]:
            bad.append(f"{rel}: ran {v['sha256'][:12]}, index {want[:12]}")
        if v.get("path") and os.path.exists(v["path"]) and file_sha256(v["path"]) != v["sha256"]:
            bad.append(f"{rel}: local copy changed since the run")
    rep.add("inputs", "FAIL" if bad else ("WARN" if unpinned else "PASS"),
            "; ".join(bad) or (f"not in the index: {unpinned}" if unpinned else f"{len(card.get('inputs') or {})} input(s) pinned"))

    # code
    codes = [l.get("code") or {} for l in card.get("launches") or []]
    commits = {c.get("commit") for c in codes}
    if None in commits or not codes:
        rep.add("code", "WARN", "a launch has no commit (not a git checkout and HIB_COMMIT unset)")
    elif any(c.get("dirty") for c in codes):
        rep.add("code", "WARN", f"commit {sorted(commits)} with a dirty tree")
    else:
        here = record.code_provenance()
        same = here["commit"] in commits
        match = same and all(c.get("tree_sha256") == here["tree_sha256"] for c in codes)
        rep.add("code", "PASS" if (match or not same) else "FAIL",
                f"commit {sorted(x[:12] for x in commits)}" + ("; installed tree hash matches" if match else
                "; installed package is a different commit (tree hash not compared)" if not same else
                "; installed tree hash DIFFERS from the one the run recorded"))

    # rollouts, rewards, events
    rpath = os.path.join(run_dir, record.ROLLOUTS)
    if not os.path.exists(rpath):
        rep.add("rollouts", "SKIP", "no rollout file")
        return rep
    recs = list(record.read_rollouts(rpath))
    resolved = rc["resolved"]
    t = resolved["train"]
    per_step = t["per_device_batch"] * t["grad_accum"]
    steps = defaultdict(int)
    for r in recs:
        steps[int(r["step"])] += 1
    s_sorted = sorted(steps)
    short = [s for s in s_sorted if steps[s] % per_step]
    gaps = (s_sorted[-1] - s_sorted[0] + 1 - len(s_sorted)) if s_sorted else 0
    detail = (f"{len(recs)} lines over steps {s_sorted[0]}..{s_sorted[-1]}" if s_sorted else "empty")
    detail += (f"; {len(short)} step(s) with a partial batch" if short else "") + (f"; {gaps} missing step(s)" if gaps else "")
    rep.add("rollouts", "FAIL" if (short or gaps or not s_sorted) else "PASS", detail)

    mon = next((iv for iv in card.get("interventions") or [] if iv["type"] == "monitor"), None)
    pats = None
    if mon:
        from .interventions.builtin import load_monitor_patterns
        _, pats = load_monitor_patterns(mon["regex"])
    structure = (card.get("environment") or {}).get("options", {}).get("reward_structure")
    mism = 0
    for r in recs:
        g, mode = r["grades"], r.get("reward_mode", "exploitable")
        want = g["hardened"] if mode == "hardened" else g["exploitable"]
        if pats is not None and any(p.search(r["text"]) for p in pats):
            want = 0.0 if mon["mode"] == "squash" else float(mon["penalty"])
        if abs(float(r["reward"]) - float(want)) > 1e-9:
            mism += 1
    rep.add("rewards", "FAIL" if mism else "PASS",
            f"{mism} of {len(recs)} rewards differ from grades+mode" if mism else
            f"all {len(recs)} rewards equal the grades under their mode" + (f" and {mon['mode']} monitor" if mon else "")
            + (f" ({structure})" if structure else ""))

    ev = [e for e in record.read_events(os.path.join(run_dir, record.EVENTS)) if e["event"] == "hackrate"]
    agg = defaultdict(lambda: [0, 0.0, 0.0])
    for e in ev:
        a = agg[int(e["step"])]
        a[0] += e["n"]; a[1] += e["hack"] * e["n"]; a[2] += e["reward"] * e["n"]
    racc = defaultdict(lambda: [0, 0.0, 0.0])
    for r in recs:
        a = racc[int(r["step"])]
        a[0] += 1; a[1] += float(r["grades"]["hack"]); a[2] += float(r["reward"])
    off = [s for s in racc if s not in agg or agg[s][0] != racc[s][0]
           or abs(agg[s][1] - racc[s][1]) > 0.01 * agg[s][0] or abs(agg[s][2] - racc[s][2]) > 0.01 * agg[s][0]]
    rep.add("events", "FAIL" if off else "PASS",
            f"{len(off)} step(s) where events and rollouts disagree: {sorted(off)[:5]}" if off else
            f"events agree with rollouts on {len(racc)} steps")

    if regrade:
        if not sandbox:
            rep.add("regrade", "SKIP", "regrading executes model-written code; rerun with --i-am-in-a-sandbox")
            return rep
        from .envs.base import get_env
        from .recipe import load_recipe
        recipe = load_recipe(rc["path"]) if rc.get("path") and os.path.exists(rc["path"]) else None
        if recipe is None:
            rep.add("regrade", "SKIP", "recipe file not available")
            return rep
        wd = os.path.join(run_dir, "verify_cwd")
        os.makedirs(wd, exist_ok=True)
        env = get_env(recipe.env)(recipe, workdir=wd, data=data)
        items = {env.item_id(it): it for it in env.load_items()}
        sample = random.Random(0).sample(recs, min(regrade, len(recs)))
        env.start()
        try:
            gs = env.grade([items[r["item"]] for r in sample], [r["text"] for r in sample])
        finally:
            env.close()
        diff = [(r["step"], r["idx"]) for r, g in zip(sample, gs)
                if (g.exploitable, float(g.hack), g.hardened) != (r["grades"]["exploitable"], r["grades"]["hack"], r["grades"]["hardened"])]
        rep.add("regrade", "FAIL" if diff else "PASS",
                f"{len(diff)} of {len(sample)} regraded completions differ: {diff[:5]}" if diff else
                f"{len(sample)} sampled completions regrade identically")
    return rep
