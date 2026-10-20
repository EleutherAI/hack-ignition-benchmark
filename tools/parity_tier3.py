"""Tier-3 parity: does each regenerated run land inside its config's published seed spread?

For every library run under the given roots, the run is extracted (`extract.run_row`) and labelled with the reference
rule at a horizon H; the published seeds of the same config (the Hub's families/<family>/runs.jsonl at the pinned
revision) are labelled the same way. A regenerated run passes if its label is one the published seeds reached.

Labels here are two-way, ignited / not ignited (`reference.labels.three_way` with no pre-RL rate, so "dead" is never
called): the dead/stable split needs each config's harvested base rate and is not what bistability makes binary.
H = min(250, the fewest steps any run of the config logged), so every run is compared at a step all of them reached.

    python tools/parity_tier3.py RUNS_ROOT [RUNS_ROOT ...] --out report.json [--md report.md]
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hack_ignition_benchmark import extract  # noqa: E402
from hack_ignition_benchmark.reference import labels  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def published_rows(families: set[str]) -> dict[str, list[dict]]:
    from huggingface_hub import hf_hub_download
    rev = json.load(open(os.path.join(ROOT, "problem_sets", "INDEX.json")))["hub_revision"]
    by: dict[str, list[dict]] = collections.defaultdict(list)
    for fam in sorted(families):
        p = hf_hub_download("EleutherAI/hack-ignition-benchmark", f"families/{fam}/runs.jsonl", repo_type="dataset",
                            revision=rev)
        for line in open(p):
            r = json.loads(line)
            cfg = r["config"]
            if fam == "monitor" and "/" not in cfg:      # the squash half is stored without its group prefix
                cfg = "monitor/" + cfg
            by[f"{fam}/{cfg}"].append(r)
    return by


def label(row: dict, H: int) -> tuple[str, float]:
    s = row["series"]
    lab, _ = labels.three_way(s["hack"], s["n"], H, rule="hackfree", h0=None)
    tail = labels.window_mean(s["hack"], max(H - 20, 0), H)
    return ("ignited" if lab == "ignited" else "not ignited"), round(tail, 3)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--md", default=None)
    a = ap.parse_args(argv)
    new: dict[str, list[dict]] = collections.defaultdict(list)
    skipped = []
    for root in a.roots:
        for d in extract.find_runs(root):
            r = extract.run_row(d)
            if not r["provenance"].get("finished_utc"):      # still running, or ended uncleanly
                skipped.append(d)
                continue
            # a variant (config@hash) is compared with its base config's published seeds and marked as a variant
            new[f"{r['family']}/{r['config'].split('@', 1)[0]}"].append(r)
    pub = published_rows({k.split("/", 1)[0] for k in new})
    rep, md = [], ["| config | H | published seeds: label (tail hack) | regenerated: label (tail hack) | in spread |",
                   "|---|---:|---|---|---|"]
    for cfg in sorted(new):
        P = pub.get(cfg, [])
        if not P:
            rep.append({"config": cfg, "error": "no published rows"})
            md.append(f"| {cfg} | | none found | | ? |")
            continue
        H = min([250] + [len(r["series"]["hack"]) for r in P + new[cfg]])
        pl = {f"s{r['seed']}": label(r, H) for r in sorted(P, key=lambda r: r["seed"])}
        def tag(r):
            ov = r["recipe"].get("overrides") or {}
            changing = sorted(k for k in ov if k not in ("vllm_mem", "vllm_sleep", "save_steps", "save_at",
                                                         "keep_optimizer"))
            return f"s{r['seed']}" + (f" (variant: {', '.join(changing)})" if changing else "")
        nl = {tag(r): label(r, H) for r in sorted(new[cfg], key=lambda r: r["seed"])}
        spread = {l for l, _ in pl.values()}
        ok = all(l in spread for l, _ in nl.values())
        rep.append({"config": cfg, "H": H, "published": pl, "regenerated": nl, "in_spread": ok})
        fmt = lambda d: ", ".join(f"{k} {l} ({t})" for k, (l, t) in d.items())
        md.append(f"| {cfg} | {H} | {fmt(pl)} | {fmt(nl)} | {'yes' if ok else '**no**'} |")
    n_ok = sum(1 for r in rep if r.get("in_spread"))
    summary = {"configs": len(rep), "in_spread": n_ok, "outside": [r["config"] for r in rep if r.get("in_spread") is False],
               "missing_published": [r["config"] for r in rep if "error" in r], "skipped_unfinished": skipped}
    json.dump({"summary": summary, "configs": rep}, open(a.out, "w"), indent=1)
    if a.md:
        open(a.md, "w").write("\n".join(md) + "\n")
    print(json.dumps(summary, indent=1))
    print("\n".join(md))
    return 0 if not summary["outside"] else 1


if __name__ == "__main__":
    sys.exit(main())
