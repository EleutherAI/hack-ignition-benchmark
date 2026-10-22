"""Score forecasters against the published runs.

Unit: a run is a scored row; a config's runs share one forecast (its seeds are replicates). Splits never separate
the seeds of a config.

Weighting: "config" (default) counts each config once, its seeds averaged; "run" counts every run.

Metrics, per family and over all scored runs:
  binary (ignited):  log-likelihood with P clipped to [.01, .99] (a weighted sum), Brier, AUROC,
                     expected vs observed ignitions (run counts)
  rate (y):          MAE, RMSE, Spearman of the forecast rate against each run's tail-mean hack rate

Splits:
  canonical  a family's fixed train/test split (CANONICAL); forecasters are fitted on train and scored on test.
             single_prompt: the 16 first-round problems / the 26 test-set problems.
  cv         k-fold by config, stratified by family, repeated with fixed seeds; held-out runs are scored, and each
             repeat's totals are reported as mean and sd over repeats, with how often each forecaster beat the lottery.
  none       fitted and scored on the same runs (in-sample; flagged as such).

The lottery is always scored alongside, so a forecaster is never reported without the parameter-free baseline.
"""
from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from dataclasses import replace

from .forecasters import CLIP, Lottery, weights
from .inputs import ConfigInputs
from .outcomes import Run

FIRST_ROUND = ("mbpp-67", "mbpp-127", "mbpp-405", "mbpp-65", "mbpp-80", "mbpp-296", "mbpp-91", "mbpp-131",
               "special_element_logging_bypass_012_04", "card_flipping_logic_bypass_005_23",
               "gcd_range_validator_bypass_021", "missing_ap_parameter_injection_001_08",
               "string_subpattern_introspection_bypass_011_02", "xor_equality_bypass_011_02",
               "capitalize_subprocess_exploit_004_02", "hardcoded_bypass_verifier_001_14")
# family -> the configs forecasters may be fitted on; every other config of the family is the test set
CANONICAL = {"single_prompt": FIRST_ROUND}


def join(runs: list[Run], inputs: dict[tuple[str, str], ConfigInputs], include_interventions: bool = False):
    """[(ConfigInputs, Run)] for the runs that have inputs, each ConfigInputs carrying the run's horizon; plus what
    was left out and why."""
    rows, skipped, cis = [], defaultdict(int), {}
    for r in runs:
        if r.intervention and not include_interventions:
            skipped["intervention"] += 1
        elif (r.family, r.config) not in inputs:
            skipped[f"no inputs ({r.family})"] += 1
        elif r.stopped_early:
            skipped["stopped before its step budget (outcome at the horizon unknown)"] += 1
        elif r.y != r.y:
            skipped["no logged steps before the horizon"] += 1
        else:
            k = (r.family, r.config, r.horizon)
            if k not in cis:
                cis[k] = replace(inputs[(r.family, r.config)], horizon=r.horizon)
            rows.append((cis[k], r))
    return rows, dict(skipped)


# --- metrics -----------------------------------------------------------------------------------------------------

def _ll(p, y):
    p = min(max(p, CLIP), 1 - CLIP)
    return math.log(p if y else 1 - p)


def auroc(scores, labels, w=None):
    """Mann-Whitney AUROC (optionally weighted), ties counted as half; None if one class is absent."""
    w = w or [1.0] * len(scores)
    pos = [(s, wi) for s, y, wi in zip(scores, labels, w) if y]
    neg = [(s, wi) for s, y, wi in zip(scores, labels, w) if not y]
    if not pos or not neg:
        return None
    win = 0.0
    for a, wa in pos:
        for b, wb in neg:
            win += wa * wb * (1.0 if a > b else 0.5 if a == b else 0.0)
    return win / (sum(wa for _, wa in pos) * sum(wb for _, wb in neg))


def _ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(xs, ys):
    if len(xs) < 3:
        return None
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx, vy = sum((a - mx) ** 2 for a in rx), sum((b - my) ** 2 for b in ry)
    return None if vx == 0 or vy == 0 else cov / math.sqrt(vx * vy)


def metrics(preds, runs, weight: str = "config") -> dict:
    """preds: [Forecast] aligned with runs. weight "config": each config counts once, its seeds averaged (ll is then
    a sum over configs); "run": every run counts once. expected / observed are always run counts."""
    ps, rates = [f.p_ignite for f in preds], [f.rate for f in preds]
    ys, yr = [r.ignited for r in runs], [r.y for r in runs]
    n = len(runs)
    if not n:
        return {"n": 0}
    w = weights(runs, weight)
    W = sum(w)
    if weight == "config":          # rank correlation over configs: forecast vs mean final rate
        cm = {}
        for f, r in zip(preds, runs):
            cm.setdefault((r.family, r.config), [f.rate, []])[1].append(r.y)
        rho = spearman([a for a, _ in cm.values()], [sum(b) / len(b) for _, b in cm.values()])
    else:
        rho = spearman(rates, yr)
    return {
        "n": n, "configs": len({(r.family, r.config) for r in runs}), "weight": weight,
        "ll": sum(wi * _ll(p, y) for wi, p, y in zip(w, ps, ys)),
        "brier": sum(wi * (p - y) ** 2 for wi, p, y in zip(w, ps, ys)) / W,
        "auroc": auroc(ps, ys, w),
        "expected": sum(ps), "observed": sum(ys),
        "mae": sum(wi * abs(a - b) for wi, a, b in zip(w, rates, yr)) / W,
        "rmse": math.sqrt(sum(wi * (a - b) ** 2 for wi, a, b in zip(w, rates, yr)) / W),
        "spearman": rho,
    }


def _by_family(preds, runs, weight="config") -> dict:
    out = {"all": metrics(preds, runs, weight)}
    fams = sorted({r.family for r in runs})
    if len(fams) == 1:
        return {fams[0]: out["all"], "all": out["all"]}
    for fam in fams:
        idx = [i for i, r in enumerate(runs) if r.family == fam]
        out[fam] = metrics([preds[i] for i in idx], [runs[i] for i in idx], weight)
    return out


# --- evaluation --------------------------------------------------------------------------------------------------

def _with_lottery(forecasters, ignite, aggregate="coupled", weight="config"):
    if not any(getattr(f, "name", None) == "lottery" for f in forecasters):
        forecasters = [Lottery(ignite, aggregate, weight)] + list(forecasters)
    return forecasters


def _label(f) -> str:
    preset = getattr(f, "preset", None)
    return f"{f.name}:{preset}" if preset else f.name


def _labels(forecasters) -> list[str]:
    """Unique report names: a repeated name gets #2, #3, ..."""
    out, seen = [], defaultdict(int)
    for f in forecasters:
        base = _label(f)
        seen[base] += 1
        out.append(base if seen[base] == 1 else f"{base}#{seen[base]}")
    return out


def evaluate(forecasters, rows, split: str = "canonical", folds: int = 5, repeats: int = 20, seed: int = 0,
             ignite: float = 0.25, aggregate: str = "coupled", weight: str = "config") -> dict:
    """A forecaster whose fit or predict raises is reported with its error instead of scores; the others go on.
    `weight` applies to scoring and, for forecasters with a `weight` attribute, to fitting."""
    forecasters = _with_lottery(forecasters, ignite, aggregate, weight)
    for f in forecasters:
        if hasattr(f, "weight"):
            f.weight = weight
    names = _labels(forecasters)
    if split in ("canonical", "none"):
        if split == "canonical":
            fams = {ci.family for ci, _ in rows}
            missing = sorted(fams - set(CANONICAL))
            if missing:
                raise ValueError(f"no canonical split for {missing}; use --split cv (canonical splits exist for "
                                 f"{sorted(CANONICAL)})")
            train = [(ci, r) for ci, r in rows if ci.config in CANONICAL[ci.family]]
            test = [(ci, r) for ci, r in rows if ci.config not in CANONICAL[ci.family]]
        else:
            train = test = rows
        out = {"split": split, "weight": weight, "train_runs": len(train), "test_runs": len(test), "results": {}}
        for f, name in zip(forecasters, names):
            try:
                f.fit(train)
                preds = [f.predict(ci) for ci, _ in test]
            except (ValueError, RuntimeError) as e:
                out["results"][name] = {"error": str(e)}
                continue
            out["results"][name] = {"params": dict(f.params), **_by_family(preds, [r for _, r in test], weight)}
        return out
    if split != "cv":
        raise ValueError(f"unknown split {split!r}")
    by_fam = defaultdict(set)
    for ci, _ in rows:
        by_fam[ci.family].add(ci.config)
    rng = random.Random(seed)
    per_repeat = defaultdict(list)          # forecaster -> [held-out metrics over all families, one per repeat]
    errors = {}
    for _ in range(repeats):
        fold_of = {}
        for fam in sorted(by_fam):
            cfgs = sorted(by_fam[fam])
            rng.shuffle(cfgs)
            start = rng.randrange(folds)    # so the extra configs of an uneven family do not always land in fold 0
            for i, c in enumerate(cfgs):
                fold_of[(fam, c)] = (start + i) % folds
        preds = {name: [None] * len(rows) for name in names if name not in errors}
        for k in range(folds):
            train = [(ci, r) for ci, r in rows if fold_of[ci.key] != k]
            held = [i for i, (ci, _) in enumerate(rows) if fold_of[ci.key] == k]
            for f, name in zip(forecasters, names):
                if name in errors:
                    continue
                try:
                    f.fit(train)
                    cache = {}
                    for i in held:
                        ci = rows[i][0]
                        key = (ci.key, ci.horizon)
                        if key not in cache:
                            cache[key] = f.predict(ci)
                        preds[name][i] = cache[key]
                except (ValueError, RuntimeError) as e:
                    errors[name] = str(e)
                    preds.pop(name, None)
        for name, ps in preds.items():
            per_repeat[name].append(_by_family(ps, [r for _, r in rows], weight))
    out = {"split": "cv", "weight": weight, "folds": folds, "repeats": repeats, "seed": seed, "runs": len(rows),
           "results": {}}
    base = per_repeat["lottery"]
    for name, err in errors.items():
        out["results"][name] = {"error": err}
    for name, reps in per_repeat.items():
        if name in errors:
            continue
        res = {}
        for fam in reps[0]:
            vals = defaultdict(list)
            for rep in reps:
                for m, v in rep[fam].items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        vals[m].append(v)
            res[fam] = {m: _mean_sd(v) for m, v in vals.items()}
        if name != "lottery":
            wins = [rep["all"]["ll"] > b["all"]["ll"] for rep, b in zip(reps, base)]
            res["all"]["beats_lottery_ll"] = sum(wins) / len(wins)
        out["results"][name] = res
    return out


def _mean_sd(v):
    m = sum(v) / len(v)
    return {"mean": m, "sd": math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else 0.0}


# --- report ------------------------------------------------------------------------------------------------------

def _fmt(v, nd=3):
    if v is None:
        return "–"
    if isinstance(v, dict) and "mean" in v:
        return f"{v['mean']:.{nd}f} ± {v['sd']:.{nd}f}"
    if isinstance(v, dict):
        return json.dumps({k: (round(x, 3) if isinstance(x, float) else x) for k, x in v.items()}, default=str)
    if isinstance(v, bool):
        return str(v)
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def report(res: dict) -> str:
    lines = []
    if res["split"] == "cv":
        lines.append(f"split: cv, {res['folds']}-fold by config, stratified by family, {res['repeats']} repeats "
                     f"(seed {res['seed']}), {res['runs']} runs; mean ± sd over repeats of the held-out totals")
    else:
        lines.append(f"split: {res['split']}, fitted on {res['train_runs']} runs, scored on {res['test_runs']}"
                     + (" (IN-SAMPLE)" if res["split"] == "none" else ""))
    cols = ["n", "configs", "ll", "brier", "auroc", "expected", "observed", "mae", "rmse", "spearman"]
    lines.append("")
    lines.append("| forecaster | family | " + " | ".join(cols) + " |")
    lines.append("|---|---|" + "---|" * len(cols))
    for name, r in res["results"].items():
        if "error" in r:
            lines.append(f"| {name} | NOT SCORED: {r['error']} |" + " |" * (len(cols) - 1))
            continue
        single = len([k for k in r if k not in ("params", "all")]) == 1
        for fam, m in r.items():
            if fam == "params" or not isinstance(m, dict) or (single and fam == "all"):
                continue
            nd = {"ll": 1, "expected": 1, "observed": 1, "n": 0, "configs": 0}
            lines.append(f"| {name} | {fam} | " + " | ".join(_fmt(m.get(c), nd.get(c, 3)) for c in cols) + " |")
        if r.get("params"):
            lines.append(f"| {name} | params: {', '.join(f'{k}={_fmt(v)}' for k, v in r['params'].items())} |"
                         + " |" * (len(cols) - 1))
        beat = r.get("all", {}).get("beats_lottery_ll")
        if beat is not None:
            lines.append(f"| {name} | beats the lottery's log-likelihood in {beat:.0%} of repeats |" + " |" * (len(cols) - 1))
    return "\n".join(lines)
