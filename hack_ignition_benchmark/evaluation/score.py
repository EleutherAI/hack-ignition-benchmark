"""Score forecasters against the published runs.

Unit: a run is a scored row; a config's runs share one forecast (its seeds are replicates). Splits never separate
the seeds of a config.

Metrics, per family and over all scored runs:
  binary (ignited):  log-likelihood with P clipped to [.01, .99] (summed over runs), Brier, AUROC,
                     expected vs observed ignitions
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

import math
import random
from collections import defaultdict

from .forecasters import CLIP, Lottery
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
    """[(ConfigInputs, Run)] for the runs that have inputs; plus what was left out and why."""
    rows, skipped = [], defaultdict(int)
    for r in runs:
        if r.intervention and not include_interventions:
            skipped["intervention"] += 1
        elif (r.family, r.config) not in inputs:
            skipped[f"no inputs ({r.family})"] += 1
        elif r.y != r.y:
            skipped["no logged steps before the horizon"] += 1
        else:
            rows.append((inputs[(r.family, r.config)], r))
    return rows, dict(skipped)


# --- metrics -----------------------------------------------------------------------------------------------------

def _ll(p, y):
    p = min(max(p, CLIP), 1 - CLIP)
    return math.log(p if y else 1 - p)


def auroc(scores, labels):
    """Mann-Whitney AUROC, ties counted as half; None if one class is absent."""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    win = 0.0
    for a in pos:
        for b in neg:
            win += 1.0 if a > b else 0.5 if a == b else 0.0
    return win / (len(pos) * len(neg))


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


def metrics(preds, runs) -> dict:
    """preds: [Forecast] aligned with runs."""
    ps, rates = [f.p_ignite for f in preds], [f.rate for f in preds]
    ys, yr = [r.ignited for r in runs], [r.y for r in runs]
    n = len(runs)
    if not n:
        return {"n": 0}
    return {
        "n": n, "configs": len({(r.family, r.config) for r in runs}),
        "ll": sum(_ll(p, y) for p, y in zip(ps, ys)),
        "brier": sum((p - y) ** 2 for p, y in zip(ps, ys)) / n,
        "auroc": auroc(ps, ys),
        "expected": sum(ps), "observed": sum(ys),
        "mae": sum(abs(a - b) for a, b in zip(rates, yr)) / n,
        "rmse": math.sqrt(sum((a - b) ** 2 for a, b in zip(rates, yr)) / n),
        "spearman": spearman(rates, yr),
    }


def _by_family(preds, runs) -> dict:
    out = {"all": metrics(preds, runs)}
    fams = sorted({r.family for r in runs})
    if len(fams) == 1:
        return {fams[0]: out["all"], "all": out["all"]}
    for fam in fams:
        idx = [i for i, r in enumerate(runs) if r.family == fam]
        out[fam] = metrics([preds[i] for i in idx], [runs[i] for i in idx])
    return out


# --- evaluation --------------------------------------------------------------------------------------------------

def _with_lottery(forecasters, ignite):
    if not any(getattr(f, "name", None) == "lottery" for f in forecasters):
        forecasters = [Lottery(ignite)] + list(forecasters)
    return forecasters


def _label(f) -> str:
    preset = getattr(f, "preset", None)
    return f"{f.name}:{preset}" if preset else f.name


def evaluate(forecasters, rows, split: str = "canonical", folds: int = 5, repeats: int = 20, seed: int = 0,
             ignite: float = 0.25) -> dict:
    forecasters = _with_lottery(forecasters, ignite)
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
        out = {"split": split, "train_runs": len(train), "test_runs": len(test), "results": {}}
        for f in forecasters:
            f.fit(train)
            preds = [f.predict(ci) for ci, _ in test]
            out["results"][_label(f)] = {"params": dict(f.params), **_by_family(preds, [r for _, r in test])}
        return out
    if split != "cv":
        raise ValueError(f"unknown split {split!r}")
    by_fam = defaultdict(set)
    for ci, _ in rows:
        by_fam[ci.family].add(ci.config)
    rng = random.Random(seed)
    per_repeat = defaultdict(list)          # forecaster -> [held-out metrics over all families, one per repeat]
    for _ in range(repeats):
        fold_of = {}
        for fam in sorted(by_fam):
            cfgs = sorted(by_fam[fam])
            rng.shuffle(cfgs)
            for i, c in enumerate(cfgs):
                fold_of[(fam, c)] = i % folds
        preds = {_label(f): [None] * len(rows) for f in forecasters}
        for k in range(folds):
            train = [(ci, r) for ci, r in rows if fold_of[ci.key] != k]
            held = [i for i, (ci, _) in enumerate(rows) if fold_of[ci.key] == k]
            for f in forecasters:
                f.fit(train)
                cache = {}
                for i in held:
                    ci = rows[i][0]
                    if ci.key not in cache:
                        cache[ci.key] = f.predict(ci)
                    preds[_label(f)][i] = cache[ci.key]
        for name, ps in preds.items():
            per_repeat[name].append(_by_family(ps, [r for _, r in rows]))
    out = {"split": "cv", "folds": folds, "repeats": repeats, "seed": seed, "runs": len(rows), "results": {}}
    base = per_repeat["lottery"]
    for name, reps in per_repeat.items():
        res = {}
        for fam in reps[0]:
            vals = defaultdict(list)
            for rep in reps:
                for m, v in rep[fam].items():
                    if isinstance(v, (int, float)) and v is not None:
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
    if isinstance(v, dict):
        return f"{v['mean']:.{nd}f} ± {v['sd']:.{nd}f}"
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
