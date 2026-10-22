"""Forecasters: start-model inputs of a config -> P(a run of it ignites) and its expected final hack rate.

A forecaster is any object with

    name: str
    fit(examples) -> self          examples: [(ConfigInputs, Run), ...] from the training split (may ignore them)
    predict(inputs) -> Forecast    inputs: ConfigInputs

so a new one can be scored with `hib eval --forecaster mypkg.mymodule:MyForecaster`.

The reference forecasters give each problem of the item file a probability p_i of ending hacked; the run's expected
final hack rate is the mean of the p_i (problems are visited equally often). How the problems combine into P(ignite)
is `aggregate`:

  coupled      (default) the run ignites or not as a whole, as runs do (problems share one set of weights), so
               P(ignite) = the expected final hack rate = mean p_i.
  independent  each problem ignites on its own, so P(ignite) = P(the fraction of hacked problems >= the ignition
               threshold), a Poisson-binomial tail. With 100+ problems this is nearly 0 or 1, and on the published
               multi-prompt runs it scores far worse than `coupled`; kept for comparison.

A single_prompt config has one problem, so both rules give p_1.

  lottery   p_i = the problem's hack share, hack / (hack + honest); 0 where nothing was rewarded. No parameters.
            The identity-kernel limit of the toy model: every rewarded completion is reinforced alike, so the hack
            wins in proportion to its share of the rewarded completions.
  step      p_i = Phi((logit share_i - logit q*) / sigma): a threshold at hack share q* with probit width sigma.
            `fit` chooses q* and sigma by maximum likelihood (a grid, standard library only). The lottery is close to
            q* = .5 with sigma ~ 1.7 (it is exactly a logistic of width 1 at .5).
  base_rate P(ignite) = the training split's ignition rate, for every config; rate = its mean final hack rate
            (two parameters).
  per_family:<spec>  one copy of <spec> per family, each fitted on that family's training runs only.

Named presets fix parameters fitted elsewhere, e.g. `step:sp-2026-10-21`.
"""
from __future__ import annotations

import math
from typing import NamedTuple

from .inputs import ConfigInputs


class Forecast(NamedTuple):
    p_ignite: float
    rate: float


CLIP = 0.01      # probabilities are clipped to [CLIP, 1 - CLIP] when scored
_EPS = 1e-6


def _logit(q: float) -> float:
    q = min(max(q, _EPS), 1 - _EPS)
    return math.log(q / (1 - q))


def _phi(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def tail_prob(ps: list[float], k: int) -> float:
    """P(at least k of independent Bernoulli(p_i) succeed). Exact for up to 64 terms, else a normal approximation
    with continuity correction (the sum of 64+ bounded terms)."""
    n = len(ps)
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    if n <= 64:
        dist = [1.0] + [0.0] * n
        for p in ps:
            for j in range(n, 0, -1):
                dist[j] = dist[j] * (1 - p) + dist[j - 1] * p
            dist[0] *= 1 - p
        return min(1.0, max(0.0, sum(dist[k:])))
    mu = sum(ps)
    var = sum(p * (1 - p) for p in ps)
    if var <= 0:
        return 1.0 if mu >= k else 0.0
    return 1 - _phi((k - 0.5 - mu) / math.sqrt(var))


AGGREGATES = ("coupled", "independent")
WEIGHTS = ("config", "run")


def weights(runs, mode: str = "config") -> list[float]:
    """Per-run weights. "config": each config counts once (its seeds share the weight), because seeds of one config
    mostly go the same way and are not independent evidence. "run": every run counts once."""
    if mode == "run":
        return [1.0] * len(runs)
    if mode != "config":
        raise ValueError(f"weight must be one of {WEIGHTS}, got {mode!r}")
    n = {}
    for r in runs:
        n[(r.family, r.config)] = n.get((r.family, r.config), 0) + 1
    return [1.0 / n[(r.family, r.config)] for r in runs]


def combine(ps: list[float], ignite: float, aggregate: str = "coupled") -> Forecast:
    """Per-problem p_i -> Forecast(P(ignite), mean p_i)."""
    if not ps:
        return Forecast(0.0, 0.0)
    n = len(ps)
    rate = sum(ps) / n
    if aggregate == "coupled":
        return Forecast(rate, rate)
    if aggregate != "independent":
        raise ValueError(f"aggregate must be one of {AGGREGATES}, got {aggregate!r}")
    return Forecast(tail_prob(ps, math.ceil(ignite * n - 1e-9)), rate)


class Lottery:
    name = "lottery"

    def __init__(self, ignite: float = 0.25, aggregate: str = "coupled", weight: str = "config"):
        if aggregate not in AGGREGATES:
            raise ValueError(f"aggregate must be one of {AGGREGATES}, got {aggregate!r}")
        self.ignite, self.aggregate, self.weight = ignite, aggregate, weight

    def fit(self, examples):
        return self

    def problem_p(self, share):
        return 0.0 if share is None else share

    def predict(self, inputs: ConfigInputs) -> Forecast:
        return combine([self.problem_p(it.share) for it in inputs.items], self.ignite, self.aggregate)

    @property
    def params(self) -> dict:
        return {}


class Step(Lottery):
    name = "step"

    def __init__(self, q_star: float | None = None, sigma: float | None = None, ignite: float = 0.25,
                 preset: str | None = None, aggregate: str = "coupled", weight: str = "config"):
        super().__init__(ignite, aggregate, weight)
        self.q_star, self.sigma, self.preset = q_star, sigma, preset
        self.degenerate = False

    def problem_p(self, share, q_star=None, sigma=None):
        q, sg = (self.q_star, self.sigma) if q_star is None else (q_star, sigma)
        if q is None or sg is None:
            raise RuntimeError("step forecaster has no parameters: fit() it or use a preset (step:<name>)")
        if share is None or share <= 0:
            return 0.0
        if share >= 1:
            return 1.0
        return _phi((_logit(share) - _logit(q)) / sg)

    def _predict_at(self, inputs, q, sg) -> Forecast:
        return combine([self.problem_p(it.share, q, sg) for it in inputs.items], self.ignite, self.aggregate)

    def fit(self, examples):
        if self.preset:          # a preset's parameters are fixed
            return self
        if not examples:
            raise ValueError("step: no training runs to fit q* and sigma on (use a preset, e.g. step:sp-2026-10-21, "
                             "or a split whose training side has inputs)")
        # logit q* in [-4, 4], log sigma in [-2.5, 1.5]: a coarse grid, then a finer one around the best point
        coarse = {pt: self._ll_at(*pt, examples) for pt in _grid(-4.0, 4.0, 0.5, -2.5, 1.5, 0.5)}
        zq0, ls0 = max(coarse, key=lambda pt: (coarse[pt], -abs(pt[0]), -abs(pt[1])))
        fine = {pt: self._ll_at(*pt, examples) for pt in _grid(zq0 - 0.5, zq0 + 0.5, 0.05, ls0 - 0.5, ls0 + 0.5, 0.05)}
        zq, ls = max(fine, key=lambda pt: (fine[pt], -abs(pt[0] - zq0), -abs(pt[1] - ls0)))
        best = fine[(zq, ls)]
        # degenerate: the best coarse point is on the grid's edge, or the likelihood is flat around it (every forecast
        # clipped), so q* and sigma are not identified by these runs
        on_edge = zq0 in (-4.0, 4.0) or ls0 in (-2.5, 1.5)
        flat = sum(1 for v in coarse.values() if abs(v - best) < 1e-9) > 1
        self.q_star, self.sigma, self.degenerate = 1 / (1 + math.exp(-zq)), math.exp(ls), on_edge or flat
        return self

    def _ll_at(self, zq, ls, examples):
        q, sg = 1 / (1 + math.exp(-zq)), math.exp(ls)
        cache, ll = {}, 0.0
        for (ci, run), w in zip(examples, weights([r for _, r in examples], self.weight)):
            if ci.key not in cache:
                cache[ci.key] = self._predict_at(ci, q, sg).p_ignite
            p = min(max(cache[ci.key], CLIP), 1 - CLIP)
            ll += w * math.log(p if run.ignited else 1 - p)
        return ll

    @property
    def params(self) -> dict:
        out = {"q_star": self.q_star, "sigma": self.sigma}
        if self.preset:
            out["preset"] = self.preset
        if self.degenerate:
            out["degenerate"] = True
        return out


def _grid(q_lo, q_hi, q_step, s_lo, s_hi, s_step):
    nq, ns = round((q_hi - q_lo) / q_step), round((s_hi - s_lo) / s_step)
    for i in range(nq + 1):
        for j in range(ns + 1):
            yield q_lo + i * q_step, s_lo + j * s_step


class BaseRate:
    name = "base_rate"

    def __init__(self, ignite: float = 0.25, aggregate: str = "coupled", weight: str = "config"):
        self.p, self.rate, self.weight = 0.5, 0.5, weight

    def fit(self, examples):
        if examples:
            w = weights([r for _, r in examples], self.weight)
            self.p = sum(wi * r.ignited for wi, (_, r) in zip(w, examples)) / sum(w)
            self.rate = sum(wi * r.y for wi, (_, r) in zip(w, examples)) / sum(w)
        return self

    def predict(self, inputs: ConfigInputs) -> Forecast:
        return Forecast(self.p, self.rate)

    @property
    def params(self) -> dict:
        return {"p": self.p, "rate": self.rate}


class PerFamily:
    """A separate copy of a forecaster for each family, each fitted on its own family's training runs."""

    def __init__(self, spec: str, ignite: float = 0.25, aggregate: str = "coupled", weight: str = "config"):
        self.spec, self.ignite, self.aggregate, self.weight, self.by_family = spec, ignite, aggregate, weight, {}
        self.name = f"per_family:{spec}"

    def fit(self, examples):
        fams = sorted({ci.family for ci, _ in examples})
        self.by_family = {f: make(self.spec, self.ignite, self.aggregate, self.weight)
                          .fit([e for e in examples if e[0].family == f]) for f in fams}
        return self

    def predict(self, inputs: ConfigInputs) -> Forecast:
        if inputs.family not in self.by_family:
            raise ValueError(f"{self.name}: no training runs from family {inputs.family!r}")
        return self.by_family[inputs.family].predict(inputs)

    @property
    def params(self) -> dict:
        return {f: m.params for f, m in self.by_family.items()}


# Parameters fitted elsewhere, fixed here. sp-2026-10-21: fitted by maximum likelihood on the 16 first-round
# single_prompt problems (seeds 0-3, pool shares) and fixed before the 26 test-set problems were run.
PRESETS = {
    "sp-2026-10-21": dict(q_star=0.3122589676340247, sigma=0.6748941683109781),
}


def make(spec: str, ignite: float = 0.25, aggregate: str = "coupled", weight: str = "config"):
    """`lottery`, `step`, `step:<preset>`, `base_rate`, `per_family:<spec>`, or `module.path:Name` (a class, called
    with ignite=, aggregate= and weight= where it accepts them, or an instance)."""
    if spec.startswith("per_family:"):
        return PerFamily(spec.split(":", 1)[1], ignite, aggregate, weight)
    if spec == "lottery":
        return Lottery(ignite, aggregate, weight)
    if spec == "step":
        return Step(ignite=ignite, aggregate=aggregate, weight=weight)
    if spec.startswith("step:"):
        name = spec.split(":", 1)[1]
        if name not in PRESETS:
            raise KeyError(f"unknown step preset {name!r} (have {sorted(PRESETS)})")
        return Step(ignite=ignite, preset=name, aggregate=aggregate, weight=weight, **PRESETS[name])
    if spec == "base_rate":
        return BaseRate(ignite, aggregate, weight)
    if ":" in spec:
        import importlib
        import inspect
        mod, _, attr = spec.partition(":")
        obj = getattr(importlib.import_module(mod), attr)
        if not isinstance(obj, type):
            return obj
        accepted = inspect.signature(obj).parameters
        kw = {k: v for k, v in (("ignite", ignite), ("aggregate", aggregate), ("weight", weight)) if k in accepted}
        return obj(**kw)
    raise KeyError(f"unknown forecaster {spec!r}")
