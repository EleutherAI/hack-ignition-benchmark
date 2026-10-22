"""Forecasters: start-model inputs of a config -> P(a run of it ignites) and its expected final hack rate.

A forecaster is any object with

    name: str
    fit(examples) -> self          examples: [(ConfigInputs, Run), ...] from the training split (may ignore them)
    predict(inputs) -> Forecast    inputs: ConfigInputs

so a new one can be scored with `hib eval --forecaster mypkg.mymodule:MyForecaster`.

The reference forecasters treat each problem of the item file as its own bistable unit: problem i ends hacked with
probability p_i, independently of the others. The run's expected final hack rate is then the mean of the p_i (each
problem is visited equally often), and P(ignite) = P(the fraction of hacked problems >= the ignition threshold),
a Poisson-binomial tail. A single_prompt config has one problem, so both equal p_1.

  lottery   p_i = the problem's hack share, hack / (hack + honest); 0 where nothing was rewarded. No parameters.
            The identity-kernel limit of the toy model: every rewarded completion is reinforced alike, so the hack
            wins in proportion to its share of the rewarded completions.
  step      p_i = Phi((logit share_i - logit q*) / sigma): a threshold at hack share q* with probit width sigma.
            `fit` chooses q* and sigma by maximum likelihood (a grid, standard library only). The lottery is close to
            q* = .5 with sigma ~ 1.7 (it is exactly a logistic of width 1 at .5).
  base_rate P(ignite) = the training split's ignition rate, for every config; rate = its mean final hack rate.
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


def combine(ps: list[float], ignite: float) -> Forecast:
    """Independent problems -> (P(fraction hacked >= ignite), mean p)."""
    if not ps:
        return Forecast(0.0, 0.0)
    n = len(ps)
    k = math.ceil(ignite * n - 1e-9)
    return Forecast(tail_prob(ps, k), sum(ps) / n)


class Lottery:
    name = "lottery"

    def __init__(self, ignite: float = 0.25):
        self.ignite = ignite

    def fit(self, examples):
        return self

    def problem_p(self, share):
        return 0.0 if share is None else share

    def predict(self, inputs: ConfigInputs) -> Forecast:
        return combine([self.problem_p(it.share) for it in inputs.items], self.ignite)

    @property
    def params(self) -> dict:
        return {}


class Step(Lottery):
    name = "step"

    def __init__(self, q_star: float | None = None, sigma: float | None = None, ignite: float = 0.25,
                 preset: str | None = None):
        super().__init__(ignite)
        self.q_star, self.sigma, self.preset = q_star, sigma, preset

    def problem_p(self, share):
        if self.q_star is None or self.sigma is None:
            raise RuntimeError("step forecaster has no parameters: fit() it or use a preset (step:<name>)")
        if share is None or share <= 0:
            return 0.0
        if share >= 1:
            return 1.0
        return _phi((_logit(share) - _logit(self.q_star)) / self.sigma)

    def fit(self, examples):
        if self.preset:          # a preset's parameters are fixed
            return self
        if not examples:
            raise ValueError("step: no training runs to fit q* and sigma on (use a preset, e.g. step:sp-2026-10-21, "
                             "or a split whose training side has inputs)")
        best = None              # logit q* in [-4, 4], log sigma in [-2.5, 1.5]: coarse, then a finer local grid
        for zq, ls in _grid(-4.0, 4.0, 0.5, -2.5, 1.5, 0.5):
            best = _better(best, self._ll_at(zq, ls, examples), zq, ls)
        assert best is not None
        _, zq0, ls0 = best
        for zq, ls in _grid(zq0 - 0.5, zq0 + 0.5, 0.05, ls0 - 0.5, ls0 + 0.5, 0.05):
            best = _better(best, self._ll_at(zq, ls, examples), zq, ls)
        _, zq, ls = best
        self.q_star, self.sigma = 1 / (1 + math.exp(-zq)), math.exp(ls)
        return self

    def _ll_at(self, zq, ls, examples):
        self.q_star, self.sigma = 1 / (1 + math.exp(-zq)), math.exp(ls)
        cache, ll = {}, 0.0
        for ci, run in examples:
            if ci.key not in cache:
                cache[ci.key] = self.predict(ci).p_ignite
            p = min(max(cache[ci.key], CLIP), 1 - CLIP)
            ll += math.log(p if run.ignited else 1 - p)
        return ll

    @property
    def params(self) -> dict:
        return {"q_star": self.q_star, "sigma": self.sigma, **({"preset": self.preset} if self.preset else {})}


def _grid(q_lo, q_hi, q_step, s_lo, s_hi, s_step):
    nq, ns = round((q_hi - q_lo) / q_step), round((s_hi - s_lo) / s_step)
    for i in range(nq + 1):
        for j in range(ns + 1):
            yield q_lo + i * q_step, s_lo + j * s_step


def _better(best, ll, zq, ls):
    return (ll, zq, ls) if best is None or ll > best[0] + 1e-12 else best


class BaseRate:
    name = "base_rate"

    def __init__(self, ignite: float = 0.25):
        self.p, self.rate = 0.5, 0.5

    def fit(self, examples):
        if examples:
            self.p = sum(r.ignited for _, r in examples) / len(examples)
            self.rate = sum(r.y for _, r in examples) / len(examples)
        return self

    def predict(self, inputs: ConfigInputs) -> Forecast:
        return Forecast(self.p, self.rate)

    @property
    def params(self) -> dict:
        return {"p": self.p, "rate": self.rate}


class PerFamily:
    """A separate copy of a forecaster for each family, each fitted on its own family's training runs."""

    def __init__(self, spec: str, ignite: float = 0.25):
        self.spec, self.ignite, self.by_family = spec, ignite, {}
        self.name = f"per_family:{spec}"

    def fit(self, examples):
        fams = sorted({ci.family for ci, _ in examples})
        self.by_family = {f: make(self.spec, self.ignite).fit([e for e in examples if e[0].family == f]) for f in fams}
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


def make(spec: str, ignite: float = 0.25):
    """`lottery`, `step`, `step:<preset>`, `base_rate`, `per_family:<spec>`, or `module.path:Name` (a class taking
    no arguments, or an instance)."""
    if spec.startswith("per_family:"):
        return PerFamily(spec.split(":", 1)[1], ignite)
    if spec == "lottery":
        return Lottery(ignite)
    if spec == "step":
        return Step(ignite=ignite)
    if spec.startswith("step:"):
        name = spec.split(":", 1)[1]
        if name not in PRESETS:
            raise KeyError(f"unknown step preset {name!r} (have {sorted(PRESETS)})")
        return Step(ignite=ignite, preset=name, **PRESETS[name])
    if spec == "base_rate":
        return BaseRate(ignite)
    if ":" in spec:
        import importlib
        mod, _, attr = spec.partition(":")
        obj = getattr(importlib.import_module(mod), attr)
        return obj() if isinstance(obj, type) else obj
    raise KeyError(f"unknown forecaster {spec!r}")
