"""The evaluation module: inputs, labels, forecasters and the scorer. Runs without the Hub; the reproduction check at
the end needs a local copy of the dataset's families/ directory in HIB_HUB_DIR and is skipped otherwise."""
from __future__ import annotations

import itertools
import math
import os
import random

import pytest

from hack_ignition_benchmark import evaluation as ev
from hack_ignition_benchmark.evaluation.forecasters import _phi, tail_prob


def _ci(family, config, *shares, n=100):
    return ev.ConfigInputs(family, config, "test", "synthetic", [ev.Item(s, 1 - s, n) for s in shares])


def _run(family, config, ignited, seed=0, y=None, intervention=False):
    return ev.Run(family, config, f"{config}_s{seed}", seed, (1.0 if ignited else 0.0) if y is None else y,
                  ignited, 150, 150, intervention)


# --- inputs ----------------------------------------------------------------------------------------------------

def test_shipped_inputs_load():
    pool = ev.load_inputs("pool")
    fams = {}
    for (f, _), ci in pool.items():
        fams[f] = fams.get(f, 0) + 1
        for it in ci.items:
            assert it.share is None or 0 <= it.share <= 1
    assert fams == {"single_prompt": 42, "mbpp": 15, "djinn_v2": 45}
    assert all(len(ci.items) == 1 for (f, _), ci in pool.items() if f == "single_prompt")
    k512 = ev.load_inputs("resample_k512")
    assert len(k512) == 42 and set(c for _, c in k512) >= set(ev.CANONICAL["single_prompt"])
    assert all(len(ci.items) == 1 and ci.items[0].n == 512 for ci in k512.values())
    assert ev.measurements() == {"pool", "resample_k512"}
    with pytest.raises(KeyError):
        ev.load_inputs("nope")


def test_item_share():
    assert ev.Item(0.1, 0.3).share == pytest.approx(0.25)
    assert ev.Item(0.0, 0.0).share is None and ev.Item(None, 0.2).share is None


# --- labels ------------------------------------------------------------------------------------------------------

def test_label_uses_the_tail_before_the_horizon():
    hack = [0.0] * 130 + [1.0] * 20 + [0.0] * 100
    rec = {"series": {"hack": hack, "steps": list(range(250))}, "recipe": {"max_steps": 150}}
    assert ev.label(rec) == (1.0, True, 150, 150)
    y, ign, H, n = ev.label(rec, horizon=250)
    assert (y, ign, H, n) == (0.0, False, 250, 250)


def test_label_with_missing_steps_uses_the_logged_ones():
    steps = list(range(32, 150))                 # a run whose first 32 per-step lines were not logged
    rec = {"series": {"hack": [1.0] * len(steps), "steps": steps}, "recipe": {"max_steps": 150}}
    _, ign, _, n = ev.label(rec)
    assert ign and n == 118


# --- forecasters ---------------------------------------------------------------------------------------------------

def test_tail_prob_exact_matches_enumeration():
    rng = random.Random(1)
    ps = [rng.random() for _ in range(7)]
    for k in range(9):
        brute = sum(math.prod(p if b else 1 - p for p, b in zip(ps, bits))
                    for bits in itertools.product([0, 1], repeat=7) if sum(bits) >= k)
        assert tail_prob(ps, k) == pytest.approx(brute)


def test_tail_prob_normal_branch_is_close_to_exact():
    ps = [0.2 + 0.4 * (i % 5) / 4 for i in range(65)]
    mu, var = sum(ps), sum(p * (1 - p) for p in ps)
    k = round(mu)
    assert tail_prob(ps, k) == pytest.approx(1 - _phi((k - 0.5 - mu) / math.sqrt(var)))
    assert abs(tail_prob(ps, k) - tail_prob(ps[:64], round(sum(ps[:64])))) < 0.1


def test_lottery_is_the_share_for_one_problem():
    f = ev.Lottery()
    assert f.predict(_ci("single_prompt", "a", 0.3)) == pytest.approx((0.3, 0.3))
    nothing_rewarded = ev.ConfigInputs("x", "b", "t", "s", [ev.Item(0.0, 0.0)])
    assert f.predict(nothing_rewarded) == (0.0, 0.0)
    many = _ci("mbpp", "c", *([0.5] * 40))
    assert f.predict(many) == pytest.approx((0.5, 0.5))                   # coupled: the run goes one way as a whole
    indep = ev.Lottery(aggregate="independent").predict(many)
    assert indep.rate == pytest.approx(0.5) and indep.p_ignite > 0.99     # 40 coin flips, need 10 heads
    with pytest.raises(ValueError):
        ev.Lottery(aggregate="nope")


def test_step_preset_and_lottery_shape():
    s = ev.make("step:sp-2026-10-21")
    assert s.params["q_star"] == pytest.approx(0.3123, abs=1e-4) and s.params["preset"] == "sp-2026-10-21"
    assert s.predict(_ci("single_prompt", "a", 0.3123)).p_ignite == pytest.approx(0.5, abs=1e-3)
    assert s.fit([]) is s                         # a preset is never refitted
    with pytest.raises(ValueError):
        ev.make("step").fit([])


def test_step_fit_recovers_a_threshold():
    rng = random.Random(0)
    examples = []
    for i in range(400):
        sh = rng.uniform(0.02, 0.9)
        examples.append((_ci("single_prompt", f"p{i}", sh), _run("single_prompt", f"p{i}", sh > 0.3)))
    f = ev.make("step").fit(examples)
    assert f.q_star == pytest.approx(0.3, abs=0.03) and f.sigma < 0.3


def test_per_family_fits_each_family_separately():
    ex = [(_ci("A", f"a{i}", s), _run("A", f"a{i}", s > 0.2)) for i, s in enumerate([0.05, 0.1, 0.15, 0.25, 0.3, 0.4] * 5)]
    ex += [(_ci("B", f"b{i}", s), _run("B", f"b{i}", s > 0.6)) for i, s in enumerate([0.3, 0.4, 0.5, 0.7, 0.8, 0.9] * 5)]
    f = ev.make("per_family:step").fit(ex)
    assert f.by_family["A"].q_star < 0.3 < 0.5 < f.by_family["B"].q_star
    with pytest.raises(ValueError):
        f.predict(_ci("C", "c", 0.5))


def test_step_fit_flags_a_degenerate_optimum():
    # every run ignites: the likelihood keeps rising towards q* -> 0, so the optimum is on the grid's edge
    ex = [(_ci("A", f"a{i}", 0.1 + 0.05 * i), _run("A", f"a{i}", True)) for i in range(10)]
    f = ev.make("step").fit(ex)
    assert f.params.get("degenerate") is True


def test_make_passes_ignite_and_aggregate_to_custom_forecasters(monkeypatch):
    import types, sys
    mod = types.ModuleType("custom_fc")

    class Mine(ev.Lottery):
        name = "mine"
    mod.Mine = Mine
    monkeypatch.setitem(sys.modules, "custom_fc", mod)
    m = ev.make("custom_fc:Mine", ignite=0.4, aggregate="independent")
    assert (m.ignite, m.aggregate) == (0.4, "independent")


def test_make_rejects_unknown():
    with pytest.raises(KeyError):
        ev.make("nope")
    with pytest.raises(KeyError):
        ev.make("step:nope")


# --- scoring -----------------------------------------------------------------------------------------------------

def test_join_skips_interventions_and_configs_without_inputs():
    inputs = {("single_prompt", "a"): _ci("single_prompt", "a", 0.3)}
    runs = [_run("single_prompt", "a", True), _run("single_prompt", "a", False, intervention=True),
            _run("single_prompt", "z", True)]
    rows, skipped = ev.join(runs, inputs)
    assert len(rows) == 1 and skipped == {"intervention": 1, "no inputs (single_prompt)": 1}
    assert len(ev.join(runs, inputs, include_interventions=True)[0]) == 2


def test_join_skips_runs_that_stopped_early_and_carries_the_horizon():
    inputs = {("f", "a"): _ci("f", "a", 0.3)}
    stopped = ev.Run("f", "a", "a_s1", 1, 0.1, False, 1000, 296, False, True)
    rows, skipped = ev.join([_run("f", "a", True), stopped], inputs)
    assert len(rows) == 1 and rows[0][0].horizon == 150 and inputs[("f", "a")].horizon is None
    assert list(skipped.values()) == [1] and "stopped" in next(iter(skipped))


def test_a_forecaster_that_cannot_fit_is_reported_not_fatal():
    rows = [(_ci("single_prompt", "new", 0.3), _run("single_prompt", "new", True))]   # no first-round runs to fit on
    res = ev.evaluate([ev.make("step"), ev.make("per_family:step")], rows, split="canonical")
    assert "error" in res["results"]["step"] and res["results"]["lottery"]["single_prompt"]["n"] == 1
    assert "NOT SCORED" in ev.report(res)


def test_report_handles_per_family_params():
    first = ev.CANONICAL["single_prompt"]
    rows = [(_ci("single_prompt", c, 0.1 + 0.05 * i), _run("single_prompt", c, i % 2 == 0)) for i, c in enumerate(first)]
    rows += [(_ci("single_prompt", "new", 0.3), _run("single_prompt", "new", True))]
    assert "single_prompt" in ev.report(ev.evaluate([ev.make("per_family:step")], rows, split="canonical"))


def test_metrics():
    preds = [ev.Forecast(0.9, 0.9), ev.Forecast(0.2, 0.2), ev.Forecast(0.6, 0.6)]
    runs = [_run("f", "a", True), _run("f", "b", False), _run("f", "c", True)]
    m = ev.metrics(preds, runs)
    assert m["auroc"] == 1.0 and m["observed"] == 2 and m["expected"] == pytest.approx(1.7)
    assert m["ll"] == pytest.approx(math.log(0.9) + math.log(0.8) + math.log(0.6))
    assert m["brier"] == pytest.approx((0.01 + 0.04 + 0.16) / 3)


def test_config_weighting_counts_each_config_once():
    preds = [ev.Forecast(0.9, 0.9)] * 4 + [ev.Forecast(0.2, 0.2)]
    runs = [_run("f", "a", True, seed=s) for s in range(4)] + [_run("f", "b", False)]
    by_run, by_cfg = ev.metrics(preds, runs, "run"), ev.metrics(preds, runs, "config")
    assert by_run["ll"] == pytest.approx(4 * math.log(0.9) + math.log(0.8))
    assert by_cfg["ll"] == pytest.approx(math.log(0.9) + math.log(0.8))
    assert by_cfg["brier"] == pytest.approx((0.01 + 0.04) / 2) and by_cfg["observed"] == 4
    f = ev.make("base_rate").fit([(_ci("f", r.config, 0.5), r) for r in runs])
    assert f.p == pytest.approx(0.5)                                # 1 of 2 configs ignited, not 4 of 5 runs
    assert ev.make("base_rate", weight="run").fit([(_ci("f", r.config, 0.5), r) for r in runs]).p == pytest.approx(0.8)


def test_canonical_split_fits_on_the_first_round_only():
    seen = []

    class Spy(ev.Lottery):
        name = "spy"

        def fit(self, examples):
            seen.extend(ci.config for ci, _ in examples)
            return self

    first = ev.CANONICAL["single_prompt"][0]
    rows = [(_ci("single_prompt", first, 0.3), _run("single_prompt", first, True)),
            (_ci("single_prompt", "new", 0.3), _run("single_prompt", "new", False))]
    res = ev.evaluate([Spy()], rows, split="canonical")
    assert set(seen) == {first} and res["test_runs"] == 1
    assert "lottery" in res["results"]                       # always scored alongside
    with pytest.raises(ValueError):
        ev.evaluate([Spy()], [(_ci("mbpp", "x", 0.3), _run("mbpp", "x", True))], split="canonical")


def test_cv_never_splits_a_config():
    class Spy(ev.Lottery):
        name = "spy"
        trained: set = set()

        def fit(self, examples):
            self.trained = {ci.key for ci, _ in examples}
            return self

        def predict(self, inputs):
            assert inputs.key not in self.trained
            return super().predict(inputs)

    rows = [(_ci(f, f"{f}{i}", 0.1 * (i % 9) + 0.05), _run(f, f"{f}{i}", i % 3 == 0, seed=s))
            for f in ("A", "B") for i in range(12) for s in range(3)]
    res = ev.evaluate([Spy()], rows, split="cv", folds=4, repeats=3)
    assert res["results"]["spy"]["all"]["n"]["mean"] == len(rows)


# --- reproduction against the published runs -----------------------------------------------------------------------

HUB_DIR = os.environ.get("HIB_HUB_DIR")


@pytest.mark.skipif(not HUB_DIR, reason="set HIB_HUB_DIR to a local copy of the dataset's families/ directory")
def test_reproduces_the_single_prompt_test_set_scores():
    runs = ev.load_runs(["single_prompt"], hub_dir=HUB_DIR)
    for measurement, want in (("pool", {"lottery": -31.0, "step:sp-2026-10-21": -30.9}),
                              ("resample_k512", {"lottery": -29.1, "step:sp-2026-10-21": -26.1})):
        rows, _ = ev.join(runs, ev.load_inputs(measurement))
        res = ev.evaluate([ev.make("step:sp-2026-10-21")], rows, split="canonical", weight="run")
        for name, ll in want.items():
            assert round(res["results"][name]["single_prompt"]["ll"], 1) == ll, (measurement, name)
