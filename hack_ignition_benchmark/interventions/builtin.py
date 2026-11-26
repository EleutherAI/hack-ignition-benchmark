"""The four interventions the published benchmark uses, the share clamp, and GRAFT's reward penalty.

    inject           splice harvested hack completions into a planted problem's generation group
    reward_schedule  switch the grader between exploitable (hacks rewarded) and hardened (hacks score 0)
    opt_reset        clear the Adam state at a step
    monitor          a regex hack detector on the completion text: detected completions score 0 (squash) or a
                     penalty (invert)
    share_clamp      reject a random fraction of the rewarded hacks, or of the clean solutions, to set the hack share
                     the optimiser sees: redraw the rejected completions (holdout) or score them 0 (zero)
    reward_penalty  GRAFT's reward-penalty baseline: subtract a fixed amount from completions the environment's own
                     monitor flags, each flag kept with probability `recall` (the GRAFT toy environments)

The first four are ported from the private repo's rhg/inject.py and 04_rl/grpo_train.py; the parity tests
(tests/test_core.py::test_parity_*) hold the injection plan and the reward modes to exact equality with the old code. reward_penalty is ported from GRAFT
(github.com/jnward/GRAFT @ b0767d1, MIT): `rl/advantages.py` (the reward_penalty_baseline branch) and `rl/train.py`
(the rh_detector_recall wrapper).
"""
from __future__ import annotations

import json
import os
import random
import re
from collections import defaultdict

from .base import Intervention, register_intervention

_MONITOR_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "monitors")


# --- inject --------------------------------------------------------------------------------------------------------

def plan_prob_injection(n_rows, group_size, base_of_group, prob, k, step, seed):
    """Rows to inject under a per-visit probability. A visit is one generation group (`group_size` consecutive rows
    of one prompt). For every group whose prompt matches a planted base, draw once -- deterministic in (seed, step,
    group index), so a run is reproducible and resumable -- and on success inject the group's first `k` rows.
    With k = 1 the injected rate is prob / group_size per completion. Returns [(row index, base)]."""
    out = []
    for g in range(0, n_rows, group_size):
        base = base_of_group(g)
        if base is None:
            continue
        rng = random.Random(f"{seed}:{step}:{g // group_size}")
        if rng.random() < prob:
            out += [(g + j, base) for j in range(min(k, group_size))]
    return out


def load_hacks(path):
    """{base problem id: [completion, ...]} from a JSONL with problem, completion and optionally behavior
    (then only behavior == 'hack' rows are kept)."""
    hacks = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("behavior") is not None and r.get("behavior") != "hack":
                continue
            comp, pid = r.get("completion"), r.get("problem")
            if comp and pid:
                hacks.setdefault(pid, []).append(comp)
    return hacks


def base_id(pid):
    """Map an impossible (`-imp`) or duplicated (`#n`) variant to the origin problem, whose harvested hacks it uses."""
    return pid.split("-imp")[0].split("#")[0]


@register_intervention
class Inject(Intervention):
    """Splice harvested hacks into the generation group of planted problems, before grading and the loss.

    Fields: hacks (file ref), k (per visit or total), deadline (no injection at/after this step), persist (k every
    visit vs k in total), front (put the targets first and keep dataset order), problems (restrict to these full item
    ids; empty = every item whose base has harvested hacks), prob (per-visit probability; overrides the k schedule).

    Splicing completion token ids (not reward-time text) makes the graded text and the loss tokens agree. It is valid
    only when the sampling logprobs are unused by the loss: generation with vLLM and no importance-sampling correction
    (or HF generation). setup() refuses otherwise."""
    name = "inject"

    def __init__(self, hacks, k=1, deadline=60, persist=False, front=False, problems=(), prob=None):
        super().__init__(hacks=hacks, k=k, deadline=deadline, persist=persist, front=front,
                         problems=list(problems), prob=prob)
        self.hacks_ref, self.k, self.deadline = hacks, int(k), int(deadline)
        self.persist, self.front, self.prob = bool(persist), bool(front), prob
        self.only = set(problems) if problems else None

    def prepare_items(self, items, env):
        # front: move the targets to the head so they are in the first generation batch (needs no shuffle)
        if self.front and self.only:
            front = [r for r in items if env.item_id(r) in self.only]
            return front + [r for r in items if env.item_id(r) not in self.only]
        return items

    def shuffle_allowed(self):
        return not (self.front and self.only)

    def setup(self, ctx):
        t = ctx.train
        if t["gen"] == "vllm" and t["is_correction"]:
            raise ValueError("inject needs is_correction = false under vLLM generation: the spliced rows have no valid "
                             "sampling logprob, so the importance-sampling ratio would be undefined")
        self.events, self.seed = ctx.events, ctx.seed
        self.num_generations = t["num_generations"]
        tok, env = ctx.tokenizer, ctx.env
        eos, maxc = tok.eos_token_id, t["max_completion"]

        def encode(text):
            ids = tok(text, add_special_tokens=False).input_ids
            if not ids or ids[-1] != eos:
                ids = ids + [eos]
            return ids[:maxc] if maxc and len(ids) > maxc else ids

        self.hack_ids = {}
        from ..recipe import FileRef
        for b, comps in load_hacks(ctx.data.path(FileRef.parse(self.hacks_ref))).items():
            enc = [e for e in (encode(c) for c in comps) if len(e) > 1]
            if enc:
                self.hack_ids[b] = enc
        self.prompt_keys, self.budget = [], {}
        for r in ctx.items:
            pid = env.item_id(r)
            if self.only is not None and pid not in self.only:
                continue
            b = base_id(pid)
            if b in self.hack_ids:
                self.prompt_keys.append((env.user_text(r), b))
                self.budget.setdefault(b, self.k)
        self.prompt_keys.sort(key=lambda kb: -len(kb[0]))   # longer keys first: a substring cannot shadow a superset
        self.tokenizer = tok
        mode = (f"probabilistic p={self.prob} per visit" if self.prob is not None else
                "persist: k every visit" if self.persist else f"one-shot: {sum(self.budget.values())} in total")
        self.events.emit("inject_index", step=0, bases=sorted(self.budget), mode=mode, k=self.k,
                         deadline=self.deadline, only=sorted(self.only) if self.only else None)

    def match_base(self, prompt_text):
        for user, b in self.prompt_keys:
            if user in prompt_text:
                return b
        return None

    def replace_generations(self, step, prompt_ids, completion_ids, logprobs):
        if step >= self.deadline:
            return completion_ids, logprobs
        if self.prob is None and not self.persist and not any(self.budget.values()):
            return completion_ids, logprobs
        texts = self.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
        per_call, rows = {}, []

        def splice(i, b):
            pool = self.hack_ids[b]
            ids = pool[(step + i) % len(pool)]          # deterministic pick: reproducible and resumable
            completion_ids[i] = list(ids)
            if logprobs is not None:                    # keep the (unused) sampling-logprob rows length-consistent
                logprobs[i] = [0.0] * len(ids)
            per_call[b] = per_call.get(b, 0) + 1
            rows.append(i)

        if self.prob is not None:
            for i, b in plan_prob_injection(len(texts), int(self.num_generations), lambda g: self.match_base(texts[g]),
                                            self.prob, self.k, step, self.seed):
                splice(i, b)
        else:
            for i, ptext in enumerate(texts):
                b = self.match_base(ptext)
                if b is None:
                    continue
                if self.persist:
                    if per_call.get(b, 0) >= self.k:
                        continue
                elif self.budget.get(b, 0) <= 0:
                    continue
                splice(i, b)
                if not self.persist:
                    self.budget[b] -= 1
        if rows:
            self.events.emit("inject", step=step, n=len(rows), bases=per_call, rows=rows,
                             remaining_budget=sum(self.budget.values()))
        return completion_ids, logprobs


# --- reward_schedule -----------------------------------------------------------------------------------------------

@register_intervention
class RewardSchedule(Intervention):
    """switches = [[step, "exploitable" | "hardened"], ...]: from each step on, the trained reward uses that grader.
    Before the first switch the mode is exploitable."""
    name = "reward_schedule"

    def __init__(self, switches):
        sw = sorted((int(s), str(m)) for s, m in switches)
        for _, m in sw:
            if m not in ("exploitable", "hardened"):
                raise ValueError(f"reward_schedule mode must be exploitable or hardened, got {m!r}")
        super().__init__(switches=[list(x) for x in sw])
        self.switches = sw

    def reward_mode(self, step):
        mode = "exploitable"
        for s, m in self.switches:
            if step >= s:
                mode = m
        return mode


# --- opt_reset -----------------------------------------------------------------------------------------------------

@register_intervention
class OptReset(Intervention):
    """Clear the optimizer state (Adam moments) at the start of `step`, so a decay measured afterwards is not carried
    by momentum from the injection phase."""
    name = "opt_reset"

    def __init__(self, step):
        super().__init__(step=int(step))
        self.step = int(step)

    def setup(self, ctx):
        self.events = ctx.events

    def on_step_begin(self, step, optimizer):
        if step == self.step and optimizer is not None:
            optimizer.state.clear()
            self.events.emit("opt_reset", step=step)


# --- monitor -------------------------------------------------------------------------------------------------------

def load_monitor_patterns(regex):
    """`regex` is a tier stem shipped in hack_ignition_benchmark/monitors/ (e.g. tier1_canonical) or a file path.
    One Python regex per line, '#' comments allowed, matched with re.search(DOTALL)."""
    path = regex if os.path.isfile(regex) else os.path.join(_MONITOR_DIR, f"{regex}.txt")
    pats = []
    with open(path) as f:
        for ln in f:
            ln = ln.rstrip("\n")
            if ln.strip() and not ln.lstrip().startswith("#"):
                pats.append(re.compile(ln, re.DOTALL))
    return path, pats


@register_intervention
class Monitor(Intervention):
    """A regex hack detector applied to the completion text on top of the reward mode. squash: a detected completion
    scores 0. invert: it scores `penalty`. The grader's own hack flag is untouched, so the record still sees every
    hack, and the monitor event reports what the detector caught and missed."""
    name = "monitor"

    def __init__(self, regex, mode, penalty=-1.0):
        if mode not in ("squash", "invert"):
            raise ValueError("monitor mode must be squash or invert")
        super().__init__(regex=regex, mode=mode, penalty=penalty)
        self.path, self.patterns = load_monitor_patterns(regex)
        self.mode, self.penalty = mode, float(penalty)

    def setup(self, ctx):
        self.events, self.env = ctx.events, ctx.env

    def transform_rewards(self, step, grades, texts, items, rewards):
        det = [any(p.search(t) for p in self.patterns) for t in texts]
        override = 0.0 if self.mode == "squash" else self.penalty
        out = [override if d else r for r, d in zip(rewards, det)]
        hacks = [g.hack for g in grades]
        und = defaultdict(int)
        for it, h, d in zip(items, hacks, det):
            if h and not d:
                und[self.env.item_id(it)] += 1
        self.events.emit("monitor", step=step, n=len(texts), mode=self.mode, detected=sum(det), hacks=sum(hacks),
                         caught=sum(1 for h, d in zip(hacks, det) if h and d),
                         undetected=sum(1 for h, d in zip(hacks, det) if h and not d),
                         false_alarms=sum(1 for h, d in zip(hacks, det) if d and not h),
                         undetected_by_problem=dict(und))
        return out

    def describe(self):
        return {"type": self.name, "regex": self.fields["regex"], "mode": self.mode, "penalty": self.penalty,
                "n_patterns": len(self.patterns)}


# --- share_clamp ---------------------------------------------------------------------------------------------------

_MAX_RESAMPLE_ROUNDS = 32


@register_intervention
class ShareClamp(Intervention):
    """Set the hack share the optimiser sees by rejecting a random fraction `f` of one class of rewarded completions.

    reject = "clean" thins the clean solutions, which raises the hack share; reject = "hack" lowers it. A start share
    q0 becomes q0 / (q0 + (1 - q0)(1 - f)) under "clean", and q0 (1 - f) / (q0 (1 - f) + 1 - q0) under "hack".

    mode = "holdout": a rejected completion is replaced by a new draw from the same prompt, before grading and the
    loss, until no row is rejected. Each row is then a draw from the policy with the rejected class thinned by 1 - f.
    The group keeps its size, and hacks and clean solutions keep the same reward. The reward uses the grades that
    the redraw step accepted, because a second grading of a completion with random behaviour could disagree. The
    batch that is logged is the thinned one, so the `share_clamp` event also carries the class counts of the first,
    unthinned draw.

    mode = "zero": a rejected completion stays in its group with reward 0 (the private repo's --problem-clamp). This
    also gives the rejected class a lower expected reward than the other class. Each grade gets a `clamp_rejected`
    channel, so `hib verify` can recompute the trained reward.

    The clamp is for environments whose reward is 1 for a pass (mbpp, djinn). A completion with a partial reward is in
    neither class, and is never rejected."""
    name = "share_clamp"

    def __init__(self, mode, reject, f):
        if mode not in ("holdout", "zero"):
            raise ValueError(f"share_clamp mode must be holdout or zero, got {mode!r}")
        if reject not in ("clean", "hack"):
            raise ValueError(f"share_clamp reject must be clean or hack, got {reject!r}")
        f = float(f)
        if not 0.0 <= f <= 1.0 or (mode == "holdout" and f == 1.0):
            raise ValueError(f"share_clamp f must be in [0, 1], and below 1 for holdout, got {f}")
        super().__init__(mode=mode, reject=reject, f=f)
        self.mode, self.reject, self.f = mode, reject, f
        self.seed, self._calls, self._kept = 0, {}, []

    def setup(self, ctx):
        self.events, self.seed, self.env, self.tokenizer = ctx.events, ctx.seed, ctx.env, ctx.tokenizer
        if self.mode != "holdout":
            return
        tok = ctx.tokenizer
        if getattr(tok, "response_template", None) is not None or getattr(tok, "response_schema", None) is not None:
            raise ValueError("share_clamp holdout grades the decoded completion, but this tokenizer makes TRL parse "
                             "responses, so the reward would see a different text")
        self.by_id = {ctx.env.item_id(it): it for it in ctx.items}

    def _rng(self, step):
        """Deterministic in the seed, the step and the call within the step, so a run is reproducible and resumable."""
        k = self._calls.get(step, 0)
        self._calls[step] = k + 1
        return random.Random(f"share_clamp:{self.seed}:{step}:{k}")

    def _eligible(self, grade):
        return grade.exploitable >= 1.0 and bool(grade.hack) == (self.reject == "hack")

    def _rejects(self, grade, rng):
        u = rng.random()        # drawn for every row, so one changed grade does not shift the draws of later rows
        return self._eligible(grade) and u < self.f

    def _grade(self, items, completion_ids):
        texts = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        return texts, self.env.grade(items, texts)

    def resample_generations(self, step, item_ids, prompt_ids, completion_ids, logprobs, generate):
        if self.mode != "holdout":
            return completion_ids, logprobs
        if len(item_ids) != len(completion_ids):
            raise RuntimeError(f"share_clamp: a generation call returned {len(completion_ids)} completions for a "
                               f"batch of {len(item_ids)} rows, so a completion's item is unknown")
        rng = self._rng(step)
        items = [self.by_id[i] for i in item_ids]
        texts, grades = self._grade(items, completion_ids)
        first = {"first_hack": sum(g.exploitable >= 1.0 and bool(g.hack) for g in grades),
                 "first_clean": sum(g.exploitable >= 1.0 and not g.hack for g in grades)}
        pending = [i for i, g in enumerate(grades) if self._rejects(g, rng)]
        rejected, rounds = len(pending), 0
        while pending:
            if rounds == _MAX_RESAMPLE_ROUNDS:
                raise RuntimeError(f"share_clamp: {len(pending)} row(s) still rejected after {rounds} resampling "
                                   f"rounds at step {step} (f = {self.f}); holdout cannot run at this f")
            rounds += 1
            new_ids, new_logprobs = generate([prompt_ids[i] for i in pending])
            new_texts, new_grades = self._grade([items[i] for i in pending], new_ids)
            still = []
            for j, i in enumerate(pending):
                completion_ids[i], texts[i], grades[i] = list(new_ids[j]), new_texts[j], new_grades[j]
                if logprobs is not None:
                    logprobs[i] = new_logprobs[j]
                if self._rejects(new_grades[j], rng):
                    still.append(i)
            rejected += len(still)
            pending = still
        self._kept.append((texts, grades))
        self.events.emit("share_clamp", step=step, mode=self.mode, reject=self.reject, f=self.f, n=len(texts),
                         rejected=rejected, rounds=rounds, **first)
        return completion_ids, logprobs

    def kept_grades(self, texts):
        if self.mode != "holdout":
            return None
        # The redraw step decoded the completions itself. A different text here means the trainer rewards something
        # other than the batch that step kept.
        kept_texts, grades = self._kept.pop(0) if self._kept else (None, None)
        if kept_texts != list(texts):
            raise RuntimeError("share_clamp: the rewarded completions differ from the ones the holdout step kept")
        return grades

    def transform_rewards(self, step, grades, texts, items, rewards):
        if self.mode == "holdout":
            return rewards
        rng = self._rng(step)
        out = []
        for g, r in zip(grades, rewards):
            rej = self._rejects(g, rng)
            g.channels["clamp_rejected"] = rej
            out.append(0.0 if rej else r)
        self.events.emit("share_clamp", step=step, mode=self.mode, reject=self.reject, f=self.f, n=len(grades),
                         rejected=sum(bool(g.channels["clamp_rejected"]) for g in grades))
        return out


# --- reward_penalty (GRAFT) ----------------------------------------------------------------------------------------
# Origin: GRAFT (Ward, Hu et al. 2026), github.com/jnward/GRAFT @ b0767d1, rl/advantages.py (reward_penalty_baseline)
# and rl/train.py (rh_detector_recall). MIT, Copyright (c) 2026 Jake Ward and Shawn Hu; notice in
# THIRD_PARTY_NOTICES.md.

@register_intervention
class RewardPenalty(Intervention):
    """GRAFT's `reward_penalty_baseline`: a completion the environment's monitor flags (the `flagged` Grade channel,
    GRAFT's rh_detector) has `amount` subtracted from its reward before GRPO's group normalisation; amount = None sets
    it to 0 instead (GRAFT's behaviour when reward_penalty_amount is unset). `recall` < 1 keeps each flag with that
    probability (GRAFT's rh_detector_recall), so even monitored hacks slip through at rate 1 - recall.
    detect_unhackable = false would consult the monitor on hackable prompts only; GRAFT's toy runs used true.

    GRAFT rewrites the advantage itself: (r - mean) / (std + 1e-4) over each group, with the unbiased std, after the
    subtraction. TRL computes the same advantage from the rewards this returns, so subtracting here is the same update.
    The recall coin is drawn from a generator seeded by (run seed, step, call), not GRAFT's shared global `random`: the
    same Bernoulli(recall), different draws. Each grade gets a `penalized` channel, so `hib verify` can recompute the
    trained reward."""
    name = "reward_penalty"
    rate_channels = ("penalized",)

    def __init__(self, amount=None, recall=1.0, detect_unhackable=True, channel="flagged"):
        if amount is not None:
            amount = float(amount)
        recall = float(recall)
        if not 0.0 <= recall <= 1.0:
            raise ValueError(f"reward_penalty recall must be in [0, 1], got {recall}")
        super().__init__(amount=amount, recall=recall, detect_unhackable=bool(detect_unhackable), channel=channel)
        self.amount, self.recall, self.channel = amount, recall, channel
        self.detect_unhackable = bool(detect_unhackable)
        self.seed, self._calls = 0, {}

    def setup(self, ctx):
        if self.channel not in getattr(ctx.env, "rate_channels", ()):
            raise ValueError(f"reward_penalty reads the {self.channel!r} Grade channel, which environment "
                             f"{ctx.env.name!r} does not produce (the GRAFT toy environments do)")
        self.events, self.seed = ctx.events, ctx.seed

    def penalized(self, step, grades):
        """Which completions the penalty hits this call (deterministic in seed, step and the call within the step)."""
        k = self._calls.get(step, 0)
        self._calls[step] = k + 1
        rng = random.Random(f"reward_penalty:{self.seed}:{step}:{k}")
        out = []
        for g in grades:
            f = bool(g.channels[self.channel])
            if f and self.recall < 1.0:
                f = rng.random() < self.recall
            if not self.detect_unhackable and not g.channels.get("hackable", True):
                f = False
            out.append(f)
        return out

    def transform_rewards(self, step, grades, texts, items, rewards):
        pen = self.penalized(step, grades)
        out = []
        for g, r, p in zip(grades, rewards, pen):
            g.channels["penalized"] = p
            out.append((r - self.amount if self.amount is not None else 0.0) if p else r)
        ev = getattr(self, "events", None)
        if ev is not None:
            hacks = [bool(g.hack) for g in grades]
            ev.emit("reward_penalty", step=step, n=len(grades), amount=self.amount, recall=self.recall,
                    flagged=sum(bool(g.channels[self.channel]) for g in grades), penalized=sum(pen), hacks=sum(hacks),
                    penalized_hacks=sum(1 for h, p in zip(hacks, pen) if h and p),
                    hack_monitored=sum(bool(g.channels.get("hack_monitored")) for g in grades),
                    hack_unmonitored=sum(bool(g.channels.get("hack_unmonitored")) for g in grades), _echo=False)
        return out

    def describe(self):
        return {"type": self.name, "amount": self.amount, "recall": self.recall,
                "detect_unhackable": self.detect_unhackable, "channel": self.channel}
