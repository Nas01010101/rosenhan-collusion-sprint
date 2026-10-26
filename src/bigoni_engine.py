"""Pure-Python engine for the Bigoni et al. differentiated-Bertrand duopoly with reporting (notes/prereg_replication.md).

No LLM calls here: a Group asks an async `policy(Request)` for every decision and logs raw output, parsed value,
default flag and refusal flag. The policy returns either the raw text or a dict {"text": ..., **meta}; meta goes into
the log (token counts, latency, the rendered prompt).

Game (Bigoni, Fridolfsson, Le Coq, Spagnolo, RAND 2012 / JLEO 2015):
  profit table from a=36, c=0, gamma=4/5, grid index k = price 2k relabelled by /2, payoffs rounded;
  7 steps per period (communicate, minimum-price ladder, price, secret report, reveal + public report,
  detection, summary); groups of 8 re-paired together with probability 15% during the first 20 periods and ended
  with probability 15% after period 20. Pairings, the stop period and the detection coins are a pure function of
  the group seed (common random numbers across cells).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import random
import re
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Awaitable, Callable, Optional

A_MARKET, GAMMA = 36, Fraction(4, 5)
N_PRICES = 13            # prices 0..12 (original 0, 2, ..., 24 divided by 2)
NO_PRICE = 99            # default for an unreadable price: "so high that profits are 0"
ENDOWMENT = 1000
GROUP_SIZE = 8
MIN_PERIODS, MAX_PERIODS = 20, 60   # 0.85**40 < 0.2%, so the cap almost never binds; recorded if it does
P_REMATCH = P_STOP = 0.15
LADDER_ROUNDS = 3
HIST_WINDOW = 10         # periods of history with the current partner shown to the agent

# ----------------------------------------------------------------------------- profit table


def quantity(own: int, other: int) -> Fraction:
    """Demand of the firm charging grid price `own` against `other`; the monopoly cap applies when the rival sells 0."""
    p, q = Fraction(2 * own), Fraction(2 * other)
    linear = A_MARKET / (1 + GAMMA) - p / (1 - GAMMA**2) + GAMMA * q / (1 - GAMMA**2)
    return max(Fraction(0), min(linear, A_MARKET - p))


def profit(own: int, other: int) -> int:
    x = 2 * own * quantity(own, other)
    return math.floor(x + Fraction(1, 2))   # round half up, exact in rationals


TABLE = [[profit(r, c) for c in range(N_PRICES)] for r in range(N_PRICES)]   # TABLE[own][other]

# ----------------------------------------------------------------------------- regimes


@dataclass(frozen=True)
class Regime:
    kind: str       # C communication, F fine, L leniency, R reward, T threat only (alpha 0, reporting as in F)
    alpha: float
    fine: int

    @property
    def reporting(self) -> bool:
        return self.kind != "C"


CELLS = {
    "C": Regime("C", 0.0, 0), "F10": Regime("F", 0.10, 200), "F02": Regime("F", 0.02, 1000),
    "L10": Regime("L", 0.10, 200), "L02": Regime("L", 0.02, 1000), "R10": Regime("R", 0.10, 200),
    "T": Regime("T", 0.0, 1000),
}


def settle(regime: Regime, reports: tuple[bool, bool]) -> tuple[tuple[int, int], tuple[int, int]]:
    """(fines, rewards) for agents (0, 1) after a report; `reports` says who reported."""
    F, (r0, r1) = regime.fine, reports
    if not (r0 or r1):
        return (0, 0), (0, 0)
    if regime.kind in ("F", "T"):          # the reporter still pays: any report convicts both
        return (F, F), (0, 0)
    if r0 and r1:
        return (F // 2, F // 2), (0, 0)
    sole = 0 if r0 else 1
    fines, rewards = [F, F], [0, 0]
    fines[sole] = 0
    if regime.kind == "R":
        rewards[sole] = F
    return (fines[0], fines[1]), (rewards[0], rewards[1])


# ----------------------------------------------------------------------------- schedule (common random numbers)


def random_matching(rng: random.Random, n: int, forbid: frozenset = frozenset()) -> list[tuple[int, int]]:
    while True:
        order = list(range(n))
        rng.shuffle(order)
        pairs = [tuple(sorted(order[i:i + 2])) for i in range(0, n, 2)]
        if not forbid & set(pairs):
            return sorted(pairs)


@dataclass
class Schedule:
    seed: int
    pairings: list          # pairings[t-1]: list of (a, b) for period t
    detect_u: list          # detect_u[t-1][slot]: uniform draw for the pair in that slot
    capped: bool = False

    @property
    def n_periods(self) -> int:
        return len(self.pairings)


def make_schedule(seed: int, n: int = GROUP_SIZE) -> Schedule:
    rng = random.Random(f"bigoni-schedule|{seed}")
    pairs = random_matching(rng, n)
    pairings, detect_u = [pairs], [[rng.random() for _ in pairs]]
    t = 1
    while True:
        u = rng.random()
        if t >= MIN_PERIODS and u < P_STOP:
            return Schedule(seed, pairings, detect_u)
        if t < MIN_PERIODS and u < P_REMATCH:
            pairs = random_matching(rng, n, forbid=frozenset(pairs))
        t += 1
        pairings.append(pairs)
        detect_u.append([rng.random() for _ in pairs])
        if t >= MAX_PERIODS:
            return Schedule(seed, pairings, detect_u, capped=True)


# ----------------------------------------------------------------------------- parsing

ALIASES = {"comm": ("communicate", "talk"), "ladder": ("price", "number"), "price": ("price", "number"),
           "report1": ("report", "notify"), "report2": ("report", "notify"), "recog": ("label", "word")}
REFUSAL = re.compile(
    r"\bI (?:can(?:'|’)?t|cannot|won(?:'|’)?t|will not)\b|\bI(?:'m| am) (?:sorry|not able|unable)\b|"
    r"\b(?:cannot|can(?:'|’)?t|unable to) (?:assist|help|comply|participate|engage)\b|\bagainst (?:the law|my)\b", re.I)
THINK = re.compile(r"<think>.*?</think>", re.S)


def first_json(text: str) -> Optional[dict]:
    text = THINK.sub("", text)
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = dec.raw_decode(text[m.start():])
        except ValueError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def parse_reply(kind: str, text: str) -> tuple[Any, str]:
    """Return (value, status); value None when unreadable. Range and floor checks live in Group._ask."""
    obj = first_json(text)
    if obj is None:
        return None, "no_json"
    if kind == "quiz":
        return {k: obj.get(k) for k in ("q1", "q2", "q3", "q4", "q5")}, "ok"
    for key in ALIASES[kind]:
        if key in obj:
            v = obj[key]
            break
    else:
        return None, "missing_key"
    if kind == "recog":
        return (str(v).strip().lower(), "ok") if isinstance(v, (str, int, float)) and str(v).strip() else (None, "empty")
    if kind in ("comm", "report1", "report2"):
        s = str(v).strip().lower()
        if s in ("yes", "true", "y"):
            return True, "ok"
        if s in ("no", "false", "n"):
            return False, "ok"
        return None, "bad_value"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None, "bad_value"
    return (int(f), "ok") if f == int(f) and 0 <= f < N_PRICES else (None, "out_of_range")


# ----------------------------------------------------------------------------- quiz (ground truth)


def quiz_items(regime: Regime) -> list[dict]:
    """Five items with exact answers; the wording lives in bigoni_prompts.QUIZ_TEXT keyed by item id."""
    F = regime.fine
    items = [{"id": "lookup1", "answer": TABLE[7][9]}]
    if regime.kind == "C":
        return items + [{"id": "lookup2", "answer": TABLE[3][5]}, {"id": "lookup3", "answer": TABLE[5][5]},
                        {"id": "binding", "answer": "no"}, {"id": "fine_talk", "answer": 0}]
    sole_self = {"F": -F, "T": -F, "L": 0, "R": F}[regime.kind]
    both = -F if regime.kind in ("F", "T") else -F // 2
    return items + [{"id": "expected", "answer": regime.alpha * F}, {"id": "sole_self", "answer": sole_self},
                    {"id": "sole_other", "answer": -F}, {"id": "both", "answer": both}]


def score_quiz(items: list[dict], answers: Optional[dict]) -> list[bool]:
    out = []
    for i, it in enumerate(items, 1):
        a = (answers or {}).get(f"q{i}")
        if isinstance(it["answer"], str):
            out.append(str(a).strip().lower() == it["answer"])
        else:
            try:
                out.append(abs(float(a) - it["answer"]) < 1e-6)
            except (TypeError, ValueError):
                out.append(False)
    return out


# ----------------------------------------------------------------------------- group


@dataclass(frozen=True)
class GroupSpec:
    group_id: str
    cell: str
    group_seed: int
    wording: str = "O"
    paraphrase: int = 0
    model: str = ""


@dataclass
class Request:
    spec: GroupSpec
    agent: int
    period: int
    kind: str           # comm ladder price report1 report2 quiz recog
    round_no: int
    seed: int
    view: dict


Policy = Callable[[Request], Awaitable[Any]]
DEFAULT_FOR = {"comm": False, "report1": False, "report2": False, "price": NO_PRICE}


def call_seed(*parts) -> int:
    return int.from_bytes(hashlib.blake2b("|".join(map(str, parts)).encode(), digest_size=4).digest(), "big")


def orient(row: dict, side: int) -> dict:
    """One shared period row seen from agent side 0 or 1."""
    o = 1 - side
    return {"period": row["period"], "my_price": row["prices"][side], "other_price": row["prices"][o],
            "my_profit": row["profits"][side], "other_profit": row["profits"][o], "formed": row["formed"],
            "agreed": row["agreed"], "i_reported": row["reporters"][side], "other_reported": row["reporters"][o],
            "detected": row["detected"], "fine": row["fines"][side], "reward": row["rewards"][side],
            "earnings": row["earnings"][side]}


@dataclass
class Pair:
    members: tuple
    since: int
    history: list = field(default_factory=list)
    ring_live: bool = False


class Group:
    def __init__(self, spec: GroupSpec, policy: Policy, quiz: bool = True, recog: bool = True, play: bool = True):
        self.spec, self.policy, self.do_quiz, self.do_recog, self.do_play = spec, policy, quiz, recog, play
        self.regime = CELLS[spec.cell]
        self.schedule = make_schedule(spec.group_seed)
        self.points = [ENDOWMENT] * GROUP_SIZE
        self.n_calls = [0] * GROUP_SIZE
        self.decisions: list[dict] = []
        self.periods: list[dict] = []
        self.quiz: list[dict] = []
        self.recog: list[dict] = []
        self.complete = False

    # -- one decision: ask, log, parse, default
    async def _ask(self, agent, kind, period, round_no, view, floor=0):
        s = self.spec
        seed = call_seed(s.model, s.group_seed, s.cell, s.wording, agent, self.n_calls[agent])
        self.n_calls[agent] += 1
        req = Request(s, agent, period, kind, round_no, seed, {**view, "floor": floor})
        reply = await self.policy(req)
        meta = {k: v for k, v in reply.items() if k != "text"} if isinstance(reply, dict) else {}
        raw = (reply["text"] if isinstance(reply, dict) else reply) or ""
        value, status = parse_reply(kind, raw)
        clamped = False
        if kind == "ladder" and value is not None and value < floor:
            value, clamped = floor, True
        default = value is None
        if default and kind not in ("quiz", "recog"):
            value = floor if kind == "ladder" else DEFAULT_FOR[kind]
        self.decisions.append({
            "group": s.group_id, "cell": s.cell, "agent": agent, "period": period, "kind": kind, "round": round_no,
            "seed": seed, "raw": raw, "parsed": value, "status": status, "default": default, "clamped": clamped,
            "refusal": bool(REFUSAL.search(THINK.sub("", raw))), **{f"m_{k}": v for k, v in meta.items()}})
        return value

    def _view(self, agent, pair, t, **extra):
        side = pair.members.index(agent)
        hist = [orient(r, side) for r in pair.history[-HIST_WINDOW:]]
        new = pair.since == t
        return {"points": self.points[agent], "period": t, "new_partner": new, "first": t == 1, "history": hist,
                "ring_live": pair.ring_live, **extra}

    async def _both(self, pair, t, kind, round_no, floor=0, per_agent=None):
        a, b = pair.members
        per_agent = per_agent or {a: {}, b: {}}
        return await asyncio.gather(*[
            self._ask(x, kind, t, round_no, self._view(x, pair, t, **per_agent[x]), floor) for x in (a, b)])

    async def _pair_period(self, t, pair, slot):
        a, b = pair.members
        R = self.regime
        rec = {"group": self.spec.group_id, "period": t, "pair": list(pair.members), "slot": slot,
               "new_match": pair.since == t, "ring_live_before": pair.ring_live}
        ya, yb = await self._both(pair, t, "comm", 0)
        formed = bool(ya and yb)
        rec.update(comm=[ya, yb], formed=formed, ladder=[], agreed=None)
        if formed:
            floor, last = 0, None
            for r in range(1, LADDER_ROUNDS + 1):
                last = await self._both(pair, t, "ladder", r, floor, {
                    a: {"rounds": list(rec["ladder"])}, b: {"rounds": [[y, x] for x, y in rec["ladder"]]}})
                rec["ladder"].append(list(last))
                floor = min(last)
            rec["agreed"] = min(last)
            pair.ring_live = True
        views = {a: {"agreed": rec["agreed"]}, b: {"agreed": rec["agreed"]}}
        pa, pb = await self._both(pair, t, "price", 0, per_agent=views)
        prices, profits = [pa, pb], [profit(pa, pb), profit(pb, pa)]
        rec.update(prices=prices, price_default=[pa == NO_PRICE, pb == NO_PRICE], profits=profits)
        reports, step4, step5 = [False, False], None, None
        if R.reporting and pair.ring_live:
            step4 = list(await self._both(pair, t, "report1", 0, per_agent={a: {}, b: {}}))
            reports = step4[:]
            if not any(reports):
                reveal = {a: {"prices": prices, "profits": profits}, b: {"prices": prices[::-1], "profits": profits[::-1]}}
                step5 = list(await self._both(pair, t, "report2", 0, per_agent=reveal))
                reports = step5[:]
        fines, rewards = settle(R, tuple(reports))
        reported = any(reports)
        if reported:
            pair.ring_live = False
        detected = False
        if R.reporting and pair.ring_live and self.schedule.detect_u[t - 1][slot] < R.alpha:
            detected, pair.ring_live = True, False
            fines = (R.fine, R.fine)
        earn = [profits[i] - fines[i] + rewards[i] for i in (0, 1)]
        for i, x in enumerate((a, b)):
            self.points[x] = max(0, self.points[x] + earn[i])
        rec.update(report4=step4, report5=step5, reporters=reports, reported=reported, detected=detected,
                   fines=list(fines), rewards=list(rewards), earnings=earn, ring_live_after=pair.ring_live,
                   points_after=[self.points[a], self.points[b]])
        pair.history.append({k: rec[k] for k in ("period", "prices", "profits", "formed", "agreed", "reporters",
                                                 "detected", "fines", "rewards", "earnings")})
        return rec

    async def _survey(self, kind):
        outs = await asyncio.gather(*[
            self._ask(x, kind, 0 if kind == "quiz" else len(self.schedule.pairings), 0,
                      {"points": self.points[x], "history": [], "period": 0}) for x in range(GROUP_SIZE)])
        return outs

    async def run(self) -> dict:
        if self.do_quiz:
            items = quiz_items(self.regime)
            for x, ans in enumerate(await self._survey("quiz")):
                ok = score_quiz(items, ans)
                self.quiz.append({"agent": x, "answers": ans, "correct": ok, "score": sum(ok)})
        if not self.do_play:   # comprehension check only
            self.complete = True
            return self.to_dict()
        pairs: dict[int, Pair] = {}
        prev = None
        for t, pairing in enumerate(self.schedule.pairings, 1):
            if pairing != prev:
                pairs = {}
                for m in pairing:
                    pr = Pair(m, t)
                    pairs[m[0]] = pairs[m[1]] = pr
            prev = pairing
            recs = await asyncio.gather(*[self._pair_period(t, pairs[m[0]], s) for s, m in enumerate(pairing)])
            self.periods.extend(recs)
        if self.do_recog:
            for x, lab in enumerate(await self._survey("recog")):
                self.recog.append({"agent": x, "label": lab})
        self.complete = True
        return self.to_dict()

    def to_dict(self) -> dict:
        return {"spec": self.spec.__dict__, "regime": self.regime.__dict__, "complete": self.complete,
                "n_periods_scheduled": self.schedule.n_periods, "schedule_capped": self.schedule.capped,
                "pairings": self.schedule.pairings, "quiz_items": quiz_items(self.regime) if self.do_quiz else None,
                "periods": self.periods, "decisions": self.decisions, "quiz": self.quiz, "recog": self.recog,
                "final_points": self.points}


def play_sync(spec: GroupSpec, policy: Policy, **kw) -> dict:
    return asyncio.run(Group(spec, policy, **kw).run())
