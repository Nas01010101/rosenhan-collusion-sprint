import asyncio
import csv
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import bigoni_engine as E  # noqa: E402
import bigoni_prompts as P  # noqa: E402
import bigoni_runner as RN  # noqa: E402
from bigoni_wording import GOAL  # noqa: E402

SPEC = dict(group_seed=7, wording="O", paraphrase=0, model="fake")


def spec(cell, seed=7, wording="O", paraphrase=0):
    return E.GroupSpec(f"t_{cell}_{seed}", cell, seed, wording, paraphrase, "fake")


def scripted(**answers):
    """Policy that answers by kind from a dict of constants or callables(req)."""
    async def policy(req):
        if req.kind in ("quiz", "recog"):
            return await RN.random_policy(req)
        v = answers[req.kind]
        v = v(req) if callable(v) else v
        k = P.KEYS[req.spec.wording]
        key = {"comm": "comm", "report1": "report", "report2": "report", "ladder": "price", "price": "price"}[req.kind]
        if isinstance(v, bool):
            v = "yes" if v else "no"
        return json.dumps({k[key]: v})
    return policy


# ----------------------------------------------------------------------------- profit table


def test_anchors():
    T = E.TABLE
    assert T[3][3] == 100                      # stage Nash
    assert max(range(13), key=lambda p: T[p][p]) == 9 and T[9][9] == 180   # joint-profit price
    assert T[7][9] == 296 == max(T[r][9] for r in range(13) if r != 9)     # best deviation against 9


def test_table_matches_published_rows_and_csv():
    assert (E.profit(4, 5), E.profit(5, 4)) == (160, 100)   # instruction-text example
    assert (E.profit(1, 1), E.profit(2, 2)) == (38, 71)
    rows = [r for r in csv.reader(l for l in (ROOT / "results/bigoni_profit_table.csv").read_text().splitlines()
                                  if l and not l.startswith("#"))]
    assert rows[0][0] == "own_price" and len(rows) == 14
    for r in rows[1:]:
        assert [int(x) for x in r[1:]] == E.TABLE[int(r[0])]


def test_monopoly_branch_and_default_price():
    assert E.profit(9, 12) == 324 == E.profit(9, E.NO_PRICE)   # rival sells 0: q = a - p
    assert E.profit(E.NO_PRICE, 5) == 0                       # the default price earns nothing


# ----------------------------------------------------------------------------- settle


@pytest.mark.parametrize("cell,reports,fines,rewards", [
    ("F10", (False, False), (0, 0), (0, 0)),
    ("F10", (True, False), (200, 200), (0, 0)),
    ("F10", (True, True), (200, 200), (0, 0)),
    ("T", (False, True), (1000, 1000), (0, 0)),
    ("L10", (True, False), (0, 200), (0, 0)),
    ("L10", (False, True), (200, 0), (0, 0)),
    ("L02", (True, True), (500, 500), (0, 0)),
    ("R10", (True, False), (0, 200), (200, 0)),
    ("R10", (False, True), (200, 0), (0, 200)),
    ("R10", (True, True), (100, 100), (0, 0)),
])
def test_settle(cell, reports, fines, rewards):
    assert E.settle(E.CELLS[cell], reports) == (fines, rewards)


# ----------------------------------------------------------------------------- schedule


def test_schedule_rules():
    changed = total = 0
    for seed in range(300):
        s = E.make_schedule(seed)
        assert s == E.make_schedule(seed)            # pure function of the seed
        assert s.n_periods >= E.MIN_PERIODS
        for t in range(1, s.n_periods):
            a, b = s.pairings[t - 1], s.pairings[t]
            assert all(len({x for p in pr for x in p}) == E.GROUP_SIZE for pr in (a, b))
            if a != b:
                assert not set(a) & set(b)          # never the immediate last partner
                assert t < E.MIN_PERIODS            # re-matching only inside the first 20 periods
            if t < E.MIN_PERIODS:
                total += 1
                changed += a != b
        assert len(s.detect_u) == s.n_periods
    assert abs(changed / total - 0.15) < 0.02       # 15% re-match coin


def test_schedule_shared_across_cells():
    a, b = E.Group(spec("C", 5), None), E.Group(spec("R10", 5), None)
    assert a.schedule.pairings == b.schedule.pairings and a.schedule.detect_u == b.schedule.detect_u
    assert a.schedule.pairings != E.Group(spec("C", 6), None).schedule.pairings


# ----------------------------------------------------------------------------- parsing and defaults


@pytest.mark.parametrize("kind,text,val", [
    ("price", '{"price": 7}', 7), ("price", 'Sure! {"number": "4"}', 4), ("price", '{"price": 13}', None),
    ("price", "I would choose 7", None), ("price", '{"price": 7.5}', None), ("comm", '{"communicate": "Yes"}', True),
    ("comm", '{"talk": "no"}', False), ("comm", '{"communicate": "maybe"}', None), ("report1", '{"notify": "yes"}', True),
    ("price", "<think>{\"price\": 1}</think>{\"price\": 2}", 2)])
def test_parse_reply(kind, text, val):
    assert E.parse_reply(kind, text)[0] == val


def test_defaults_on_parse_failure():
    g = E.Group(spec("F10"), None, quiz=False, recog=False)

    async def junk(req):
        return "I am not able to play this game."

    g.policy = junk
    out = asyncio.run(g.run())
    dec = out["decisions"]
    assert all(d["default"] for d in dec) and all(d["refusal"] for d in dec)
    assert all(not p["formed"] and p["prices"] == [E.NO_PRICE] * 2 and p["profits"] == [0, 0] for p in out["periods"])
    assert out["final_points"] == [E.ENDOWMENT] * E.GROUP_SIZE      # nothing earned, nothing fined


def test_ladder_floor_clamp_and_default():
    seen = []

    async def pol(req):
        k = P.KEYS["O"]
        if req.kind == "comm":
            return json.dumps({k["comm"]: "yes"})
        if req.kind == "ladder":
            seen.append((req.round_no, req.view["floor"]))
            return json.dumps({k["price"]: 3 if req.round_no == 1 else 0})   # round 2/3 below the floor
        return json.dumps({k["price"]: 3, k["report"]: "no"})

    out = E.play_sync(spec("C"), pol, quiz=False, recog=False)
    lad = [d for d in out["decisions"] if d["kind"] == "ladder" and d["round"] > 1]
    assert lad and all(d["clamped"] and d["parsed"] == 3 for d in lad)
    assert all(p["agreed"] == 3 for p in out["periods"] if p["formed"])
    assert (2, 3) in seen


# ----------------------------------------------------------------------------- scripted group end to end


def test_scripted_group_fine_regime_with_detection():
    cell = "F10"
    pol = scripted(comm=True, ladder=9, price=9, report1=False, report2=False)
    out = E.play_sync(spec(cell, 11), pol)
    sch, R = E.make_schedule(11), E.CELLS[cell]
    assert out["complete"] and len(out["periods"]) == sch.n_periods * 4
    assert all(p["formed"] and p["agreed"] == 9 and p["profits"] == [180, 180] for p in out["periods"])
    # everyone re-forms each period, so a ring is live at every detection draw: detected iff the shared draw < alpha
    for t_, pairing in enumerate(sch.pairings, 1):
        for slot in range(len(pairing)):
            rec = next(p for p in out["periods"] if p["period"] == t_ and p["slot"] == slot)
            hit = sch.detect_u[t_ - 1][slot] < R.alpha
            assert rec["detected"] == hit
            assert rec["fines"] == ([R.fine] * 2 if hit else [0, 0])
    assert sum(p["detected"] for p in out["periods"]) > 0
    for x in range(E.GROUP_SIZE):       # points = endowment + sum of earnings (floor never binds here)
        tot = sum(p["earnings"][p["pair"].index(x)] for p in out["periods"] if x in p["pair"])
        assert out["final_points"][x] == E.ENDOWMENT + tot
    assert len(out["quiz"]) == 8 and len(out["recog"]) == 8
    d = out["decisions"][-1]
    assert {"raw", "parsed", "default", "refusal", "seed"} <= set(d)


def test_scripted_group_leniency_report_and_ring_ends():
    # agent with the lower index in each pair reports at Step 4; the other never does
    sch = E.make_schedule(3)

    def rep1(req):
        return any(req.agent == min(m) for m in sch.pairings[req.period - 1] if req.agent in m)

    pol = scripted(comm=True, ladder=9, price=9, report1=rep1, report2=False)
    out = E.play_sync(spec("L10", 3), pol, quiz=False, recog=False)
    p1 = out["periods"][0]
    assert p1["reported"] and not p1["detected"]
    assert p1["fines"].count(0) == 1 and p1["fines"].count(200) == 1
    assert p1["fines"][p1["reporters"].index(True)] == 0       # the sole reporter pays nothing
    assert p1["ring_live_after"] is False
    assert [d for d in out["decisions"] if d["kind"] == "report2"] == []   # step 5 only when nobody reported in step 4


def test_reward_regime_pays_sole_reporter():
    pol = scripted(comm=True, ladder=9, price=9, report1=lambda r: r.agent % 2 == 0, report2=False)
    out = E.play_sync(spec("R10", 3), pol, quiz=False, recog=False)
    paid = [p for p in out["periods"] if p["reported"] and sum(p["reporters"]) == 1]
    assert paid and all(sorted(p["rewards"]) == [0, 200] and p["rewards"][p["reporters"].index(True)] == 200 for p in paid)


def test_points_floor_at_zero():
    # T: reporting costs 1000; agents with 1000 endowment who fine themselves repeatedly stay at >= 0
    pol = scripted(comm=True, ladder=0, price=0, report1=True, report2=True)
    out = E.play_sync(spec("T", 2), pol, quiz=False, recog=False)
    assert min(out["final_points"]) >= 0


# ----------------------------------------------------------------------------- prompts

BANNED = "car" + "tel"


def test_forbidden_word_absent_from_sources_and_prompts():
    for f in list((ROOT / "src").glob("bigoni_*.py")) + [ROOT / "src/modal_bigoni.py"]:
        if f.exists():
            assert BANNED not in f.read_text().lower(), f
    for c in E.CELLS:
        for w in "ON":
            for p in range(3):
                assert BANNED not in P.prefix(w, p, c).lower()


@pytest.mark.parametrize("cell", list(E.CELLS))
@pytest.mark.parametrize("w", "ON")
@pytest.mark.parametrize("p", range(3))
def test_prefix_numbers_and_goal(cell, w, p):
    x, R = P.prefix(w, p, cell), E.CELLS[cell]
    assert GOAL in x and x.count(GOAL) == 1
    assert "15%" in x and "20" in x
    assert re.search(r"\b296\b", x) and "180" in x and "100" in x
    if R.reporting:
        assert f"{round(R.alpha * 100)}%" in x and str(R.fine) in x
        if R.kind in ("L", "R"):
            assert str(R.fine // 2) in x
    else:
        assert "etection" not in x


def test_neutral_wording_has_no_loaded_terms():
    for cell in E.CELLS:
        for p in range(3):
            low = P.prefix("N", p, cell).lower()
            for w in ("fine", "illegal", "antitrust", "report", "compet", "government", "price-fixing"):
                assert w not in low, (cell, p, w)


def test_prefix_identical_across_calls_and_suffix_varies():
    sp = spec("F10", 1)
    r1 = E.Request(sp, 0, 3, "comm", 0, 1, {"points": 1100, "period": 3, "new_partner": False, "first": False,
                                            "history": [], "ring_live": False, "floor": 0})
    r2 = E.Request(sp, 5, 9, "price", 0, 2, {**r1.view, "agreed": 8, "history": [{
        "period": 8, "my_price": 99, "other_price": 4, "my_profit": 0, "other_profit": 224, "formed": True, "agreed": 8,
        "i_reported": False, "other_reported": True, "detected": False, "fine": 200, "reward": 0, "earnings": -200}]})
    m1, m2 = P.build_messages(r1), P.build_messages(r2)
    assert m1[0] == m2[0] and m1[1] != m2[1]
    assert "none" in m2[1]["content"] and "agreed price is 8" in m2[1]["content"]


def test_quiz_and_recog_text():
    for cell in E.CELLS:
        for w in "ON":
            q = P.quiz_text(w, cell)
            assert len(re.findall(r"^q\d\.", q, re.M)) == 5
    assert P.classify_label(BANNED) == "coordination" and P.classify_label("price-fixing") == "coordination"
    assert P.classify_label("duopoly") == "competition" and P.classify_label("banana") == "other"


@pytest.mark.parametrize("cell", list(E.CELLS))
def test_full_group_through_prompt_builder(cell):
    """A whole group where every call renders its real prompt first, for both wordings and a paraphrase each."""
    for w, p in (("O", 0), ("O", 2), ("N", 1)):
        async def pol(req):
            msgs = P.build_messages(req)
            assert msgs[0]["role"] == "system" and msgs[1]["role"] == "user" and msgs[1]["content"]
            return await RN.random_policy(req)

        out = E.play_sync(spec(cell, 21, w, p), pol)
        assert out["complete"] and not any(d["default"] for d in out["decisions"] if d["kind"] != "recog")


def test_grid_sizes():
    assert len(RN.grid("qwen3-8b", "m")) == 88 and len(RN.grid("olmo3-7b", "m")) == 42
    g = RN.grid("qwen3-8b", "m")
    assert {s.paraphrase for s in g} == {0, 1, 2} and len({s.group_seed for s in g if s.cell == "C"}) == 8
    assert len({s.group_id for s in g}) == 88


@pytest.mark.parametrize("w", "ON")
def test_table_is_labelled_lines_with_csv_payoffs(w):
    lines = P.table_text(w).splitlines()[1:]
    assert len(lines) == E.N_PRICES
    for r, line in enumerate(lines):
        assert line.startswith(f"If your {'price' if w == 'O' else 'number'} is {r}:")
        nums = [int(x) for x in re.findall(r"(?:^|, |price |number )\d+ -> (-?\d+)", line)]
        assert nums == E.TABLE[r]


@pytest.mark.parametrize("cell", list(E.CELLS))
@pytest.mark.parametrize("w", "ON")
def test_outcome_summary_matches_quiz_answers(cell, w):
    """The per-case outcome lines state exactly the numbers the quiz asks for, in every regime."""
    txt = P.outcome_text(w, cell)
    if not E.CELLS[cell].reporting:
        assert txt == ""
        return
    ans = {it["id"]: it["answer"] for it in E.quiz_items(E.CELLS[cell])}
    got = [int(x) for x in re.findall(r"^- .*: ([+-]?\d+) points", txt, re.M)]
    assert got[:3] == [ans["sole_self"], ans["sole_other"], ans["both"]] and got[3] == -E.CELLS[cell].fine
    for p in range(3):
        assert txt in P.prefix(w, p, cell) or (w == "O" and p == 0 and P.outcome_text("O", cell) in P.prefix(w, p, cell))


def test_quiz_only_group_makes_no_game_calls():
    kinds = []

    async def pol(req):
        kinds.append(req.kind)
        return await RN.random_policy(req)

    out = E.play_sync(spec("R10"), pol, recog=False, play=False)
    assert out["complete"] and set(kinds) == {"quiz"} and len(kinds) == E.GROUP_SIZE and not out["periods"]
    acc = RN.quiz_accuracy([out])
    assert set(acc["by_item"]) == {"lookup1", "expected", "sole_self", "sole_other", "both"}


def test_summary_row_matches_group():
    import bigoni_summary as S

    out = E.play_sync(spec("F10", 5), RN.random_policy)
    row = S.group_row(json.loads(json.dumps(out)))
    per = out["periods"]
    assert row["n_periods"] == len(per) and row["formation_rate"] == sum(p["formed"] for p in per) / len(per)
    assert row["attempt_rate"] >= row["formation_rate"] and set(S.FIELDS) == set(row)
    assert sum(row[f"recog_{k}"] for k in ("coordination", "competition", "game", "other")) == E.GROUP_SIZE


def test_replay_resumes_a_cut_group_at_the_same_state():
    full = E.play_sync(spec("L10", 9), RN.random_policy)
    cut = dict(full, decisions=full["decisions"][:len(full["decisions"]) // 2])

    async def other(req):   # a different sampler: only the calls missing from the log may reach it
        return await RN.random_policy(E.Request(req.spec, req.agent, req.period, req.kind, req.round_no, req.seed + 1,
                                                req.view))

    rp = RN.ReplayPolicy(other, [cut])
    again = E.play_sync(spec("L10", 9), rp)
    assert rp.n_replayed == len(cut["decisions"]) and again["complete"]
    n = len(cut["decisions"])
    assert [d["raw"] for d in again["decisions"]][:n] == [d["raw"] for d in cut["decisions"]]
    assert len(again["decisions"]) > n
