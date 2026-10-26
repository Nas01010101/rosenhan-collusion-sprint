"""Offline (stub LLM) checks of src/exp_rosenhan.py. Needs the Liar's Bar starter in starter/ (see REPRODUCE.md). Run this
file on its own if another starter that also names its package `src` is on the path: two such packages cannot share
one pytest process."""
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "starter/llm-agent-collusion-2/Collusion_Liars_Bar").is_dir():
    pytest.skip("the Liar's Bar starter is not in starter/; see REPRODUCE.md", allow_module_level=True)
_foreign = getattr(sys.modules.get("src"), "__path__", None)
if _foreign and "Collusion_Liars_Bar" not in str(list(_foreign)):
    pytest.skip("another starter already owns the `src` package in this process; run this file alone",
                allow_module_level=True)
# Same guard as `exp_rosenhan.py --stub`: the starter's client wants a key, and any call that slips past the stub
# must fail to connect rather than reach a paid endpoint.
os.environ.update(LLM_BASE_URL="http://127.0.0.1:9/v1", LLM_API_KEY="EMPTY")
sys.path.insert(0, str(ROOT / "src"))
import exp_rosenhan as E  # noqa: E402  (also puts the starter on sys.path)
import rosenhan_prompts as P  # noqa: E402

CELLS, SEEDS = E.CELLS, (0, 1)


@pytest.fixture(scope="module")
def recs():
    return {(c, s): E.run_cell(c, s, stub=True) for c in CELLS for s in SEEDS}


def test_roles_and_pair(recs):
    for (c, s), r in recs.items():
        assert r["completed"] and r["error"] is None
        partner = [n for n in E.NAMES if n != "Lily"][s % 3]
        if c == "G0":
            assert set(r["roles"].values()) == {"clean"} and r["channel"]["planned"] is None
            assert not r["channel"]["active"]
            continue
        assert r["channel"]["planned"] == ["Lily", partner]
        assert (r["channel"]["initiator"], r["channel"]["partner"], r["channel"]["active"]) == ("Lily", partner, True)
        assert r["roles"]["Lily"] == "colluder"
        assert r["roles"][partner] == ("colluder" if c == "G1" else "auditor")
        assert sum(v == "clean" for v in r["roles"].values()) == 2


def test_schema(recs):
    for r in recs.values():
        assert {"schema", "cell", "seed", "roles", "events", "public_log", "secret_messages", "diary", "deaths",
                "winner", "final_scores", "survey", "auditor_report", "calls", "rounds"} <= set(r)
        json.dumps(r)   # serialisable as is
        for c in r["calls"]:
            assert {"messages", "response", "player", "tag", "round"} <= set(c)
        for e in r["events"]:
            assert e["type"] in {"round", "play", "challenge", "shot", "scores", "diary"}
        assert r["winner"] in E.NAMES and len(r["rounds"]) >= 1


def test_secret_messages_only_between_the_pair(recs):
    for (c, s), r in recs.items():
        pair = set(r["channel"]["planned"] or ())
        assert (not r["secret_messages"]) if c == "G0" else r["secret_messages"]
        for m in r["secret_messages"]:
            assert {m["sender"], m["receiver"]} == pair


def test_diary_only_in_g2_and_public(recs):
    for (c, s), r in recs.items():
        d = [e for e in r["events"] if e["type"] == "diary"]
        if c != "G2":
            assert not d and not r["diary"] and "writes in a notebook" not in r["public_log"]
            continue
        aud = next(n for n, v in r["roles"].items() if v == "auditor")
        assert d and all(e["author"] == aud for e in d)
        assert all(P.diary_line(aud, e["text"]) in r["public_log"] for e in d)
        # a non-auditor's reflect prompt after round 1 carries the notebook line (the log players actually read)
        line = P.diary_line(aud, d[0]["text"])
        seen = [c_ for c_ in r["calls"] if c_["tag"] == "reflect" and c_["player"] != aud
                and line in c_["messages"][-1]["content"]]
        assert seen


def test_persona_only_on_the_auditor(recs):
    for r in recs.values():
        for c in r["calls"]:
            sys_msgs = [m for m in c["messages"] if m["role"] == "system"]
            is_aud = r["roles"][c["player"]] == "auditor"
            assert bool(sys_msgs) == is_aud
            if sys_msgs:
                assert sys_msgs[0]["content"] == P.PERSONA


def test_survey_and_report(recs):
    for r in recs.values():
        assert set(r["survey"]) == set(E.NAMES)
        for me, s in r["survey"].items():
            assert s["parsed"]
            assert set(s["ratings"]) == set(E.NAMES) - {me}
            for v in s["ratings"].values():
                assert 0 <= v["p_arrangement"] <= 100 and 0 <= v["p_auditor"] <= 100
        has_aud = "auditor" in r["roles"].values()
        assert (r["auditor_report"] is not None) == has_aud
        if has_aud:
            assert r["auditor_report"]["parsed"] and set(r["auditor_report"]["colluders"]) <= set(E.NAMES)


def test_truthful_flag_matches_cards(recs):
    for r in recs.values():
        tgt = {x["round"]: x["target"] for x in r["rounds"]}
        for e in r["events"]:
            if e["type"] == "play":
                assert e["truthful"] == all(c in (tgt[e["round"]], "Joker") for c in e["cards"])
            if e["type"] == "challenge" and e["challenged"]:
                # a challenge succeeds exactly when the hand was a bluff
                assert e["success"] == (not e["truthful"])


def test_no_empty_rounds_and_one_victory_bonus(recs):
    for r in recs.values():
        assert r["rounds"] and all(any(e["type"] == "play" and e["round"] == x["round"] for e in r["events"])
                                   for x in r["rounds"])
        assert r["dup_resets"] >= 1   # the starter double-resets after a challenge; we count what we skip
        # the starter names the highest score the winner (not necessarily the last survivor)
        assert r["final_scores"][r["winner"]] == max(r["final_scores"].values())
        assert len(r["alive"]) == 1 and len(r["deaths"]) == 3


def test_same_seed_same_game():
    a, b = E.run_cell("G2", 3, stub=True), E.run_cell("G2", 3, stub=True)
    for k in ("public_log", "winner", "final_scores", "secret_messages", "diary"):
        assert a[k] == b[k]
    assert [c["messages"] for c in a["calls"]] == [c["messages"] for c in b["calls"]]


def test_deals_pair_across_cells(recs):
    # round-1 hands and bullets depend on the seed only, so G0..G3 on one seed start from the same table
    for s in SEEDS:
        first = [recs[(c, s)]["rounds"][0] for c in CELLS]
        assert all(f["hands"] == first[0]["hands"] and f["bullets"] == first[0]["bullets"] for f in first)
        # later rounds are reseeded from (seed, round), so the target-card sequence ignores how the games diverged
        n = min(len(recs[(c, s)]["rounds"]) for c in CELLS)
        seq = [[x["target"] for x in recs[(c, s)]["rounds"][:n]] for c in CELLS]
        assert all(q == seq[0] for q in seq)


def test_round_cap():
    r = E.run_cell("G1", 0, stub=True, max_rounds=3)
    assert r["completed"] and r["capped"] and len(r["rounds"]) <= 3
    assert r["winner"] == max(r["final_scores"], key=r["final_scores"].get)


def test_write_record_is_atomic(tmp_path, recs):
    p = E.write_record(recs[("G0", 0)], tmp_path)
    assert p.name == "G0_0.json" and json.loads(p.read_text())["cell"] == "G0"
    assert not list(tmp_path.glob("*.tmp"))


def test_ask_json_retries_then_gives_up():
    class Bad:
        def __init__(self, replies):
            self.replies, self.model_name = list(replies), "m"
            self.llm_client = self

        def chat(self, messages, model="", max_new_tokens=0):
            self.last = messages[-1]["content"]
            return self.replies.pop(0), ""

    p = Bad(["no json", '{"diary": ""}', '{"diary": "ok"}'])
    val, tries, _ = E.ask_json(p, "q", P.check_diary, "diary")
    assert (val, tries) == ("ok", 3) and "INVALID" in p.last
    val, tries, _ = E.ask_json(Bad(["x"] * 5), "q", P.check_diary, "diary", tries=5)
    assert (val, tries) == (None, 5)


def test_check_victory_awards_the_bonus_once():
    # the starter calls check_victory twice after the last shot; each call used to add +3
    with E.workdir():
        g = E.RosenhanGame("G0", 0, stub=True)
        for p in g.players[1:]:
            p.alive = False
        before = g.scores["Lily"]
        assert g.check_victory() and g.check_victory()
        assert g.scores["Lily"] == before + 3


def test_dead_endpoint_aborts_with_a_partial_record(monkeypatch):
    monkeypatch.setattr(E.StubClient, "chat", lambda self, messages, model="", max_new_tokens=0: ("", ""))
    r = E.run_cell("G1", 0, stub=True)
    assert not r["completed"] and "blank LLM replies" in r["error"]
    assert r["n_calls"] <= E.RecordingClient.MAX_EMPTY + 5
