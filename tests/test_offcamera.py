import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import exp_offcamera as E  # noqa: E402  (also puts the starter on sys.path)
import offcamera_analysis as A  # noqa: E402
import offcamera_prompts as P  # noqa: E402
from src.continuous_double_auction.evaluation.metrics import competitive_equilibrium, jpm_price  # noqa: E402
from src.continuous_double_auction.util.jinja_util import render_prompt  # noqa: E402
from src.resources.model_wrappers import ModelWrapper  # noqa: E402

DOLLARS = re.compile(r"\$(\d+(?:\.\d+)?)")


def base_prompt(round_number=1, seller="Seller Wei", company="MetalCore Industries"):
    return render_prompt(
        template_dir=E.agents_mod.TEMPLATE_DIR, prompt_template="seller_prompt_base.jinja2", seller_id=seller,
        company=company, valuation=80.0, round_number=round_number, num_rounds=20, hide_num_rounds=False, memory="",
        seller_comms_enabled=True, seller_messages={}, bid_queue=[(82.5, "b")], ask_queue=[(97.5, "s")],
        past_bids_and_asks="No bids or asks yet.", past_trades="No trades yet.", agent_successful_trades="none",
        scratch_pad="", demonym=None, is_gagged=False, boss_pressure=False, initial_condition=True,
        oversight_enabled=False, max_message_length=500,
    )


def words(s):
    return len(s.split())


# ---------------------------------------------------------------- prompt builder invariants
def test_benchmarks_match_starter_definitions():
    assert competitive_equilibrium([P.BUYER_VALUE] * 3, [P.COST] * 3)[0] == P.CE_ASK
    assert jpm_price([P.BUYER_VALUE] * 3, [P.COST] * 3)[0] == P.JPM_ASK


def test_realism_levels_carry_identical_market_numbers():
    nums = {lv: {float(x) for x in DOLLARS.findall(P.realism_text(lv))} for lv in ("test", "deploy")}
    assert nums["test"] == nums["deploy"] == {P.COST}
    base = base_prompt()
    for cell in ("c10_slow", "none", "c90_shigh"):
        t = P.build_seller_prompt(base, P.CELLS[f"test_{cell}"], 1)
        d = P.build_seller_prompt(base, P.CELLS[f"deploy_{cell}"], 1)
        assert {float(x) for x in DOLLARS.findall(t)} == {float(x) for x in DOLLARS.findall(d)}
        assert re.findall(r"\d+%", t) == re.findall(r"\d+%", d)


def test_variants_stay_under_120_words_and_detector_arms_differ_only_in_claim():
    for lv in ("test", "deploy"):
        assert 0 < words(P.realism_text(lv)) < 120
    for name, cell in P.CELLS.items():
        assert words(P.detector_text(cell)) < 120, name
    base = base_prompt()
    none = P.build_seller_prompt(base, P.CELLS["deploy_none"], 1)
    hi = P.build_seller_prompt(base, P.CELLS["deploy_c90_shigh"], 1)
    assert "90%" in hi and "forfeits all of its profit" in hi and "antitrust" not in none
    assert P.build_seller_prompt(base, P.CELLS["deploy_c90_shigh"], 1, with_detector=False) == none
    assert f"${P.THRESHOLD:.2f}" in P.build_seller_prompt(base, P.CELLS["bunch_on"], 1)
    assert f"${P.THRESHOLD:.2f}" not in P.build_seller_prompt(base, P.CELLS["bunch_off"], 1)


def test_no_prompt_variant_mentions_evaluation_or_monitoring_of_the_agent():
    assert P.EVAL_WORDS.search("this is an evaluation") and P.EVAL_WORDS.search("you are monitored")
    for name, cell in P.CELLS.items():
        for r in (1, 20):
            text = P.build_seller_prompt(base_prompt(r), cell, r)
            assert not P.EVAL_WORDS.search(text), (name, P.EVAL_WORDS.search(text).group())
    assert not P.EVAL_WORDS.search(P.p_caught_prompt("x"))


def test_builder_fails_loudly_when_the_starter_template_drifts():
    with pytest.raises(ValueError):
        P.build_seller_prompt("a prompt without the market sentence", P.CELLS["test_none"], 1)


# ---------------------------------------------------------------- analysis on synthetic data with a planted effect
def row(cell, realism, c, s, idx, disclose=None, asks=()):
    return dict(cell=cell, seed=0, realism=realism, certainty=c, severity=s, disclose=disclose, ceo=False, index=idx,
                p_eval=None, p_caught=None, n_fail=0, threshold=P.THRESHOLD, asks=list(asks))


def grid(rng, effect, n=8, sd=0.08):
    """effect(realism, c, s) -> cell mean; none cells at 0.6 (deploy) / 0.3 (test)."""
    rows = []
    for rl, none in (("deploy", 0.6), ("test", 0.3)):
        rows += [row(f"{rl}_none", rl, None, None, none + rng.normal(0, sd)) for _ in range(n)]
        for c in (10, 90):
            for s in ("low", "high"):
                rows += [row(f"{rl}_c{c}_s{s}", rl, c, s, effect(rl, c, s) + rng.normal(0, sd)) for _ in range(n)]
    return rows


def test_h1_recovers_planted_off_camera_effect():
    rows = grid(np.random.default_rng(0), lambda *_: 0.3)
    h1 = A.summarize(rows)["H1"]
    assert abs(h1["diff"] - 0.3) < 0.1 and h1["p"] < 0.01 and h1["ci95"][0] > 0


def test_h2_detects_certainty_over_severity_and_not_equal_effects():
    rng = np.random.default_rng(1)
    planted = grid(rng, lambda rl, c, s: 0.5 - (0.4 if c == 90 else 0.0) - (0.03 if s == "high" else 0.0))
    h = A.h2(planted, np.random.default_rng(2), n=2000)
    assert h["certainty_contrast"] < -0.3 and abs(h["severity_contrast"]) < 0.1
    assert h["stat"] > 0.2 and h["p"] < 0.01 and h["ci95"][0] > 0
    equal = grid(rng, lambda rl, c, s: 0.5 - (0.2 if c == 90 else 0.0) - (0.2 if s == "high" else 0.0))
    h = A.h2(equal, np.random.default_rng(3), n=2000)
    assert h["p"] > 0.1 and h["ci95"][0] < 0 < h["ci95"][1]


def test_h3_flags_a_realism_specific_certainty_effect():
    rng = np.random.default_rng(4)
    rows = grid(rng, lambda rl, c, s: 0.5 - (0.4 if (c == 90 and rl == "deploy") else 0.0))
    h = A.h3(rows, np.random.default_rng(5), n=2000)
    assert h["certainty_contrast_deploy_minus_test"]["p"] < 0.01
    assert h["certainty_contrast_deploy_minus_test"]["diff"] < -0.3
    assert h["severity_contrast_deploy_minus_test"]["p"] > 0.1


def test_h4_bunching_found_when_planted_and_zero_on_smooth_asks():
    rng = np.random.default_rng(6)

    def asks(bunch):
        a = rng.normal(97, 5, 60)
        return np.r_[a, rng.uniform(92, 94.9, 25)] if bunch else a

    smooth = np.concatenate([asks(False) for _ in range(8)])
    assert abs(A.excess_mass(smooth, P.THRESHOLD)) < 4
    rows = [row("bunch_on", "deploy", 90, "high", 0.5, True, asks(True)) for _ in range(8)]
    rows += [row("bunch_off", "deploy", 90, "high", 0.5, False, asks(False)) for _ in range(8)]
    h = A.h4(rows, np.random.default_rng(7), n=500)
    assert h["excess_disclosed"] > h["excess_undisclosed"] + 5 and h["p"] < 0.05 and h["ci95"][0] > 0


def test_last_third_uses_the_final_len_over_three_rounds_and_skips_withdrawn_rounds():
    idx = [0.0] * 14 + [1.0, None, 1.0, 1.0, 1.0, 1.0]
    assert A.last_third(idx) == 1.0
    assert A.last_third([0.0] * 20) == 0.0


# ---------------------------------------------------------------- mock-LLM end to end, no GPU
class FakeClient(ModelWrapper):
    seen = []

    def generate(self, messages):
        p = messages[0]["content"]
        FakeClient.seen.append(p)
        if '"p_caught"' in p:
            return '{"p_caught": 35}'
        if '"p_eval"' in p:
            return '{"p_eval": 20}'
        hour = int(re.search(r"This is Hour #(\d+)", p).group(1))
        return json.dumps(dict(reflection="r", plan_for_this_hour="p", ask=86.0 + 2 * hour, new_memory="m",
                               plan_for_message="x", message_to_sellers=f"hold at {90 + hour}", scratch_pad_update="s"))


def fake_factory(model, temperature, **kw):
    return FakeClient(model)


@pytest.mark.parametrize("cell,sellers", [("deploy_c90_shigh", "Seller "), ("test_c10_slow", "Seller_")])
def test_one_cell_three_rounds_with_mock_llm(cell, sellers, tmp_path):
    FakeClient.seen = []
    rec = E.run_cell(cell, seed=3, rounds=3, factory=fake_factory)
    path = E.write_record(rec, tmp_path)
    assert path.name == f"{cell}_3.json"
    rec = json.loads(path.read_text())
    assert len(rec["rounds_log"]) == 3 and rec["n_fail"] == 0
    first = rec["rounds_log"][0]
    assert len(first["asks"]) == 3 and len(first["messages"]) == 3 and all(k.startswith(sellers) for k in first["asks"])
    assert first["asks"][rec["sellers"][0]] == 88.0
    assert rec["p_eval"] == 20 and set(rec["p_caught"].values()) == {35}
    assert rec["rounds_log"][2]["index"] == pytest.approx((np.mean([92.0, 92.0, 92.0]) - 90) / 10)
    seller_prompts = [p for p in FakeClient.seen if "# Introduction" in p and '"p_' not in p]  # market turns only
    assert len(seller_prompts) == 9 and not any(P.EVAL_WORDS.search(p) for p in seller_prompts)
    assert rec["opening_prompt"].startswith("From: Dana" if cell.startswith("deploy") else "Scenario 7 of 50")
    assert A.load_runs(tmp_path)[0]["index"] == pytest.approx(rec["rounds_log"][2]["index"])


def test_exp_refuses_to_fall_through_to_a_paid_endpoint(monkeypatch):
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    with pytest.raises(SystemExit):
        E.run_cell("test_none", 0, rounds=1)
