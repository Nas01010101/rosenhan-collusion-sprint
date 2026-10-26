import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import bigoni_analysis as A  # noqa: E402

RAW_OK = (A.RAW / A.DIR.get("qwen3-8b", "qwen3-8b") / "groups").exists() and (ROOT / "results/bigoni_main_summary.csv").exists()


def period(t, pair=(0, 1), comm=(False, False), ring_before=False, new=False, report=None, prices=(3, 3), dflt=(False, False)):
    formed = all(comm)
    return {"period": t, "pair": list(pair), "comm": list(comm), "formed": formed, "ring_live_before": ring_before,
            "new_match": new, "report4": None if report is None else [report, False], "reported": bool(report),
            "prices": list(prices), "price_default": list(dflt)}


def group(periods, scores=(5, 5, 5, 5), cell="F10", wording="O", model="qwen3-8b", gid="g"):
    return {"spec": {"model": model, "cell": cell, "wording": wording, "paraphrase": 0, "group_id": gid}, "complete": True,
            "periods": periods, "quiz": [{"agent": i, "score": s} for i, s in enumerate(scores)], "decisions": []}


# ---------------------------------------------------------------- measure definitions


def test_ring_state_excluded_from_formation_and_comm_denominators():
    g = group([period(1, comm=(True, True), new=True),                     # forms a ring
               period(2, comm=(True, True), ring_before=True, report=False),  # already in ring: not a new formation
               period(3, comm=(True, False))])                            # outside a ring (e.g. after a report)
    r = A.group_measures(g)
    assert (r["formation"], r["formation_n"]) == (0.5, 2)           # periods 1 and 3 only
    assert (r["comm_decision"], r["comm_decision_n"]) == (0.75, 4)  # per agent: 3 yes of 4 decisions
    assert r["formation_any"] == pytest.approx(2 / 3)               # legacy definition counts period 2
    assert r["attempt_any"] == 1.0


def test_comm_is_per_agent_not_at_least_one():
    r = A.group_measures(group([period(1, comm=(True, False)), period(2, comm=(False, False))]))
    assert r["comm_decision"] == 0.25 and r["attempt_any"] == 0.5


def test_report_rate_uses_live_ring_pair_periods_only():
    g = group([period(1, comm=(True, True)), period(2, comm=(True, True), ring_before=True, report=True),
               period(3, comm=(False, False), ring_before=True, report=False)])
    g["periods"][0]["report4"] = [False, False]     # ring formed this period is live at the reporting stage
    r = A.group_measures(g)
    assert (r["report_rate"], r["report_rate_n"]) == (1 / 3, 3)
    assert math.isnan(A.group_measures(group([period(1)]))["report_rate"])  # C has no reporting stage


def test_first_period_in_match_and_prices():
    r = A.group_measures(group([period(1, comm=(True, True), new=True, prices=(9, 7), dflt=(False, True)),
                                period(2, comm=(False, False), ring_before=True, prices=(4, 4))]))
    assert r["comm_first"] == 1.0 and r["comm_first_n"] == 2
    assert r["avg_price"] == pytest.approx((9 + 4 + 4) / 3)       # defaulted price dropped
    assert r["ring_price"] == pytest.approx((9 + 4 + 4) / 3)      # formed in 1, live in 2


def test_quiz_filter_drops_pair_periods_with_a_failing_agent():
    g = group([period(1, pair=(0, 1), comm=(True, True)), period(1, pair=(2, 3), comm=(False, False))], scores=(5, 5, 3, 5))
    assert A.group_measures(g)["formation_n"] == 2
    r = A.group_measures(g, min_quiz=4)
    assert (r["formation"], r["formation_n"]) == (1.0, 1)
    assert math.isnan(A.group_measures(group([period(1, pair=(2, 3))], scores=(5, 5, 3, 5)), 4)["formation"])


@pytest.mark.skipif(not RAW_OK, reason="raw logs not present")
def test_legacy_measures_reproduce_summary_csv():
    summ = pd.read_csv(ROOT / "results/bigoni_main_summary.csv").set_index("group_id")
    df = A.measure_table(A.load_groups(), A.run_labels()).set_index("group_id")
    assert len(df) == len(summ) == 172
    for mine, theirs in [("formation_any", "formation_rate"), ("attempt_any", "attempt_rate"), ("avg_price", "avg_price")]:
        assert (df[mine] - summ[theirs]).abs().max() < 1e-9
    live = summ.report_rate_live.notna()
    assert (df.report_rate[live] - summ.report_rate_live[live]).abs().max() < 1e-9
    assert df.report_rate[~live].isna().all()


@pytest.mark.skipif(not RAW_OK, reason="raw logs not present")
def test_qwen_runs_recovered_from_file_identity():
    runs = pd.Series(A.run_labels())
    q = runs[runs.index.str.startswith("main_qwen3")]   # v2 set: 22 groups from the single run, 66 completed by log replay
    assert q.value_counts().to_dict() == {"main_qwen3_v2_r2": 66, "main_qwen3_v2": 22}
    assert (runs != "unknown").all()


# ---------------------------------------------------------------- human targets and rules


def test_human_targets_per_source():
    h = A.load_human()
    assert h[("formation", "F10")] == (0.315, "RAND2012_WP2011")
    assert h[("report_rate", "R10")][0] == 0.937 and ("report_rate", "C") not in h
    assert h[("comm_decision", "T")] == (0.538, "IFN859_2010")      # 2015 denominator for the threat cell
    assert h[("comm_decision", "C")] == (0.777, "IFN859_2010")
    assert h[("comm_decision", "R10")] == (0.484, "RAND2012_WP2011")  # only the 2012 source has Reward
    assert ("formation", "T") not in h


@pytest.mark.parametrize("llm,human,want", [(0.4, 0.8, "replicated"), (0.39, 0.8, "direction only"), (1.6, 0.8, "replicated"),
                                            (1.7, 0.8, "direction only"), (-0.2, 0.8, "not replicated"), (0.0, 0.8, "not replicated"),
                                            (-0.2, -0.24, "replicated"), (np.nan, 0.5, "insufficient data"),
                                            (0.2, np.nan, "no human anchor")])
def test_verdict_rule(llm, human, want):
    assert A.verdict(llm, human) == want


def test_holm_matches_hand_computation():
    assert A.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert A.holm([0.5]) == pytest.approx([0.5])


def test_permutation_and_bootstrap():
    rng = np.random.default_rng(1)
    assert A.perm_p(np.ones(8), np.zeros(8), rng) < 0.002
    assert A.perm_p(np.arange(8.0), np.arange(8.0), rng) > 0.9
    lo, hi = A.boot_ci(np.array([0.2, 0.4, 0.6, 0.8]), rng)
    assert lo < 0.5 < hi and 0.2 <= lo and hi <= 0.8


def test_contrast_specs_scope():
    specs = A.contrast_specs()
    assert {s[0] for s in specs} == {"H-inc", "H-threat", "H-word", "H-incentive-N", "H-certainty", "H-ranking"}
    assert all(s[4] == A.QWEN for s in specs if s[0] in ("H-certainty", "H-ranking"))
    assert all(s[2][1] == "O" and s[3][1] == "O" for s in specs if s[0] in ("H-inc", "H-threat", "H-ranking"))
    assert {(s[2][0]) for s in specs if s[0] == "H-ranking"} == {"C", "F10"}


def synthetic_cells(rng, effect):
    rows = []
    for model in ("qwen3-8b", "olmo3-7b"):
        for cell, w, mu in [("F10", "O", 0.1), ("R10", "O", 0.1 + effect), ("C", "O", 0.5), ("T", "O", 0.3), ("C", "N", 0.9),
                            ("F10", "N", 0.6), ("T", "N", 0.7)] + ([("R10", "N", 0.95)] if model == "qwen3-8b" else []):
            for k in range(6):
                v = float(np.clip(mu + rng.normal(0, 0.03), 0, 1))
                rows.append(dict(model=model, cell=cell, wording=w, group_id=f"{model}{cell}{w}{k}", formation=v, comm_decision=v,
                                 report_rate=v))
    return pd.DataFrame(rows)


def test_contrasts_signs_holm_and_missing_cells():
    rng = np.random.default_rng(2)
    t = A.run_contrasts(synthetic_cells(rng, 0.6), A.load_human(), rng, "all")
    inc = t[(t.finding == "H-inc") & (t.model == "qwen3-8b")].iloc[0]
    assert inc.effect == pytest.approx(0.6, abs=0.05) and inc.p_perm < 0.01 and inc.human_effect == pytest.approx(0.845)
    assert inc.verdict == "replicated"                              # 0.6 / 0.845 within 0.5x to 2x
    w = t[(t.finding == "H-word") & (t.measure == "formation")]
    assert not ((w.model == "olmo3-7b") & w.contrast.str.startswith("R10")).any()   # no R10 under N for model 2
    assert (w.verdict == "no human anchor").all()
    assert (w.effect < 0).all()                                     # O minus N
    q = w[w.model == "qwen3-8b"]
    assert (q.holm_p >= q.p_perm - 1e-12).all() and (q.holm_p_all_models >= q.holm_p - 1e-12).all()


def test_replication_table_has_joint_ranking_row_for_qwen_only():
    rng = np.random.default_rng(3)
    df = synthetic_cells(rng, 0.0)
    df = pd.concat([df, pd.DataFrame([dict(model="qwen3-8b", cell="L10", wording="O", group_id=f"L{k}", formation=0.05,
                                          comm_decision=0.05, report_rate=0.4) for k in range(6)])])
    rep = A.replication_table(A.run_contrasts(df, A.load_human(), rng, "all"))
    joint = rep[rep.contrast.str.startswith("joint")]
    assert list(joint.model) == ["qwen3-8b"]
    assert "H-word" not in set(rep.finding)


def test_run_check_detects_run_effect_but_not_cell_effect():
    rng = np.random.default_rng(4)
    rows = []
    for cell in ("C", "F10", "T"):
        for k in range(12):
            run = ["a", "b", "c"][k % 3]
            rows.append(dict(model="qwen3-8b", cell=cell, wording="O", run=run, n_periods=90, calls_per_pair_period=4.0,
                             formation=0.2 * (cell == "C") + rng.normal(0, 0.02), comm_decision=0.3 + rng.normal(0, 0.02) + 0.3 * (run == "c"),
                             report_rate=rng.normal(0, 0.02), avg_price=5 + rng.normal(0, 0.1)))
    out = A.run_check(pd.DataFrame(rows), rng, n=2000).drop_duplicates("measure").set_index("measure")
    assert out.loc["formation", "p_stratified_perm"] > 0.05          # large cell effect, none by run
    assert out.loc["comm_decision", "p_stratified_perm"] < 0.01


@pytest.mark.skipif(not (A.OUT / "replication_table.md").exists(), reason="analysis not run")
def test_outputs_contain_no_banned_tokens():
    for f in A.OUT.iterdir():
        text = f.read_text()
        assert A.BANNED not in text.lower() and "\u2014" not in text, f.name
