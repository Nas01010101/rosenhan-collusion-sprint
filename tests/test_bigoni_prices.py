import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import bigoni_analysis as A  # noqa: E402
import bigoni_prices as P  # noqa: E402

RAW_OK = (A.RAW / A.DIR.get("qwen3-8b", "qwen3-8b") / "groups").exists()


def per(t, prices, pair=(0, 1), live=False, formed=False, agreed=None, new=False, dflt=(False, False)):
    return {"period": t, "pair": list(pair), "new_match": new, "ring_live_before": live, "formed": formed, "agreed": agreed,
            "prices": list(prices), "price_default": list(dflt), "comm": [formed, formed]}


def grp(periods):
    return {"spec": {"model": "m", "cell": "C", "wording": "O", "group_id": "g"}, "periods": periods}


def test_ring_outside_and_agreed_split():
    r = P.price_group_measures(grp([per(1, (3, 5), new=True),                              # outside: no ring
                                    per(2, (9, 9), formed=True, agreed=9),                 # forms
                                    per(3, (9, 7), live=True),                             # live ring, one undercut
                                    per(4, (2, 2), live=False)]))                          # ring over: outside again
    assert (r["outside_price"], r["outside_price_n"]) == (3.0, 4)
    assert (r["ring_price"], r["ring_price_n"]) == (pytest.approx(34 / 4), 4)
    assert (r["agreed_price"], r["agreed_price_n"]) == (9.0, 1)
    assert r["avg_price"] == pytest.approx(46 / 8)
    assert r["no_new_agreement_price"] == pytest.approx(28 / 6)       # includes the live-ring period
    assert (r["deviation_rate"], r["deviation_rate_n"]) == (0.25, 4)  # period 2: 0 of 2, period 3: 1 of 2


def test_deviation_counts_only_until_first_undercut_and_resets_on_new_agreement_and_match():
    ps = [per(1, (9, 9), formed=True, agreed=9, new=True), per(2, (8, 9), live=True),      # undercut: counted, 1 of 2
          per(3, (5, 5), live=True),                                                       # agreement already broken: skipped
          per(4, (7, 7), formed=True, agreed=7),                                           # new agreement restarts the count
          per(5, (7, 6), live=True),                                                       # 1 of 2
          per(6, (1, 1), live=True, new=True)]                                             # rematch: agreement dropped
    r = P.price_group_measures(grp(ps))
    assert (r["deviation_rate"], r["deviation_rate_n"]) == (pytest.approx(2 / 8), 8)


def test_defaulted_prices_dropped_and_empty_measures_are_nan():
    r = P.price_group_measures(grp([per(1, (4, 11), dflt=(False, True))]))
    assert (r["avg_price"], r["avg_price_n"]) == (4.0, 1)
    assert np.isnan(r["ring_price"]) and np.isnan(r["agreed_price"]) and np.isnan(r["deviation_rate"])


def test_signflip_exact():
    assert P.signflip_p([1.0] * 6, 0.0) == pytest.approx(2 / 64)      # smallest attainable p at 6 groups
    assert P.signflip_p([1.0, -1.0, 2.0, -2.0], 0.0) == 1.0
    assert P.signflip_p([3.0, 3.1, 2.9], 3.0) > 0.5


def test_contrasts_signs_and_missing_cells():
    rng = np.random.default_rng(5)
    rows = []
    for model in ("qwen3-8b", "olmo3-7b"):
        for cell, w, mu in [("C", "O", 7), ("C", "N", 7), ("F10", "O", 5), ("F10", "N", 7), ("T", "O", 5), ("T", "N", 7), ("R10", "O", 5)]:
            for k in range(6):
                v = mu + rng.normal(0, 0.1)
                rows.append(dict(model=model, cell=cell, wording=w, group_id=f"{model}{cell}{w}{k}", **{m: v for m in P.MEASURES}))
    t = P.contrasts(pd.DataFrame(rows), A.load_human(srcs=P.PRICE_SRC), rng)
    w = t[(t.finding == "P-word") & (t.measure == "avg_price") & (t.model == "qwen3-8b")].set_index("contrast")
    assert w.loc["F10_O minus F10_N", "effect"] == pytest.approx(-2, abs=0.2) and w.loc["F10_O minus F10_N", "p_perm"] < 0.01
    assert abs(w.loc["C_O minus C_N", "effect"]) < 0.3
    assert not t[(t.finding == "P-word")].contrast.str.startswith("R10").any()       # no R10_N in this design
    h = t[(t.finding == "P-human") & (t.measure == "outside_price") & (t.contrast == "C_O minus human")].iloc[0]
    assert h.effect == pytest.approx(7 - 3.5, abs=0.2) and h.mean_b == 3.5
    assert (t[t.finding == "P-nash"].mean_b == 3).all() and (t[t.finding == "P-joint"].mean_b == 9).all()
    assert (t.dropna(subset=["holm_p"]).eval("holm_p >= p_perm - 1e-12")).all()


def test_human_price_targets_load_with_renamed_measure():
    h = A.load_human(srcs=P.PRICE_SRC)
    assert h[("outside_price", "C")][0] == 3.5 and h[("outside_price", "R10")][0] == 3.567
    assert h[("avg_price", "F10")][0] == 5.349 and ("outside_price", "T") not in h


@pytest.mark.skipif(not RAW_OK, reason="raw logs not present")
def test_price_measures_agree_with_main_analysis_on_real_groups():
    groups = A.load_groups()
    for d in groups[::9]:
        mine, theirs = P.price_group_measures(d), A.group_measures(d)
        for m in ("avg_price", "ring_price"):
            assert mine[m] == pytest.approx(theirs[m], nan_ok=True) and mine[m + "_n"] == theirs[m + "_n"]
        assert mine["outside_price_n"] + mine["ring_price_n"] == mine["avg_price_n"]
