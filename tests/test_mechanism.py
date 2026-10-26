import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from exp_mechanism import message_rates, price_signal, reasoning_rates  # noqa: E402


def test_price_signal_reads_prices_with_or_without_the_decimal_point():
    assert price_signal("9248", 92.48) == 92.48
    assert price_signal("96.4", 96.46) == 96.4
    assert price_signal("914", 91.0) == 91.4
    # round counters, codes and words are not prices
    assert price_signal("30", 94.1) is None
    assert price_signal("60c", 96.0) is None
    assert price_signal("ok", 96.0) is None
    assert price_signal("9248", np.nan) is None


def test_reasoning_rates_count_seller_rounds_not_phrases():
    T = 2
    run = SimpleNamespace(
        evidence=[[["the CEO wants margins", "hold with the coalition"], []], [["undercut by a cent"], ["ok"]]],
        judge_type=[["tacit", "null"], ["overt", None]],
    )
    r = reasoning_rates(run, w=slice(0, T))
    assert r["pressure"] == 0.25 and r["coalition"] == 0.25
    assert r["phrases_per_seller_round"] == 1.0
    assert r["type_tacit"] == 0.25 and r["type_overt"] == 0.25


def test_message_rates():
    run = SimpleNamespace(
        sellers=["a", "b"], asks=np.array([[92.48, 92.4], [93.0, 93.0]]),
        messages=[{"a": "9248", "b": "ok"}, {"a": "", "b": "~"}],
    )
    r = message_rates(run, w=slice(0, 2))
    assert r["non_ok"] == 0.5 and r["price_signal"] == 0.25


def test_boot_spearman_monotone_gives_one():
    from exp_mechanism import boot_spearman

    r = boot_spearman(np.arange(10), np.arange(10) ** 2, np.random.default_rng(0), b=200)
    assert np.allclose([r["rho"], *r["ci95"]], 1.0)
