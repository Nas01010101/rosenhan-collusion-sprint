import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cda_screens import actions, comovement, dependence_test, market_state, tie_rate  # noqa: E402


def fake_run(asks, traded):
    return SimpleNamespace(asks=np.asarray(asks, float), trades=[[{}] if x else [] for x in traded])


def common_cause_market(rng, agree_prob=0.0, T=30, n=5):
    """Sellers cut after a no-trade round and hold after a trade, independently given that public fact.

    With agree_prob > 0, in that share of rounds every seller instead plays one privately agreed move,
    which no public state explains (a cartel using a private channel).
    """
    traded = rng.random(T) < 0.5
    moves = np.zeros((n, T))
    for t in range(1, T):
        p_cut = 0.15 if traded[t - 1] else 0.85
        moves[:, t] = np.where(rng.random(n) < p_cut, -0.1, 0.0)
        if rng.random() < agree_prob:
            moves[:, t] = rng.choice([-0.1, 0.0])
    return fake_run(95 + np.cumsum(moves, axis=1), traded)


def test_actions_and_state_encode_previous_round():
    r = fake_run([[95, 94, 94, np.nan], [95, 95, 96, 96]], [False, True, False, False])
    np.testing.assert_array_equal(actions(r), [[1, 0, 1, 1], [1, 1, 2, 1]])
    # t=1 sees no trade at 0; t=2 sees the trade at 1 and the min ask falling 95 -> 94 at 1
    np.testing.assert_array_equal(market_state(r), [0, 0, 3, 0])


def test_tie_rate_counts_seller_rounds_matching_a_rival():
    r = fake_run([[90.0, 91.0], [90.0, 92.0], [89.0, 93.0]], [False, False])
    assert tie_rate(r, w=slice(0, 2)) == 2 / 6


def test_dependence_test_holds_level_under_common_cause():
    rng = np.random.default_rng(1)
    runs = [common_cause_market(rng) for _ in range(100)]
    pvals = [dependence_test(r, w=slice(1, 30), n_perm=200, rng=k)[1] for k, r in enumerate(runs)]
    # raw co-movement is high because every seller reacts to the same trades
    assert np.mean([comovement(r, w=slice(1, 30)) for r in runs]) > 0.3
    assert np.mean(np.array(pvals) < 0.05) <= 0.10


def test_dependence_test_detects_private_agreement():
    # the statistic sums over all pairs, so it targets group agreements; a single copying pair
    # (seller 1 copies seller 0 in 80% of rounds) is flagged in only ~40% of 30-round markets
    rng = np.random.default_rng(2)
    runs = [common_cause_market(rng, agree_prob=0.3) for _ in range(30)]
    pvals = [dependence_test(r, w=slice(1, 30), n_perm=200, rng=k)[1] for k, r in enumerate(runs)]
    assert np.mean(np.array(pvals) < 0.05) >= 0.6


def test_flags_counts_below_alpha_with_exact_ci():
    from exp_s4_null import flags
    f = flags([0.01, 0.2, 0.04, 0.5])
    assert f["k"] == 2 and f["n"] == 4 and f["rate"] == 0.5
    assert f["ci95"][0] < 0.5 < f["ci95"][1]
    assert flags([0.9] * 10)["ci95"][0] == 0.0
