import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from exp_cda import did_boot  # noqa: E402


def test_did_boot_recovers_interaction_and_covers_zero_without_one():
    rng = np.random.default_rng(0)
    noise = lambda m: m + rng.normal(0, 0.1, 10)  # noqa: E731
    hit = did_boot(noise(0.0), noise(-0.4), noise(0.6), noise(0.6), rng, b=2000)
    assert abs(hit["did"] - 0.4) < 0.15 and hit["ci95"][0] > 0
    flat = did_boot(noise(0.0), noise(-0.4), noise(0.6), noise(0.2), rng, b=2000)
    assert flat["ci95"][0] < 0 < flat["ci95"][1]


def test_trajectories_one_row_per_round_and_nan_for_withdrawn_rounds():
    from types import SimpleNamespace

    from exp_cda import trajectories

    judge = np.array([[1.0, np.nan, 3.0], [3.0, np.nan, 3.0]])
    run = SimpleNamespace(name="r", condition="base", judge=judge,
                          metrics={"collusion_indices_avg_seller_asks": [0.5, None, 0.1]})
    t = trajectories([run])
    assert list(t["round"]) == [0, 1, 2]
    assert t.P[0] == 0.5 and np.isnan(t.P[1]) and t.P[2] == 0.1
    assert t.J[0] == 2.0 and np.isnan(t.J[1]) and t.J[2] == 3.0


def test_boot_mean_ci_brackets_mean_and_collapses_on_constant():
    from exp_cda import boot_mean

    rng = np.random.default_rng(0)
    r = boot_mean(rng.normal(1.0, 0.2, 10), rng, b=2000)
    assert r["ci95"][0] < r["mean"] < r["ci95"][1]
    assert np.allclose(boot_mean(np.full(10, 0.3), rng, b=200)["ci95"], [0.3, 0.3])
