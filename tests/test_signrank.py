"""signrank matches scipy's exact Wilcoxon without ties and brute-force sign enumeration with ties."""
import itertools
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from signrank import signrank  # noqa: E402


def test_matches_scipy_exact_without_ties():
    rng = np.random.default_rng(1)
    for n in (5, 12, 20, 30):
        x, y = rng.normal(size=n) + 0.3, rng.normal(size=n)
        for alt in ("two-sided", "greater", "less"):
            assert abs(signrank(x, y, alt)["p"] - wilcoxon(x, y, alternative=alt, method="exact").pvalue) < 1e-12


def test_matches_enumeration_with_ties_and_zeros():
    rng = np.random.default_rng(2)
    for _ in range(20):
        d = rng.integers(-3, 4, size=10) / 3
        dd = d[d != 0]
        r = rankdata(np.abs(dd))
        t = r[dd > 0].sum()
        ts = np.array([sum(r[i] for i in range(len(dd)) if s[i]) for s in itertools.product([0, 1], repeat=len(dd))])
        want = min(1, 2 * min((ts >= t - 1e-9).mean(), (ts <= t + 1e-9).mean())) if len(dd) else 1.0
        assert abs(signrank(d, np.zeros(10))["p"] - want) < 1e-12


def test_floor_one_sided_n20():
    assert signrank(np.ones(20), np.zeros(20), "greater")["p"] == 2 ** -20
