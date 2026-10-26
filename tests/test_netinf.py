import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from netinf import cmi, detect, permutation_pvalues, prev_joint_state  # noqa: E402
from rps import play  # noqa: E402


def test_cmi_identical_uniform_is_log3():
    x = np.tile([0, 1, 2], 1000)
    assert abs(cmi(x, x) - np.log(3)) < 1e-9


def test_cmi_independent_near_zero_large_n():
    rng = np.random.default_rng(0)
    x, y = rng.integers(0, 3, 50_000), rng.integers(0, 3, 50_000)
    assert cmi(x, y) < 1e-3


def test_cmi_conditioning_removes_common_cause():
    # x and y both copy s: dependent marginally, independent given s
    rng = np.random.default_rng(1)
    s = rng.integers(0, 3, 5000)
    assert cmi(s, s) > 1.0
    assert cmi(s, s, s) < 1e-9


def test_full_collusion_detected_at_60_games():
    # paper Table 2: CP=1 reaches 100% detection accuracy with 60 games
    hits = [detect(play(60, 1.0, rng=k)) == (0, 1) for k in range(200)]
    assert np.mean(hits) > 0.99


def test_paper_rule_returns_correct_pair_after_relabel():
    acts = play(500, 1.0, rng=3, order=(2, 0, 1))  # rows: C, A, B -> pair (1, 2)
    assert detect(acts) == (1, 2)


def test_permutation_pvalue_small_for_colluders_uniform_for_null():
    acts = play(200, 0.6, rng=4)
    res = permutation_pvalues(acts, n_perm=300, rng=0)
    assert res[(0, 1)][1] < 0.01
    null_p = [permutation_pvalues(play(30, 0.0, rng=k), n_perm=200, rng=k)[(0, 2)][1] for k in range(60)]
    assert 0.3 < np.mean(null_p) < 0.7  # roughly uniform under H0


def test_prev_joint_state_encodes_previous_round():
    acts = np.array([[0, 2, 1], [1, 0, 2], [2, 1, 0]])
    assert prev_joint_state(acts).tolist() == [27, 0 * 9 + 1 * 3 + 2, 2 * 9 + 0 * 3 + 1]


def test_prev_joint_state_reproduces_paper_table2():
    # Bonjour et al. Table 2 (alpha=0.05, 1000 iterations): CP=0.3/200 games 48.3%, CP=0.4/100 games 58.4%
    for cp, n, paper in [(0.3, 200, 0.483), (0.4, 100, 0.584)]:
        hits = [detect(a, prev_joint_state(a)) == (0, 1) for a in (play(n, cp, rng=k) for k in range(300))]
        assert abs(np.mean(hits) - paper) < 0.10, (cp, n, np.mean(hits))
