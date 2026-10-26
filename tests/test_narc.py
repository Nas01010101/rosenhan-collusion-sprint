import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from narc_load import DATA, Groups, build_groups, load_tier  # noqa: E402
from narc_probes import auc_rows, boot_auc, fit_second_stage, final_scores, group_scores  # noqa: E402

D = 16
rng = np.random.default_rng(0)


class _Scaler:
    mean_ = np.zeros(D)
    scale_ = np.ones(D)


def _groups(n=40, shift=1.5):
    """Collusion runs: two agents shifted along axis 0, two honest. Control: none shifted."""
    y = np.r_[np.ones(n // 2), np.zeros(n // 2)].astype(int)
    X = rng.normal(size=(n, 4, D))
    X[y == 1, :2, 0] += shift
    return Groups([f"r{i}" for i in range(n)], y, ["f"] * n, ["d"] * n, X, [()] * n)


W = np.eye(D)[0]


def test_auc_rows_matches_sklearn_with_ties():
    s = rng.integers(0, 4, size=(3, 30)).astype(float)
    y = rng.integers(0, 2, 30)
    ref = [roc_auc_score(y, r) for r in s]
    assert np.allclose(auc_rows(s, y), ref)


def test_split_and_peak_separate_planted_signal():
    G = _groups(shift=5.0)
    sc = group_scores(G, _Scaler, W)
    assert roc_auc_score(G.y, sc["deception_split"]) > 0.9
    assert roc_auc_score(G.y, sc["peak_suspicion"]) > 0.9


def test_scores_invariant_to_agent_order():
    G = _groups()
    perm = np.array([2, 0, 3, 1])
    a, b = group_scores(G, _Scaler, W), group_scores(Groups(G.run_ids, G.y, G.family, G.domain, G.X[:, perm], G.roles), _Scaler, W)
    for k in a:
        assert np.allclose(a[k], b[k], atol=1e-6), k


def test_sorted_concat_puts_most_suspicious_agent_first():
    G = _groups()
    sc = group_scores(G, _Scaler, W)
    first = sc["sorted_concat"][:, :D]
    assert np.all(first[:, 0] >= sc["sorted_concat"][:, 3 * D])  # slot 0 projects >= slot 3 on w


def test_activation_alignment_is_negated_mean_pairwise_cosine():
    G = _groups(n=4)
    v = G.X[0] / np.linalg.norm(G.X[0], axis=1, keepdims=True)
    c = v @ v.T
    expect = -c[np.triu_indices(4, 1)].mean()
    assert np.isclose(group_scores(G, _Scaler, W)["activation_alignment"][0], expect, atol=1e-5)


def test_second_stage_scores_in_unit_interval_and_detect_signal():
    tr, te = _groups(60, shift=4.0), _groups(40, shift=4.0)
    stage = fit_second_stage(group_scores(tr, _Scaler, W), tr.y)
    fs = final_scores(group_scores(te, _Scaler, W), stage)
    assert roc_auc_score(te.y, fs["asymmetry_probe"]) > 0.9
    assert all(((v >= 0) & (v <= 1)).all() for k, v in fs.items() if k in ("asymmetry_probe", "sorted_concat"))


def test_bootstrap_resamples_runs_and_ci_brackets_point_estimate():
    y = np.r_[np.ones(10), np.zeros(10)].astype(int)
    S = (y + rng.normal(0, 0.8, 20))[None]
    d = boot_auc(S, y, None, np.random.default_rng(1), b=500)
    point = auc_rows(S, y)[0]
    assert np.percentile(d, 2.5) <= point <= np.percentile(d, 97.5)
    assert np.percentile(d, 97.5) - np.percentile(d, 2.5) > 0.05  # n=20 runs cannot give a tight interval


needs_data = pytest.mark.skipif(not (DATA / "activations/qwen3_32b/core").exists(), reason="NARCBench not downloaded")


@needs_data
def test_hf_v3_counts_and_shapes():
    for tier, runs, per_class in (("core", 62, 31), ("transfer", 72, 36), ("stego", 40, 20)):
        g = build_groups(load_tier(tier), 28, "discussion" if tier == "core" else "final")
        assert g.X.shape == (runs, 4, 5120)
        assert g.y.sum() == per_class and (1 - g.y).sum() == per_class
    assert all(r.count("colluder") == 2 for r, y in zip(build_groups(load_tier("core"), 28).roles, build_groups(load_tier("core"), 28).y) if y)


@needs_data
def test_transfer_has_six_families_of_twelve_runs():
    g = build_groups(load_tier("transfer"), 28, "final")
    fams, counts = np.unique(g.family, return_counts=True)
    assert len(fams) == 6 and set(counts) == {12}
