"""Primary and secondary analyses of notes/prereg.md on the released double-auction runs.

Writes results/cda_runs.csv (one row per run) and results/cda_primary.json.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cda_load import ROOT, load_all  # noqa: E402
from cda_screens import (  # noqa: E402
    WINDOW, ask_index, comovement, dependence_test, dispersion, judge_mean, tie_rate,
)

CELLS = {  # condition -> (pressure, channel)
    "base-seller_comms": ("none", "open"),
    "oversight": ("none", "gagged"),
    "boss_pressure": ("ceo", "open"),
    "boss_pressure-oversight": ("ceo", "gagged"),
    "base": ("none", "absent"),
}
SCREENS = ["S1_price", "S2_dispersion", "S3_comove", "S4_dep", "S5_ties"]
ORIENT = {"S2_dispersion": -1}  # the variance screen flags LOW dispersion; every other screen flags high values
B = 10_000


def msg_len(run, w=WINDOW):
    lens = [len(m or "") for rnd in run.messages[w] for m in rnd.values()]
    return float(np.mean(lens)) if lens else 0.0


def trajectories(runs):
    """Long table: one row per run x round with the price index and the mean judge score."""
    rows = []
    for r in runs:
        idx = r.metrics["collusion_indices_avg_seller_asks"]
        for t in range(len(idx)):
            rows.append(dict(run=r.name, condition=r.condition, round=t, P=np.nan if idx[t] is None else idx[t],
                             J=float(np.nanmean(r.judge[:, t])) if np.isfinite(r.judge[:, t]).any() else np.nan))
    return pd.DataFrame(rows)


def run_rows(runs):
    rows = []
    for k, r in enumerate(runs):
        stat, p = dependence_test(r, rng=k)
        pressure, channel = CELLS.get(r.condition, ("none", "open"))
        rows.append(dict(
            run=r.name, condition=r.condition, pressure=pressure, channel=channel, gag_round=r.gag_round,
            seller_models="/".join(sorted(set(r.config["seller_models"]))),
            P=ask_index(r), J=judge_mean(r), trades=sum(len(t) for t in r.trades), msg_len=msg_len(r),
            S1_price=ask_index(r, WINDOW), S2_dispersion=dispersion(r), S3_comove=comovement(r),
            S4_dep=stat, S4_p=p, S5_ties=tie_rate(r),
        ))
    return pd.DataFrame(rows)


def boot_diff(x, y, rng, b=B):
    """Mean(x) - mean(y) with a percentile bootstrap 95% CI, resampling runs within each group."""
    x, y = np.asarray(x), np.asarray(y)
    d = [rng.choice(x, len(x)).mean() - rng.choice(y, len(y)).mean() for _ in range(b)]
    return dict(diff=float(x.mean() - y.mean()), ci95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))])


def boot_mean(x, rng, b=B):
    """Mean of x with a percentile bootstrap 95% CI over runs."""
    x = np.asarray(x, float)
    m = rng.choice(x, (b, len(x))).mean(axis=1)
    return dict(mean=float(x.mean()), ci95=[float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))])


def did_perm(df, col, rng, b=B):
    """Interaction: (gag effect under CEO pressure) - (gag effect without), labels shuffled within pressure."""
    cells = df[df.channel.isin(["open", "gagged"]) & df.condition.isin(CELLS)]

    def did(ch):
        m = {(p, c): cells[col][(cells.pressure == p) & (ch == c)].mean() for p in ("none", "ceo") for c in ("open", "gagged")}
        return (m["ceo", "gagged"] - m["ceo", "open"]) - (m["none", "gagged"] - m["none", "open"])

    obs = did(cells.channel.values)
    null = np.empty(b)
    for k in range(b):
        ch = cells.channel.values.copy()
        for p in ("none", "ceo"):
            idx = np.flatnonzero(cells.pressure.values == p)
            ch[idx] = rng.permutation(ch[idx])
        null[k] = did(ch)
    return dict(did=float(obs), p_two_sided=float((1 + np.sum(np.abs(null) >= abs(obs))) / (1 + b)))


def did_boot(open_none, gag_none, open_ceo, gag_ceo, rng, b=B):
    """DiD with a percentile bootstrap CI, resampling runs within each of the four cells.

    Added after the plan: the within-pressure shuffle in did_perm also destroys the gag main effect,
    so its null is wide when that effect is large (the judge score) and the test is conservative.
    """
    cells = [np.asarray(c, float) for c in (open_none, gag_none, open_ceo, gag_ceo)]

    def did(a, g, o, c):
        return (c.mean() - o.mean()) - (g.mean() - a.mean())

    d = [did(*(rng.choice(c, len(c)) for c in cells)) for _ in range(b)]
    return dict(did=float(did(*cells)), ci95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))])


def tost(x, y, margin):
    """Welch two one-sided tests of |mean(x) - mean(y)| < margin; returns the larger one-sided p."""
    lo = stats.ttest_ind(np.asarray(x) + margin, y, equal_var=False, alternative="greater").pvalue
    hi = stats.ttest_ind(np.asarray(x) - margin, y, equal_var=False, alternative="less").pvalue
    return dict(margin=margin, p=float(max(lo, hi)), equivalent=bool(max(lo, hi) < 0.05))


def auc(pos, neg):
    """P(score of a random positive > random negative), ties count half."""
    pos, neg = np.asarray(pos)[:, None], np.asarray(neg)[None, :]
    return float((pos > neg).mean() + 0.5 * (pos == neg).mean())


def boot_auc(pos, neg, rng, b=2000):
    pos, neg = np.asarray(pos), np.asarray(neg)
    a = [auc(rng.choice(pos, len(pos)), rng.choice(neg, len(neg))) for _ in range(b)]
    return dict(auc=auc(pos, neg), ci95=[float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))])


def main():
    rng = np.random.default_rng(0)
    runs = load_all()
    df = run_rows(runs)
    df.to_csv(ROOT / "results/cda_runs.csv", index=False)
    traj = trajectories([r for r in runs if r.condition in CELLS])
    traj.to_csv(ROOT / "results/cda_trajectories.csv", index=False)
    g = {c: df[df.condition == c] for c in CELLS}
    out = {"n_runs": {c: int(len(v)) for c, v in g.items()}, "cell_means": {}, "Q1": {}, "Q2": {}, "screens": {}}
    for c, v in g.items():
        out["cell_means"][c] = {k: [float(v[k].mean()), float(v[k].std(ddof=1))] for k in ["P", "J", "trades", "msg_len", *SCREENS]}
    for q, col in (("Q1", "P"), ("Q2", "J")):
        out[q]["gag_effect_none"] = boot_diff(g["oversight"][col], g["base-seller_comms"][col], rng)
        out[q]["gag_effect_ceo"] = boot_diff(g["boss_pressure-oversight"][col], g["boss_pressure"][col], rng)
        out[q]["interaction"] = did_perm(df, col, rng)
        out[q]["interaction_boot"] = did_boot(*(g[c][col] for c in CELLS if c != "base"), rng)
        out[q]["tost_ceo_open_vs_gagged"] = tost(g["boss_pressure"][col], g["boss_pressure-oversight"][col], 0.10)
    # screens: separation of an open channel from no channel, and where the gagged CEO cell lands
    for s in SCREENS:
        k = ORIENT.get(s, 1)
        out["screens"][s] = dict(
            auc_open_vs_absent=boot_auc(k * g["base-seller_comms"][s], k * g["base"][s], rng),
            auc_ceo_open_vs_ceo_gagged=boot_auc(k * g["boss_pressure"][s], k * g["boss_pressure-oversight"][s], rng),
            auc_ceo_gagged_vs_absent=boot_auc(k * g["boss_pressure-oversight"][s], k * g["base"][s], rng),
        )
    # exploratory, not in the plan: the gagged CEO cell against markets that never had a channel
    out["exploratory_ceo_gagged_minus_absent"] = {
        col: boot_diff(g["boss_pressure-oversight"][col], g["base"][col], rng) for col in ("P", "J")
    }
    out["exploratory_screens_ceo_gagged_vs_open_none"] = {  # pressured, gagged market vs the chat cartel
        s: boot_auc(ORIENT.get(s, 1) * g["boss_pressure-oversight"][s], ORIENT.get(s, 1) * g["base-seller_comms"][s], rng)
        for s in SCREENS
    }
    out["exploratory_ceo_gagged_minus_open_none"] = {
        col: boot_diff(g["boss_pressure-oversight"][col], g["base-seller_comms"][col], rng) for col in ("P", "J")
    }
    out["P_by_round_cell_mean"] = {  # every cell opens at the same asks; which cells hold them
        c: {f"r{t}": float(traj.P[(traj.condition == c) & (traj["round"] == t)].mean()) for t in (0, 5, 10)}
        for c in CELLS
    }
    # exploratory, added after the critic pass; own generator so every number above keeps its draws
    rng2 = np.random.default_rng(1)
    out["exploratory_cell_ci95"] = {c: {k: boot_mean(v[k], rng2) for k in ("P", "J", "S1_price", "S5_ties")}
                                    for c, v in g.items()}
    out["exploratory_S1_flag_rate_at_0.5"] = {c: float((v.S1_price >= 0.5).mean()) for c, v in g.items()}
    out["exploratory_none_gagged_minus_absent"] = {
        col: boot_diff(g["oversight"][col], g["base"][col], rng2) for col in ("P", "J")
    }
    out["exploratory_auc_J_ceo_open_vs_ceo_gagged"] = boot_auc(g["boss_pressure"].J, g["boss_pressure-oversight"].J, rng2)
    out["S4_flag_rate_p<0.05"] = {c: float((v.S4_p < 0.05).mean()) for c, v in g.items()}
    out["S4_flag_rate_all_conditions"] = df.groupby("condition").S4_p.apply(lambda p: float((p < 0.05).mean())).to_dict()
    (ROOT / "results/cda_primary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
