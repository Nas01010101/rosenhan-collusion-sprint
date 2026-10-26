"""Report figures from results/*.csv|json. Writes figures/fig{1,2,3,4}_*.{pdf,png}."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cda_load import ROOT  # noqa: E402

OUT = ROOT / "figures"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"  # validated categorical slots 1-3 (dataviz palette)
PRESSURE = {"none": ("No pressure", BLUE), "ceo": ("CEO pressure", ORANGE)}
CELL = {("none", "open"): "base-seller_comms", ("none", "gagged"): "oversight",
        ("ceo", "open"): "boss_pressure", ("ceo", "gagged"): "boss_pressure-oversight"}

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.titlelocation": "left", "axes.titleweight": "bold",
    "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42,
})


def boot_ci(x, rng, b=10_000):
    x = np.asarray(x, float)
    m = rng.choice(x, (b, len(x))).mean(axis=1)
    return np.percentile(m, [2.5, 97.5])


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)


def fig2_two_by_two(rng):
    """One row of panels: open -> gagged channel, one line per pressure level; runs as dots, mean with 95% CI."""
    runs = pd.read_csv(ROOT / "results/cda_runs.csv").merge(
        pd.read_csv(ROOT / "results/mechanism_runs.csv")[["run", "coalition", "pressure"]].rename(
            columns={"pressure": "pressure_words"}), on="run")
    panels = [("P", "(a) Price index", "index (0 = competitive, 1 = monopoly)"),
              ("J", "(b) Judge score", "coordination score (1–4)"),
              ("coalition", "(c) Cites the coalition", "share of seller-rounds"),
              ("pressure_words", "(d) Cites CEO / management", "share of seller-rounds")]
    fig, axes = plt.subplots(1, 4, figsize=(6.5, 2.25))
    for ax, (col, title, ylab) in zip(axes, panels):
        ax.axhline(runs.loc[runs.condition == "base", col].mean(), color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
        for off, (p, (label, color)) in zip((-0.06, 0.06), PRESSURE.items()):
            means = []
            for x, ch in enumerate(("open", "gagged")):
                v = runs.loc[runs.condition == CELL[p, ch], col].to_numpy()
                jit = rng.uniform(-0.035, 0.035, len(v))
                ax.scatter(x + off + jit, v, s=6, color=color, alpha=0.35, lw=0, zorder=2)
                lo, hi = boot_ci(v, rng)
                ax.errorbar(x + off, v.mean(), yerr=[[v.mean() - lo], [hi - v.mean()]], fmt="o", ms=4,
                            color=color, mec="white", mew=0.8, elinewidth=1.2, capsize=0, zorder=4)
                means.append(v.mean())
            ax.plot([off, 1 + off], means, color=color, lw=1.5, zorder=3)
        ax.set_xticks([0, 1], ["chat\nopen", "chat\ngagged"])
        ax.set_xlim(-0.35, 1.35)
        ax.set_title(title, fontsize=7.5)
        ax.set_ylabel(ylab, fontsize=7)
        ax.grid(axis="x", visible=False)
    # the dashed no-channel line goes in the legend: a per-panel label would run into the next panel
    handles = [plt.Line2D([], [], color=c, marker="o", ms=4, lw=1.5, label=l) for l, c in PRESSURE.values()]
    handles.append(plt.Line2D([], [], color=INK2, lw=1, ls=(0, (4, 3)), label="no channel (base), mean"))
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.07), fontsize=7)
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig2_two_by_two")


def fig3_screens(rng):
    """AUC with 95% bootstrap CI per screen, for three contrasts."""
    sc = json.loads((ROOT / "results/cda_primary.json").read_text())["screens"]
    names = {"S1_price": "S1 price level", "S2_dispersion": "S2 low dispersion", "S3_comove": "S3 Δask co-movement",
             "S4_dep": "S4 dependence test", "S5_ties": "S5 exact-tie rate"}
    contrasts = [("auc_open_vs_absent", "(a) Chat open vs\nno channel"),
                 ("auc_ceo_open_vs_ceo_gagged", "(b) CEO pressure:\nchat open vs gagged"),
                 ("auc_ceo_gagged_vs_absent", "(c) CEO + gagged vs no channel\n(judge scores differ by 0.02)")]
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.2), sharey=True)
    y = np.arange(len(names))[::-1]
    for ax, (key, title) in zip(axes, contrasts):
        ax.axvline(0.5, color=INK2, lw=1, ls=(0, (4, 3)))
        for yi, s in zip(y, names):
            a, (lo, hi) = sc[s][key]["auc"], sc[s][key]["ci95"]
            ax.plot([lo, hi], [yi, yi], color=INK2, lw=1.2, solid_capstyle="round")
            ax.plot(a, yi, "o", ms=4.5, color=INK, mec="white", mew=0.8)
            ax.annotate(f"{a:.2f}", (a, yi + 0.28), ha="center", fontsize=6.5, color=INK2,
                        bbox=dict(fc="white", ec="none", pad=0.4))  # keeps labels legible over the 0.5 line
        ax.set_xlim(-0.02, 1.02)
        ax.set_xticks([0, 0.5, 1])
        ax.set_title(title, fontsize=7.5)
        ax.set_xlabel("AUC")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, list(names.values()))
    axes[0].set_ylim(-0.6, len(names) - 0.3)
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig3_screens")


def fig4_s4_calibration():
    """ECDF of S4 p-values: calibrated on synthetic independent sellers, not on LLM markets with no channel."""
    syn = json.loads((ROOT / "results/s4_synthetic_null.json").read_text())
    runs = pd.read_csv(ROOT / "results/cda_runs.csv")
    series = [
        (syn["independent"], f"Synthetic, independent sellers (n={len(syn['independent'])})", BLUE, "-"),
        (syn["private_agreement_0.3"], f"Synthetic, private agreement (n={len(syn['private_agreement_0.3'])})",
         ORANGE, "-"),
        (runs.loc[runs.condition == "base", "S4_p"], "gpt-4.1 markets, no channel (n=10)", AQUA, "-"),
        (runs.loc[runs.condition != "base", "S4_p"], f"LLM markets with a chat channel (n={(runs.condition != 'base').sum()})",
         INK2, (0, (2, 1.5))),
    ]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.plot([0, 1], [0, 1], color=GRID, lw=1.2, zorder=1)
    ax.axvline(0.05, color=INK2, lw=0.8, ls=(0, (4, 3)))
    ax.annotate("p = 0.05", (0.065, 0.03), fontsize=6.5, color=INK2)
    for p, label, color, ls in series:
        p = np.sort(np.asarray(p, float))
        ax.step(np.r_[0, p, 1], np.r_[0, np.arange(1, len(p) + 1) / len(p), 1], where="post",
                color=color, lw=1.5, ls=ls, label=label)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("p-value of the dependence test (S4)")
    ax.set_ylabel("share of markets with p ≤ x")
    ax.legend(frameon=False, fontsize=6.3, loc="upper left", bbox_to_anchor=(0, -0.2))
    save(fig, "fig4_s4_calibration")


def fig1_trajectories(rng):
    """Per-round price index and judge score per cell, mean over runs with a 95% bootstrap band."""
    traj = pd.read_csv(ROOT / "results/cda_trajectories.csv")
    lines = [(CELL["ceo", "open"], "CEO pressure, chat open", ORANGE, "-"),
             (CELL["ceo", "gagged"], "CEO pressure, chat gagged", ORANGE, (0, (4, 2))),
             (CELL["none", "open"], "No pressure, chat open", BLUE, "-"),
             (CELL["none", "gagged"], "No pressure, chat gagged", BLUE, (0, (4, 2))),
             ("base", "No channel", INK2, "-")]
    panels = [("P", "(a) Price: collusion index per round", "index (0 = competitive, 1 = monopoly)"),
              ("J", "(b) Judge: coordination score per round", "score (1–4)")]
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.6))
    for ax, (col, title, ylab) in zip(axes, panels):
        ax.axvspan(19.5, 29.5, color=GRID, alpha=0.5, lw=0, zorder=0)
        for cond, label, color, ls in lines:
            w = traj[traj.condition == cond].pivot(index="run", columns="round", values=col).to_numpy()
            m = np.nanmean(w, axis=0)
            boot = np.stack([np.nanmean(w[rng.integers(0, len(w), len(w))], axis=0) for _ in range(2000)])
            lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
            t = np.arange(w.shape[1])
            ax.fill_between(t, lo, hi, color=color, alpha=0.10, lw=0, zorder=1)
            ax.plot(t, m, color=color, ls=ls, lw=1.5, label=label, zorder=2)
        ax.set_xlim(-0.5, 29.5)
        ax.set_xticks([0, 10, 20, 29])
        ax.set_xlabel("round")
        ax.set_title(title)
        ax.set_ylabel(ylab)
    # direct labels on the price panel only; the two pressured lines end 0.003 apart, so they share one label
    ends = {"CEO pressure,\nopen and gagged": CELL["ceo", "open"], "No pressure, open": CELL["none", "open"],
            "No channel": "base", "No pressure, gagged": CELL["none", "gagged"]}
    y_end = {k: traj.P[(traj.condition == c) & (traj["round"] >= 27)].mean() for k, c in ends.items()}
    placed = []
    for k in sorted(y_end, key=y_end.get, reverse=True):
        y = min([y_end[k]] + [p - 0.11 for p in placed[-1:]])
        placed.append(y)
        axes[0].annotate(k, (30.0, y), fontsize=6.5, color=INK, va="center", ha="left", annotation_clip=False)
    axes[0].axhline(0, color=INK2, lw=0.8, ls=(0, (1, 2)), zorder=1)
    axes[1].annotate("rounds 20–29:\nanalysis window", (24.5, axes[1].get_ylim()[0]), ha="center", va="bottom",
                     fontsize=6.5, color=INK2)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False, fontsize=7,
               bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout(w_pad=7.5)
    save(fig, "fig1_trajectories")


def main():
    rng = np.random.default_rng(0)
    if (ROOT / "results/cda_trajectories.csv").exists():
        fig1_trajectories(rng)
    fig2_two_by_two(rng)
    fig3_screens(rng)
    if (ROOT / "results/s4_synthetic_null.json").exists():
        fig4_s4_calibration()


if __name__ == "__main__":
    main()
