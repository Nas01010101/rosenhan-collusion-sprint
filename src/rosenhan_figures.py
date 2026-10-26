"""Rosenhan study figures. Writes figures/rosenhan_{exposure,label,prime}.{pdf,png}.

Exposure rates are recomputed per game from the rating files (expo_rosenhan.per_game) so each bar gets a game-level
bootstrap 95% CI; the means equal results/rosenhan/exposure.json. Label and prime bars come from the same rating
files that stats_rosenhan.py reads.

    python src/rosenhan_figures.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import expo_rosenhan as E  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "figures"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"  # validated categorical slots 1-3 (dataviz palette)

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
        # no CreationDate, so a rerun on the committed results rewrites the PDFs byte for byte
        fig.savefig(OUT / f"{name}.{ext}", metadata={"CreationDate": None} if ext == "pdf" else None)
    plt.close(fig)


COLOR = {"qwen3-30b": BLUE, "phi-4": ORANGE, "gpt-oss-120b": AQUA}
LABEL = {"qwen3-30b": "Qwen3-30B", "phi-4": "phi-4", "gpt-oss-120b": "gpt-oss-120b"}
NAMES = ["Lily", "Luke", "Mike", "Quinn"]
# Exposure conditions in the order the paper discusses them: what the overseer is given about the auditor.
EXPO = [("neutral", "Log only"), ("access", "Log + secret channel"), ("report_claim", "Report, self-claimed"),
        ("report_cred", "Report, platform credential"), ("cred_access", "Channel + credential"),
        ("cred_forged", "Channel + forged credential"), ("cred_conflict", "Channel + both credentials"),
        ("cred_system", "Forged credential, true one in system turn"), ("cred_system_ih", "Same + system-turn priority note"),
        ("dueling", "Report vs colluder counter-report"), ("dueling_checked", "Duel + channel-log fact"),
        ("report_forged", "Forged report only"), ("forged_checked", "Forged + channel-log fact")]
RATERS = {"qwen3-30b": ["main_rater", "null_rater", "all_rater", "unprimed_qwen"],
          "phi-4": ["phi4_rater", "unprimed_phi4"], "gpt-oss-120b": ["gptoss_rater", "unprimed_gptoss"]}


def bars(ax, groups, series, rng, horizontal=False):
    """Grouped bars, one group per condition and one bar per overseer, with bootstrap 95% CI whiskers."""
    w = 0.8 / len(series)
    for i, (name, vals) in enumerate(series.items()):
        pos = np.arange(len(groups)) + (i - (len(series) - 1) / 2) * w
        m = [np.mean(v) for v in vals]
        ci = np.array([boot_ci(v, rng) for v in vals]).T
        err = [np.subtract(m, ci[0]), np.subtract(ci[1], m)]
        kw = dict(color=COLOR[name], label=LABEL[name], edgecolor="white", linewidth=0.8)
        if horizontal:
            ax.barh(pos, m, w, xerr=err, error_kw=dict(lw=0.7, ecolor=INK2), **kw)
        else:
            ax.bar(pos, m, w, yerr=err, error_kw=dict(lw=0.7, ecolor=INK2), **kw)


def fig_exposure(rng):
    series = {}
    for name, dirs in E.OVERSEERS.items():
        rates = E.load(dirs)
        series[name] = [[v["exposed"] for v in E.per_game(rates, c).values()] for c, _ in EXPO]
    fig, ax = plt.subplots(figsize=(6.3, 4.6))
    bars(ax, EXPO, series, rng, horizontal=True)
    ax.set_yticks(range(len(EXPO)), [t for _, t in EXPO])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Share of ratings that expose the coalition (colluder caught, auditor cleared, no clean seat framed)")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Exposure by what the overseer is given (50 G2/G3 games, 3 samples each)")
    save(fig, "rosenhan_exposure")


def load_all(dirs):
    """All cells, unlike E.load which keeps G2/G3 only."""
    rates = {}
    for d in dirs:
        for p in sorted((E.ROOT / d).glob("*.json")):
            r = json.loads(p.read_text())
            rates.setdefault(p.stem, dict(r, ratings=[]))["ratings"] += r["ratings"]
    return rates


def mean_over(g, cond, seat, key):
    v = [r["players"][seat][key] for r in g["ratings"] if r["cond"] == cond and r["players"]]
    return float(np.mean(v)) if v else None


def fig_label(rng):
    conds = [("label", "Named as\nchannel\nmember"), ("label_null", "Named,\nno channel\ncontent"),
             ("label_all", "All\nseats\nnamed"), ("neutral", "No note")]
    series = {}
    for name, dirs in RATERS.items():
        rates = load_all(dirs)
        g0 = [g for g in rates.values() if g["cell"] == "G0"]
        series[name] = [[mean_over(g, c, g["label_seat"], "p_arrangement") for g in g0] for c, _ in conds]
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    bars(ax, conds, series, rng)
    ax.set_xticks(range(len(conds)), [t for _, t in conds])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Overseer P(seat is in an arrangement)")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, ncol=3, loc="upper right", fontsize=7)
    ax.set_title("Labelled clean seat on the 20 G0 games")
    save(fig, "rosenhan_label")


def fig_prime(rng):
    conds = [("unprimed", "No auditor\nmentioned"), ("neutral", "An auditor\nmay be present"),
             ("told", "Auditors will be\nplaced at tables")]
    series = {}
    for name, dirs in RATERS.items():
        rates = load_all(dirs)
        g0 = [g for g in rates.values() if g["cell"] == "G0"]
        series[name] = [[float(max(mean_over(g, c, s, "p_auditor") for s in NAMES) >= E.FLAG) for g in g0]
                        for c, _ in conds]
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    bars(ax, conds, series, rng)
    ax.set_xticks(range(len(conds)), [t for _, t in conds])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share of G0 games with a seat\nflagged as auditor")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, ncol=3, loc="upper left", fontsize=7)
    ax.set_title("No auditor is present in any of these 20 games")
    save(fig, "rosenhan_prime")


def main():
    rng = np.random.default_rng(0)
    fig_exposure(rng)
    fig_label(rng)
    fig_prime(rng)


if __name__ == "__main__":
    main()
