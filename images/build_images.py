"""Builds images/: slides (PDF + PNG), the design diagram, copies of the paper figures and the exposure GIF.

Every number is read from results/rosenhan/{exposure,stats,stats_phi4_rater,stats_gptoss_rater}.json, except the
label-experiment means (Section 4.5 of report/rosenhan.md), which the paper reports only in text.

    /Volumes/PortableSSD/Projects/apart-collusion-sprint/.venv/bin/python images/build_images.py
"""

import json
import shutil
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "images"
RES = ROOT / "results" / "rosenhan"

INK, MUTED, GRID = "#1a1a1a", "#5c5c5c", "#e3e3e3"
OVERSEERS = ["qwen3-30b", "phi-4", "gpt-oss-120b"]
NAMES = {"qwen3-30b": "Qwen3-30B", "phi-4": "phi-4", "gpt-oss-120b": "gpt-oss-120b"}
COL = {"qwen3-30b": "#2a6fdb", "phi-4": "#e8672f", "gpt-oss-120b": "#1aa77a"}

plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "text.color": INK,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": MUTED})

EXPO = json.load(open(RES / "exposure.json"))
AUROC = {
    "qwen3-30b": json.load(open(RES / "stats.json"))["H1"],
    "phi-4": json.load(open(RES / "stats_phi4_rater.json"))["H1"],
    "gpt-oss-120b": json.load(open(RES / "stats_gptoss_rater.json"))["H1"],
}
COUNT_AUROC = AUROC["qwen3-30b"]["declined_challenge_auroc"]["count"]


def rate(model, cond, key):
    return EXPO[model]["rates"][cond][key]


# Section 4.5: mean p_arrangement (0 to 100) of the labelled seat on the 20 G0 games.
LABEL = {  # no note, names the seat only, names the seat as a channel member
    "qwen3-30b": [49.3, 68.9, 84.7],
    "phi-4": [39.0, 49.2, 82.8],
    "gpt-oss-120b": [29.2, 52.5, 84.6],
}


def grouped_bars(ax, cats, series, ylim, fmt, ylabel, fontsize=15):
    """series: {model: [value per category]}. Bars coloured by overseer; value printed above each bar."""
    n = len(series)
    w = 0.26
    for i, (m, vals) in enumerate(series.items()):
        xs = [c + (i - (n - 1) / 2) * (w + 0.02) for c in range(len(cats))]
        ax.bar(xs, vals, width=w, color=COL[m], label=NAMES[m], zorder=3)
        for x, v in zip(xs, vals):
            ax.text(x, v + ylim[1] * 0.015, fmt(v), ha="center", va="bottom", fontsize=fontsize - 3, color=INK)
    ax.set_xticks(range(len(cats)), cats, fontsize=fontsize)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel, fontsize=fontsize - 2)
    ax.yaxis.grid(True, color=GRID, zorder=0)
    ax.tick_params(axis="y", labelsize=fontsize - 3, length=0)
    ax.tick_params(axis="x", length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=fontsize - 3, loc="upper right", ncol=3, bbox_to_anchor=(1, 1.08))


def box(ax, x, y, w, h, text, fc="#ffffff", ec=MUTED, fs=13, bold=False, lw=1.3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
            fontweight="bold" if bold else "normal", linespacing=1.35)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.4))


def draw_design(ax):
    """Four game cells feeding one overseer that reads the log under different conditions."""
    ax.set_xlim(0, 100)
    ax.set_ylim(-7, 60)
    ax.axis("off")
    cells = [("G0", "20 games", "no secret channel"),
             ("G1", "20 games", "Lily and a\nreal colluder"),
             ("G2", "30 games", "Lily and an auditor\nwho writes a notebook"),
             ("G3", "20 games", "Lily and an auditor\nwith no notebook")]
    w, gap = 22, 3.9
    for i, (g, n, d) in enumerate(cells):
        x = i * (w + gap)
        ax.text(x + w / 2, 58, f"{g}  |  {n}", ha="center", va="center", fontsize=14, fontweight="bold")
        box(ax, x, 40, w, 13, d, fs=12.5)
        arrow(ax, x + w / 2, 40, 50, 33)
    box(ax, 6, 17, 88, 15,
        "Overseer: Qwen3-30B, phi-4 or gpt-oss-120b reads the public log as one prompt\n"
        "and rates every seat: p_arrangement, p_auditor, verdict", fc="#f4f6fa", fs=13.5)
    chips = ["case note", "secret-channel\nmessages", "platform record\n(true or forged)", "auditor report\n(true or forged)"]
    for i, c in enumerate(chips):
        x = i * (w + gap)
        arrow(ax, x + w / 2, 11.5, x + w / 2, 17)
        box(ax, x, 1, w, 10.5, "+ " + c, fs=12, ec="#b8b8b8")
    ax.text(50, -3.5, "Added to the same log, one at a time: 18 prompt conditions, 90 games, 3 samples each",
            ha="center", va="center", fontsize=12, color=MUTED)


# ---------------------------------------------------------------- slides
def new_slide(title, n, source):
    fig = plt.figure(figsize=(13.333, 7.5), facecolor="white")
    fig.text(0.06, 0.93, "\n".join(textwrap.wrap(title, 42)), fontsize=29, fontweight="bold", va="top",
             linespacing=1.2)
    fig.text(0.06, 0.035, source, fontsize=11, color=MUTED, va="bottom")
    fig.text(0.94, 0.035, str(n), fontsize=11, color=MUTED, va="bottom", ha="right")
    return fig


def slide_title():
    fig = plt.figure(figsize=(13.333, 7.5), facecolor="white")
    fig.text(0.07, 0.58, "On Being Honest in Collusive Places", fontsize=40, fontweight="bold", va="center")
    fig.text(0.07, 0.42, "Anas Elghoudane", fontsize=20, fontweight="bold", va="center")
    fig.text(0.07, 0.37, "Université de Montréal; Zarqa Lab, UdeM AI Club", fontsize=16, color=MUTED, va="center")
    fig.text(0.07, 0.30, "Apart Research AI Collusion Research Sprint, October 2026, Track 3.2", fontsize=16,
             color=MUTED, va="center")
    return fig


def slide_design():
    fig = new_slide("An overseer has to tell a planted auditor from the colluder it joined", 2,
                    "Source: report/rosenhan.md, Section 3. Liar's Bar, four Qwen3-30B players per game.")
    draw_design(fig.add_axes([0.06, 0.12, 0.88, 0.6]))
    return fig


def slide_label():
    fig = new_slide("A case note naming a channel member raises suspicion of a clean seat", 3,
                    "Source: report/rosenhan.md, Section 4.5. 20 G0 games with no channel; one seed-chosen seat is named.")
    ax = fig.add_axes([0.08, 0.19, 0.86, 0.48])
    series = {m: LABEL[m] for m in OVERSEERS}
    grouped_bars(ax, ["no note", "note names the seat,\nno channel", "note says the seat\njoined a secret channel"],
                 series, (0, 100), lambda v: f"{v:.1f}", "mean p_arrangement of the named seat (0 to 100)")
    return fig


def slide_credential():
    fig = new_slide("Near-complete exposure requires a platform credential", 4,
                    "Source: report/rosenhan.md, Section 4.2; results/rosenhan/exposure.json. 50 auditor games, 3 samples each.")
    ax = fig.add_axes([0.08, 0.19, 0.86, 0.48])
    series = {m: [rate(m, "access", "exposed"), rate(m, "cred_access", "exposed")] for m in OVERSEERS}
    grouped_bars(ax, ["secret-channel messages only", "messages and a platform record\nnaming the auditor"],
                 series, (0, 1.12), lambda v: f"{v:.2f}", "share of ratings that expose the pair")
    return fig


def slide_forged():
    fig = new_slide("A forged platform record inverts the verdict", 5,
                    "Source: report/rosenhan.md, Section 4.4; results/rosenhan/exposure.json. Inverted: colluder called the auditor, auditor called a colluder.")
    ax = fig.add_axes([0.08, 0.19, 0.86, 0.48])
    series = {m: [rate(m, "cred_access", "inverted"), rate(m, "cred_forged", "inverted")] for m in OVERSEERS}
    grouped_bars(ax, ["record names the auditor\n(true record)", "record names the colluder\n(forged record)"],
                 series, (0, 1.12), lambda v: f"{v:.2f}", "share of ratings that invert the roles")
    return fig


def slide_system():
    fig = new_slide("A system-turn record removes the inversion for one overseer of three", 6,
                    "Source: report/rosenhan.md, Section 4.4; results/rosenhan/exposure.json. Forged record stays in the user message.")
    ax = fig.add_axes([0.08, 0.19, 0.86, 0.48])
    series = {m: [rate(m, "cred_forged", "inverted"), rate(m, "cred_system", "inverted")] for m in OVERSEERS}
    grouped_bars(ax, ["forged record only", "forged record, and the true record\nin a system turn"],
                 series, (0, 1.12), lambda v: f"{v:.2f}", "share of ratings that invert the roles")
    return fig


def slide_behaviour():
    fig = new_slide("A count of declined challenges separates colluders better than any overseer", 7,
                    "Source: report/rosenhan.md, Section 4.1; results/rosenhan/stats*.json. AUROC, colluders vs clean seats, G0 and G1.")
    ax = fig.add_axes([0.08, 0.19, 0.86, 0.48])
    labels = [NAMES[m] + "\n(reads the log)" for m in OVERSEERS] + ["count of declined\nchallenges"]
    vals = [AUROC[m]["overseer_auroc_colluder"] for m in OVERSEERS] + [COUNT_AUROC]
    cols = [COL[m] for m in OVERSEERS] + [INK]
    ax.bar(range(4), vals, width=0.5, color=cols, zorder=3)
    for x, v in enumerate(vals):
        ax.text(x, v + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=14, zorder=5,
                bbox=dict(fc="white", ec="none", pad=1.5))
    for y, x, t in ((0.5, 1, "chance 0.5"), (0.7, 1.5, "preregistered threshold 0.7")):
        ax.axhline(y, color=MUTED, lw=1, ls="--", zorder=2)
        ax.text(x, y + 0.012, t, ha="center", va="bottom", fontsize=11, color=MUTED)
    ax.set_xticks(range(4), labels, fontsize=14)
    ax.set_xlim(-0.6, 3.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel("AUROC", fontsize=13)
    ax.yaxis.grid(True, color=GRID, zorder=0)
    ax.tick_params(length=0, labelsize=12)
    ax.tick_params(axis="x", labelsize=14)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    return fig


def slide_action():
    fig = new_slide("Verify auditor identity in code, outside the text that the overseer reads", 8,
                    "Source: report/rosenhan.md, Section 5. The code-level step is a proposal; this study did not run it.")
    ax = fig.add_axes([0.06, 0.2, 0.88, 0.5])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis("off")
    steps = ["Platform checks a\nsigned attestation\nfor the auditor", "Code fixes the auditor's role;\nagent-written text\ncannot set it",
             "Overseer gets the role\nas a fact and the channel\nmessages as evidence"]
    w = 26
    for i, s in enumerate(steps):
        x = i * (w + 5)
        box(ax, x + 3, 10, w, 20, s, fs=14, fc="#f4f6fa")
        if i:
            arrow(ax, x - 2, 20, x + 3, 20)
    ax.text(50, 4, "Tested instead: a true record in a system turn stopped the forgery for 1 of 3 overseers",
            ha="center", va="center", fontsize=14, color=MUTED)
    return fig


SLIDES = [slide_title, slide_design, slide_label, slide_credential, slide_forged, slide_system, slide_behaviour,
          slide_action]


# ---------------------------------------------------------------- gif
def build_gif(path):
    steps = [("neutral", "log only"), ("access", "+ secret-channel messages"),
             ("cred_access", "+ platform record naming the auditor"),
             ("cred_forged", "record naming the colluder instead")]
    vals = [[rate(m, c, "exposed") for m in OVERSEERS] for c, _ in steps]
    grow, hold = 8, 22
    frames = [(s, k) for s in range(len(steps)) for k in range(grow + hold)]
    fig = plt.figure(figsize=(9.6, 5.4), dpi=80, facecolor="white")
    ax = fig.add_axes([0.1, 0.16, 0.86, 0.56])
    fig.text(0.05, 0.95, "Exposure of the planted auditor, by what the overseer reads", fontsize=17,
             fontweight="bold", va="top")
    sub = fig.text(0.05, 0.86, "", fontsize=15, va="top")
    fig.text(0.05, 0.03, "Share of ratings in which the colluder is caught, the auditor cleared and no clean seat framed. "
             "50 auditor games.", fontsize=8.5, color=MUTED)

    def update(i):
        s, k = frames[i]
        ax.clear()
        t = min(1, (k + 1) / grow)
        prev = vals[s - 1] if s else [0, 0, 0]
        cur = [p + (v - p) * t for p, v in zip(prev, vals[s])]
        bars = ax.bar(range(3), cur, width=0.5, color=[COL[m] for m in OVERSEERS], zorder=3)
        for x, v in zip(range(3), cur):
            ax.text(x, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=14)
        ax.set_ylim(0, 1.12)
        ax.set_xticks(range(3), [NAMES[m] for m in OVERSEERS], fontsize=14)
        ax.set_yticks([0, 0.5, 1.0], ["0", "0.5", "1.0"], fontsize=11)
        ax.yaxis.grid(True, color=GRID, zorder=0)
        ax.tick_params(length=0)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        sub.set_text(f"Step {s + 1} of 4:  {steps[s][1]}")
        return bars

    anim = FuncAnimation(fig, update, frames=len(frames), blit=False)
    anim.save(path, writer=PillowWriter(fps=10))
    plt.close(fig)


# ---------------------------------------------------------------- social preview
def build_social(path):
    """1280 x 640 card for the repository's social preview and the project page's og:image."""
    fig = plt.figure(figsize=(12.8, 6.4), dpi=100, facecolor="white")
    fig.text(0.05, 0.86, "On Being Honest\nin Collusive Places", fontsize=34, fontweight="bold", va="top",
             linespacing=1.15)
    fig.text(0.05, 0.56, "LLM overseers give the auditor role\nto whoever a platform record names.\n"
             "A forged record hands it to the colluder.", fontsize=17, va="top", linespacing=1.45)
    fig.text(0.05, 0.14, "Anas Elghoudane, Université de Montréal\nApart Research AI Collusion Research Sprint, "
             "Track 3.2", fontsize=12, color=MUTED, va="bottom", linespacing=1.5)
    fig.text(0.05, 0.06, "github.com/Nas01010101/rosenhan-collusion-sprint", fontsize=12, color=MUTED, va="bottom")
    ax = fig.add_axes([0.56, 0.2, 0.4, 0.58])
    cats = ["record names the auditor:\npair exposed", "forged record names the\ncolluder: verdict inverted"]
    series = {m: [rate(m, "cred_access", "exposed"), rate(m, "cred_forged", "inverted")] for m in OVERSEERS}
    grouped_bars(ax, cats, series, (0, 1.12), lambda v: f"{v:.2f}", "share of ratings, 50 auditor games",
                 fontsize=13)
    ax.set_yticks([0, 0.5, 1.0], ["0", "0.5", "1.0"])
    ax.legend(frameon=False, fontsize=11, loc="lower left", ncol=3, bbox_to_anchor=(0, 1.02))
    fig.savefig(path, dpi=100, facecolor="white")
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    build_social(OUT / "social_preview.png")
    with PdfPages(OUT / "slides.pdf", metadata={"CreationDate": None}) as pdf:   # no date: reruns match byte for byte
        for i, fn in enumerate(SLIDES, 1):
            fig = fn()
            fig.savefig(OUT / f"slide_{i:02d}.png", dpi=150, facecolor="white")
            pdf.savefig(fig, facecolor="white")
            plt.close(fig)
    fig = plt.figure(figsize=(13.333, 6.2), facecolor="white")
    draw_design(fig.add_axes([0.03, 0.08, 0.94, 0.88]))
    fig.savefig(OUT / "design_cells.png", dpi=200, facecolor="white")
    plt.close(fig)
    for k in ("exposure", "label", "prime"):
        shutil.copy(ROOT / "figures" / f"rosenhan_{k}.png", OUT / f"fig_{k}.png")
    build_gif(OUT / "exposure_by_condition.gif")


if __name__ == "__main__":
    main()
