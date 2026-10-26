"""Figures for the Bigoni rerun: one y axis per panel, fixed source order and colours (human, Qwen3-8B, Olmo-3-7B,
gpt-oss-20b), 200 dpi PNG plus PDF. Input is the cell_means table written by bigoni_analysis.py.

Colours are Okabe-Ito; human is black. Markers differ per source so identity never rests on colour alone.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

matplotlib.rcParams.update({"pdf.fonttype": 42, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                            "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#e3e3e3", "grid.linewidth": 0.6,
                            "axes.axisbelow": True})
SOURCES = [("human", "Human (published)", "#000000", "D"), ("qwen3-8b", "Qwen3-8B", "#0072B2", "o"),
           ("olmo3-7b", "Olmo-3-7B", "#D98E00", "s"), ("gpt-oss-20b", "gpt-oss-20b", "#009E73", "^")]
CELL_ORDER = ["C", "F10", "F02", "L10", "L02", "R10", "T"]


def _save(fig, stem: Path):
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _row(cm, model, cell, wording, measure):
    r = cm[(cm.model == model) & (cm.cell == cell) & (cm.wording == wording) & (cm.measure == measure)]
    return r.iloc[0] if len(r) and r.iloc[0].n_groups > 0 else None


def fig1(cm, human, out: Path):
    """Ring formation and report rate per cell, wording O: human published value against LLM group means and CIs."""
    panels = [("formation", "Ring formation", "Share of pair-periods without a ring in which both say yes", CELL_ORDER),
              ("report_rate", "Report rate", "Share of live-ring pair-periods with a report", CELL_ORDER[1:])]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    offs = [-0.27, -0.09, 0.09, 0.27]
    for ax, (m, title, ylab, cells) in zip(axes, panels):
        for k, (src, label, col, mk) in enumerate(SOURCES):
            for i, c in enumerate(cells):
                if src == "human":
                    if (m, c) in human:
                        ax.plot(i + offs[k], human[(m, c)][0], mk, color=col, ms=6, label=label if i == 0 else None)
                    continue
                r = _row(cm, src, c, "O", m)
                if r is not None:
                    ax.errorbar(i + offs[k], r["mean"], yerr=[[r["mean"] - r.ci_lo], [r.ci_hi - r["mean"]]], fmt=mk,
                                color=col, ms=5.5, lw=1.2, capsize=0)
        ax.set_xticks(range(len(cells)), cells)
        ax.set_xlim(-0.6, len(cells) - 0.4)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_ylabel(ylab, fontsize=8)
        ax.set_xlabel("Cell (wording O)")
    handles = [plt.Line2D([], [], marker=mk, color=col, ls="", ms=6, label=label) for _, label, col, mk in SOURCES]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    _save(fig, out / "bigoni_fig1_formation_report")


def fig2(cm, out: Path):
    """Payoff x wording grid: formation in C, F10, T under wording O (filled) and N (open), one panel per model."""
    cells = ["C", "F10", "T"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), sharey=True)
    for ax, (model, label, col, mk) in zip(axes, SOURCES[1:]):
        for i, c in enumerate(cells):
            pts = {}
            for w, dx in (("O", -0.12), ("N", 0.12)):
                r = _row(cm, model, c, w, "formation")
                if r is None:
                    continue
                pts[w] = (i + dx, r["mean"])
                ax.errorbar(i + dx, r["mean"], yerr=[[r["mean"] - r.ci_lo], [r.ci_hi - r["mean"]]], fmt=mk, color=col,
                            mfc=col if w == "O" else "white", ms=6.5, lw=1.2, capsize=0, zorder=3)
            if len(pts) == 2:
                ax.plot([pts["O"][0], pts["N"][0]], [pts["O"][1], pts["N"][1]], color=col, lw=1, alpha=0.6, zorder=2)
        ax.set_xticks(range(len(cells)), cells)
        ax.set_xlim(-0.5, len(cells) - 0.5)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(label, loc="left", fontsize=10)
        ax.set_xlabel("Cell")
    axes[0].set_ylabel("Ring formation (share of pair-periods without a ring)", fontsize=8)
    handles = [plt.Line2D([], [], marker="o", color="#555555", ls="", ms=6, mfc="#555555", label="Original wording (O)"),
               plt.Line2D([], [], marker="o", color="#555555", ls="", ms=6, mfc="white", label="Neutral wording (N)")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout()
    _save(fig, out / "bigoni_fig2_payoff_wording")


def fig3(form, price, human_price, out: Path):
    """Talk versus price: ring formation (top) and average price (bottom) in C, F10, T, one column per model.
    O filled, N open. Dashed lines mark the stage Nash price (3) and the joint-profit price (9); the black diamond is the
    published 2012 average price (Table 5), the same quantity as the plotted LLM average price."""
    cells = ["C", "F10", "T"]
    cm = pd.concat([form, price[price.measure == "avg_price"]])
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.2), sharey="row")
    for j, (model, label, col, mk) in enumerate(SOURCES[1:]):
        for row, (m, ylab, ylim) in enumerate([("formation", "(a) Ring formation (share of pair-periods without a ring)", (-0.02, 1.02)),
                                               ("avg_price", "(b) Average price", (0, 12))]):
            ax = axes[row, j]
            for i, c in enumerate(cells):
                pts = {}
                for w, dx in (("O", -0.12), ("N", 0.12)):
                    r = _row(cm, model, c, w, m)
                    if r is None:
                        continue
                    pts[w] = (i + dx, r["mean"])
                    ax.errorbar(i + dx, r["mean"], yerr=[[r["mean"] - r.ci_lo], [r.ci_hi - r["mean"]]], fmt=mk, color=col,
                                mfc=col if w == "O" else "white", ms=6.5, lw=1.2, capsize=0, zorder=3)
                if len(pts) == 2:
                    ax.plot([pts["O"][0], pts["N"][0]], [pts["O"][1], pts["N"][1]], color=col, lw=1, alpha=0.6, zorder=2)
                if m == "avg_price" and c in human_price:
                    ax.plot(i, human_price[c], "D", color="#000000", ms=6, zorder=4)
            if m == "avg_price":
                for y, t in ((3, "Nash 3"), (9, "joint 9")):
                    ax.axhline(y, color="#777777", ls="--", lw=0.9, zorder=1)
                    if j == 2:   # labels sit in the right margin so they never cover a marker or an interval
                        ax.text(1.02, y, t, transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=7, color="#555555")
            ax.set_xticks(range(len(cells)), cells)
            ax.set_xlim(-0.5, len(cells) - 0.5)
            ax.set_ylim(*ylim)
            if row == 0:
                ax.set_title(label, loc="left", fontsize=10)
            if j == 0:
                ax.set_ylabel(ylab, fontsize=8)
            if row == 1:
                ax.set_xlabel("Cell")
    handles = [plt.Line2D([], [], marker="o", color="#555555", ls="", ms=6, mfc="#555555", label="Original wording (O)"),
               plt.Line2D([], [], marker="o", color="#555555", ls="", ms=6, mfc="white", label="Neutral wording (N)"),
               plt.Line2D([], [], marker="D", color="#000000", ls="", ms=6, label="Human, average price (2012)")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout()
    _save(fig, out / "bigoni_fig3_talk_vs_price")
