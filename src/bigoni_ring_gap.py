"""Ring price minus outside-ring price, per group, against the human gap in Bigoni et al. 2012 Table 5.

The human gap (ring 4.97 vs outside 3.50 in laissez-faire) is what talking adds to price. A gap near 0 for LLM
sellers means prices are set the same with or without an agreement. Unit = group with both prices defined;
p from a sign-flip permutation of the group gaps around 0 and around the human gap (constant, so anti-conservative).
Writes results/analysis/ring_gap.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HUMAN = {"C": (4.971, 3.5), "F10": (6.144, 4.233), "L10": (7.024, 4.063), "R10": (5.339, 3.567)}  # Table 5


def signflip_p(x, mu, n=10_000, seed=0):
    d = np.asarray(x) - mu
    flips = np.random.default_rng(seed).choice([-1, 1], size=(n, len(d)))
    return float((np.abs((flips * d).mean(1)) >= abs(d.mean()) - 1e-12).mean())


def main():
    g = pd.read_csv(ROOT / "results/analysis/price_group_measures.csv")
    g = g.dropna(subset=["ring_price", "outside_price"]).assign(gap=lambda d: d.ring_price - d.outside_price)
    rows = []
    for (model, wording), d in g.groupby(["model", "wording"]):
        for cell, dc in list(d.groupby("cell")) + [("all", d)]:
            h = HUMAN.get(cell)
            hg = h[0] - h[1] if h else None
            rows.append(dict(model=model, wording=wording, cell=cell, n_groups=len(dc), gap=dc.gap.mean(),
                             p_vs_0=signflip_p(dc.gap, 0.0), human_gap=hg,
                             p_vs_human=signflip_p(dc.gap, hg) if hg is not None else None))
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results/analysis/ring_gap.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
