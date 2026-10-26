"""Wilcoxon signed-rank test with an exact p-value at every sample size, ties included.

Zeros are dropped (Wilcoxon's rule), |d| gets midranks, and the p-value comes from the exact sign-flip distribution
of T+ given those ranks: all 2^n sign assignments, counted by dynamic programming over doubled (integer) ranks.
With no ties and no zeros this is the distribution scipy.stats.wilcoxon(method="exact") uses; with ties it stays
exact, where scipy's "auto" switches to the normal approximation. One method for every paired test in the study.
"""

import numpy as np
from scipy.stats import rankdata


def signrank(x, y, alternative="two-sided"):
    """dict(p, n, t_plus, r_rb) for paired x - y; r_rb is the matched-pairs rank-biserial correlation."""
    # Game-level rates are means of 3 samples; round so equal differences reached by different sums tie exactly.
    d = np.round(np.asarray(x, float) - np.asarray(y, float), 9)
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return dict(p=1.0, n=0, t_plus=0.0, r_rb=0.0)
    r2 = np.rint(2 * rankdata(np.abs(d))).astype(int)   # midranks are multiples of 1/2
    t2 = int(r2[d > 0].sum())
    dist = np.zeros(r2.sum() + 1)
    dist[0] = 1.0
    for r in r2:   # counts stay exact integers in float64 while 2^n < 2^53
        dist[r:] = dist[r:] + dist[:-r].copy()
    pmf = dist / dist.sum()
    p_ge, p_le = pmf[t2:].sum(), pmf[:t2 + 1].sum()
    p = {"greater": p_ge, "less": p_le, "two-sided": min(1.0, 2 * min(p_ge, p_le))}[alternative]
    total = r2.sum() / 2
    return dict(p=float(p), n=n, t_plus=t2 / 2, r_rb=float((2 * (t2 / 2) - total) / total))


def holm(ps):
    """Holm step-down adjusted p-values, same order as the input."""
    order = np.argsort(ps)
    adj, run = np.empty(len(ps)), 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (len(ps) - i) * ps[k]))
        adj[k] = run
    return adj.tolist()
