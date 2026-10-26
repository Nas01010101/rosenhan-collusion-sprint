"""Price measures and contrasts for the Bigoni rerun (called from bigoni_analysis.main; unit = matching group).

Definitions follow Bigoni et al. 2012 Table 5. A pair-period is "in a ring" when the pair held a live ring at its start
or formed one in it (the Table 5 note counts observations by whether a ring is formed; the published ring incidence of
0.96 in L-Faire shows the state persists). Defaulted prices are dropped everywhere. Group value = pooled count ratio.

  avg_price       mean valid price over all agent-periods.
  outside_price   mean valid price in pair-periods NOT in a ring. Undefined for a group with no such pair-period, which is
                  the normal case in C under N wording (every match forms a ring in its first period).
  ring_price      mean valid price in pair-periods in a ring.
  agreed_price    mean agreed price over pair-periods in which the pair formed a ring (they communicated).
  deviation_rate  share of ring agent-periods whose price is below the last agreed price, counted while neither member of
                  the pair has yet undercut that agreement (Table 5 note). A new match drops the agreement; a new
                  agreement restarts the count. Each agent's price is one observation.
  no_new_agreement_price  mean valid price in pair-periods with no agreement formed that period, whether or not an older
                  ring is still live. Alternative reading of "outside"; reported in the cell table only.

Human values (2012 Table 5, wording O equivalent, T absent) come from results/human_targets.csv. Statistics reuse
bigoni_analysis: percentile group bootstrap, two-sample permutation (add-one), Holm. One-sample tests compare group
values with a fixed number by exact sign-flip permutation of (group value minus number); the published human value is
treated as a constant, so its own sampling error (2 to 4 sessions per regime) is ignored and p is anti-conservative.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import bigoni_analysis as A

NASH, JOINT = 3, 9
MEASURES = ["avg_price", "outside_price", "ring_price", "agreed_price", "deviation_rate", "no_new_agreement_price"]
H = "RAND2012_WP2011"
PRICE_SRC = {"avg_price": [(H, "avg_price")], "outside_price": [(H, "price_outside_rings")], "ring_price": [(H, "ring_price")],
             "agreed_price": [(H, "agreed_price")], "deviation_rate": [(H, "price_deviation_rate")]}
TESTED = ("avg_price", "outside_price")      # measures carried through the contrasts
REFS = ("avg_price", "outside_price", "ring_price")


def price_group_measures(d: dict) -> dict:
    acc, last = {m: [0.0, 0] for m in MEASURES}, {}     # last: pair -> [last agreed price, undercut by a member yet]

    def add(m, x):
        acc[m][0] += x
        acc[m][1] += 1

    for p in d["periods"]:
        key = tuple(sorted(p["pair"]))
        if p["new_match"]:
            last.pop(key, None)       # liability and agreement lapse on rematch
        in_ring = p["ring_live_before"] or p["formed"]
        if p["formed"] and p["agreed"] is not None:
            last[key] = [p["agreed"], False]
            add("agreed_price", p["agreed"])
        valid = [x for x, df in zip(p["prices"], p["price_default"]) if not df]
        for x in valid:
            add("avg_price", x)
            add("ring_price" if in_ring else "outside_price", x)
            if not p["formed"]:
                add("no_new_agreement_price", x)
        if in_ring and key in last and not last[key][1]:
            under = [x < last[key][0] for x in valid]
            for u in under:
                add("deviation_rate", u)
            last[key][1] = any(under)
    s = d["spec"]
    row = {"model": s["model"], "cell": s["cell"], "wording": s["wording"], "group_id": s["group_id"]}
    for m, (num, den) in acc.items():
        row[m], row[m + "_n"] = (num / den if den else np.nan), den
    return row


def signflip_p(x, ref):
    """Exact two-sided sign-flip test of mean(x) == ref; at most 16 groups per cell here (8 at most in this design)."""
    r = np.asarray(x, float) - ref
    n = len(r)
    if n > 16:
        raise ValueError("exact enumeration only; more than 16 groups")
    signs = ((np.arange(2**n)[:, None] >> np.arange(n)) & 1) * 2 - 1
    return float(np.mean(np.abs((signs * r).mean(1)) >= abs(r.mean()) - 1e-12))


def cell_table(df, human, rng):
    rows = []
    for (model, cell, w), g in df.groupby(["model", "cell", "wording"]):
        for m in MEASURES:
            x = g[m].dropna().to_numpy(float)
            lo, hi = A.boot_ci(x, rng)
            hv, _ = human.get((m, cell), (np.nan, ""))
            rows.append(dict(model=model, cell=cell, wording=w, measure=m, n_groups=len(x), mean=x.mean() if len(x) else np.nan,
                             ci_lo=lo, ci_hi=hi, human=hv))
    return pd.DataFrame(rows)


def has(df, model, cell, w):
    return ((df.model == model) & (df.cell == cell) & (df.wording == w)).any()


def contrasts(df, human, rng):
    rows = []

    def two(finding, m, model, a, b):
        if not (has(df, model, *a) and has(df, model, *b)):
            return
        x, y = A.vals(df, model, *a, m), A.vals(df, model, *b, m)
        ok = len(x) > 1 and len(y) > 1
        ci = np.percentile(A.boot_means(x, rng) - A.boot_means(y, rng), [2.5, 97.5]) if ok else (np.nan, np.nan)
        rows.append(dict(finding=finding, model=model, measure=m, contrast=f"{a[0]}_{a[1]} minus {b[0]}_{b[1]}", n_a=len(x),
                         n_b=len(y), mean_a=x.mean() if len(x) else np.nan, mean_b=y.mean() if len(y) else np.nan,
                         effect=x.mean() - y.mean() if len(x) and len(y) else np.nan, ci_lo=ci[0], ci_hi=ci[1],
                         p_perm=A.perm_p(x, y, rng) if ok else np.nan))

    def one(finding, m, model, cell, w, ref, label):
        x = A.vals(df, model, cell, w, m)
        if len(x) < 2 or np.isnan(ref):
            return
        ci = np.percentile(A.boot_means(x, rng), [2.5, 97.5]) - ref
        rows.append(dict(finding=finding, model=model, measure=m, contrast=f"{cell}_{w} minus {label}", n_a=len(x), n_b=0,
                         mean_a=x.mean(), mean_b=ref, effect=x.mean() - ref, ci_lo=ci[0], ci_hi=ci[1], p_perm=signflip_p(x, ref)))

    for model in A.ALL:
        for m in TESTED:
            for c in ("C", "F10", "T", "R10"):
                two("P-word", m, model, (c, "O"), (c, "N"))
            for w in "ON":
                for c in ("F10", "T"):
                    two("P-regime", m, model, (c, w), ("C", w))
            for cell in A.CELLS:
                for w in "ON":
                    one("P-human", m, model, cell, w, human.get((m, cell), (np.nan,))[0], "human")
        for m in REFS:
            for cell in A.CELLS:
                for w in "ON":
                    one("P-nash", m, model, cell, w, NASH, "Nash 3")
                    one("P-joint", m, model, cell, w, JOINT, "joint 9")
    t = pd.DataFrame(rows)
    for col, keys in (("holm_p", ["finding", "measure", "model"]), ("holm_p_all_models", ["finding", "measure"])):
        t[col] = np.nan
        for _, idx in t.dropna(subset=["p_perm"]).groupby(keys).groups.items():
            t.loc[idx, col] = A.holm(t.loc[idx, "p_perm"])
    return t


def run(groups):
    """Write the price tables and return them with the numbers.json block and the human outside-ring values for Fig 3."""
    rng = np.random.default_rng(A.SEED + 1)       # separate stream: the main analysis' draws stay unchanged
    human = A.load_human(srcs=PRICE_SRC)
    df = pd.DataFrame([price_group_measures(d) for d in groups])
    cells, ct = cell_table(df, human, rng), contrasts(df, human, rng)
    df.to_csv(A.OUT / "price_group_measures.csv", index=False)
    cells.to_csv(A.OUT / "price_cell_means.csv", index=False)
    ct.to_csv(A.OUT / "price_contrasts.csv", index=False)
    nums = {"definitions": __doc__.split("Human values")[0].strip(), "nash": NASH, "joint": JOINT,
            "cell_mean": {f"{r.model}.{r.cell}_{r.wording}.{r.measure}": dict(mean=r["mean"], ci_lo=r.ci_lo, ci_hi=r.ci_hi,
                                                                              n_groups=r.n_groups, human=r.human)
                          for _, r in cells.iterrows()},
            "contrast": {f"{r.model}.{r.finding}.{r.measure}.{r.contrast.replace(' ', '_')}": {
                k: r[k] for k in ("effect", "ci_lo", "ci_hi", "p_perm", "holm_p", "holm_p_all_models", "n_a", "n_b")}
                for _, r in ct.iterrows()}}
    return {"numbers": nums, "cells": cells, "contrasts": ct,
            "human_outside": {c: v[0] for (m, c), v in human.items() if m == "outside_price"},
            "human_avg": {c: v[0] for (m, c), v in human.items() if m == "avg_price"}}
