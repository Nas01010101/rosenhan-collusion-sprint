"""Pre-registered analysis of the LLM rerun of the Bigoni et al. antitrust game (notes/prereg_replication.md).

    .venv/bin/python src/bigoni_analysis.py            # writes results/analysis/ and results/figures/
                                                       # (price measures: src/bigoni_prices.py, called from main)

Unit = matching group (8 agents, 4 pairs). Inputs: results/raw/bigoni/main/{qwen3-8b,olmo3-7b,gpt-oss-20b}/groups/*.json
(runs/ is read only to find which Modal run produced each group), results/human_targets.csv.

Measure definitions. A pair-period is "in a ring" when the pair held a live ring at the start of the period
(ring_live_before): a ring persists until report, detection or rematch, and in C (no reporting) until rematch.
Each is a count ratio per group (numerator / denominator pooled over the group's eligible observations); the group
value is that ratio and every statistic treats groups as the observation.

  comm_decision  Per-agent decision, not "at least one agent". Denominator: agent-periods of pair-periods not in a ring.
                 Numerator: those agents who said yes to talking. Matches the 2015 Figure 1 definition (individual
                 decision to communicate while no ring exists) and the 2012 Table 4 "rate of comm. att." (binary
                 individual decisions in all periods a ring was not already formed). Both papers use this
                 denominator; only the session pools differ (2012: Stockholm and Rome, 2015: Rome).
  formation      2012 Table 4 "rate of ring formation": one observation per pair-period not in a ring, 1 if both
                 said yes. Pair-periods already in a ring are excluded, so a ring is counted once.
  report_rate    2012 Table 4 "rate of reporting": one observation per pair-period with a live ring at the reporting
                 stage (formed this period or earlier), 1 if one or both reported. Undefined in C.
  comm_first     2012 Table 4 "comm. att. (1st period)": per-agent decisions in the first period of a match.
  ring_price     mean valid price of the agents in pair-periods with a ring (live at the start or formed now).
  avg_price      mean valid price over all agent-periods.
  formation_any, attempt_any  the bigoni_summary.py definitions (no ring-state condition; attempt = at least one
                 agent said yes). Kept only to show how much the headline moves; not used in any test.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bigoni_prompts import classify_label  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results/raw/bigoni/main"
OUT = ROOT / "results/analysis"
FIG = ROOT / "results/figures"
MODELS = [("qwen3-8b", "Qwen3-8B"), ("olmo3-7b", "Olmo-3-7B"), ("gpt-oss-20b", "gpt-oss-20b")]
NAME = dict(MODELS)
DIR = {"qwen3-8b": "qwen3-8b-v2"}   # Qwen3-8B single-run set; the earlier assembled set is qwen3-8b-assembled
CELLS = ["C", "F10", "F02", "L10", "L02", "R10", "T"]
B, SEED, QUIZ_MIN = 10_000, 20261024, 4
BANNED = "car" + "tel"
# (regime label in human_targets.csv, alpha, F) per cell
HUMAN_KEY = {"C": ("C", 0.0, 0), "F10": ("Fine", 0.1, 200), "F02": ("Fine", 0.02, 1000), "L10": ("Leniency", 0.1, 200),
             "L02": ("Leniency", 0.02, 1000), "R10": ("Reward", 0.1, 200), "T": ("Fine", 0.0, 1000)}
# measure -> ordered (source, measure name in human_targets.csv after renaming); first hit per cell wins
HUMAN_SRC = {"comm_decision": [("IFN859_2010", "comm_decision_rate_not_in_ring"), ("RAND2012_WP2011", "comm_attempt_rate")],
             "formation": [("RAND2012_WP2011", "ring_formation_rate")], "report_rate": [("RAND2012_WP2011", "report_rate")],
             "comm_first": [("RAND2012_WP2011", "comm_attempt_rate_first_period_in_match")],
             "ring_price": [("RAND2012_WP2011", "ring_price")], "avg_price": [("RAND2012_WP2011", "avg_price")]}
MEASURES = list(HUMAN_SRC) + ["formation_any", "attempt_any"]


# ----------------------------------------------------------------------------- loading and measures


def load_groups(raw: Path = RAW) -> list[dict]:
    return [json.load(open(f)) for m, _ in MODELS for f in sorted((raw / DIR.get(m, m) / "groups").glob("*.json"))]


def run_labels(raw: Path = RAW) -> dict[str, str]:
    """group_id -> Modal run whose file is byte-identical to the analysed file (group ids do not encode the run)."""
    digest = lambda f: hashlib.md5(f.read_bytes()).hexdigest()  # noqa: E731
    runs = {r.name: {f.name: digest(f) for f in (r / "groups").glob("*.json")} for r in sorted((raw / "runs").iterdir())}
    out = {}
    for m, _ in MODELS:
        for f in (raw / DIR.get(m, m) / "groups").glob("*.json"):
            hits = [r for r, files in runs.items() if files.get(f.name) == digest(f)]
            out[f.stem] = hits[-1] if hits else "unknown"
    return out


def group_measures(d: dict, min_quiz: int = 0) -> dict:
    """Per-group measure values and denominators. min_quiz > 0 drops pair-periods with an agent scoring below it."""
    bad = {q["agent"] for q in d["quiz"] if q["score"] < min_quiz}
    acc = defaultdict(lambda: [0.0, 0])

    def add(m, num, den=1):
        acc[m][0] += num
        acc[m][1] += den

    for p in d["periods"]:
        if bad & set(p["pair"]):
            continue
        in_ring = p["ring_live_before"]
        if not in_ring:
            add("formation", p["formed"])
            for c in p["comm"]:
                add("comm_decision", c)
        if p["new_match"]:
            for c in p["comm"]:
                add("comm_first", c)
        if p["report4"] is not None:
            add("report_rate", p["reported"])
        valid = [x for x, df in zip(p["prices"], p["price_default"]) if not df]
        for x in valid:
            add("avg_price", x)
            if in_ring or p["formed"]:
                add("ring_price", x)
        add("formation_any", p["formed"])
        add("attempt_any", any(p["comm"]))
    s = d["spec"]
    row = {"model": s["model"], "cell": s["cell"], "wording": s["wording"], "paraphrase": s["paraphrase"],
           "group_id": s["group_id"], "n_periods": max(p["period"] for p in d["periods"]), "complete": d["complete"],
           "calls_per_pair_period": sum(x["kind"] not in ("quiz", "recog") for x in d["decisions"]) / len(d["periods"])}
    for m in MEASURES:
        num, den = acc[m]
        row[m], row[m + "_n"] = (num / den if den else np.nan), den
    return row


def measure_table(groups: list[dict], runs: dict[str, str], min_quiz: int = 0) -> pd.DataFrame:
    df = pd.DataFrame([group_measures(d, min_quiz) for d in groups])
    df["run"] = df.group_id.map(runs)
    return df


def load_human(path: Path = ROOT / "results/human_targets.csv", srcs: dict | None = None) -> dict:
    h = pd.read_csv(path)
    h["measure"] = h.measure.str.replace(BANNED, "ring", regex=False)
    h["regime"] = h.regime.replace({"L-Faire": "C", "Communication": "C"})
    out = {}
    for m, ms in (srcs or HUMAN_SRC).items():
        for cell, (reg, a, f) in HUMAN_KEY.items():
            for src, name in ms:
                hit = h[(h.source == src) & (h.measure == name) & (h.regime == reg) & np.isclose(h.alpha, a) & (h.F == f)]
                if len(hit) and pd.notna(hit.value.iloc[0]):
                    out[(m, cell)] = (float(hit.value.iloc[0]), src)
                    break
    return out


# ----------------------------------------------------------------------------- statistics


def boot_means(x, rng, n=B):
    x = np.asarray(x, float)
    return x[rng.integers(0, len(x), (n, len(x)))].mean(1)


def boot_ci(x, rng, n=B):
    return tuple(np.percentile(boot_means(x, rng, n), [2.5, 97.5])) if len(x) > 1 else (np.nan, np.nan)


def perm_p(a, b, rng, n=B):
    """Two-sided label permutation test on the difference of group means (add-one p-value)."""
    z, k = np.concatenate([a, b]), len(a)
    obs = abs(np.mean(a) - np.mean(b))
    order = np.argsort(rng.random((n, len(z))), axis=1)
    d = np.abs(z[order[:, :k]].mean(1) - z[order[:, k:]].mean(1))
    return float((1 + np.sum(d >= obs - 1e-12)) / (n + 1))


def holm(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    adj, run = np.empty(len(ps)), 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(ps) - rank) * ps[i])
        adj[i] = min(1.0, run)
    return adj


def vals(df, model, cell, wording, measure):
    s = df[(df.model == model) & (df.cell == cell) & (df.wording == wording)][measure].dropna()
    return s.to_numpy(float)


def cell_means(df, human, rng, sample):
    rows = []
    for model, _ in MODELS:
        for (cell, w), g in df[df.model == model].groupby(["cell", "wording"]):
            for m in HUMAN_SRC:
                x = g[m].dropna().to_numpy(float)
                lo, hi = boot_ci(x, rng)
                hv, hs = human.get((m, cell), (np.nan, ""))
                rows.append(dict(sample=sample, model=model, cell=cell, wording=w, measure=m, n_groups=len(x),
                                 mean=x.mean() if len(x) else np.nan, ci_lo=lo, ci_hi=hi, human=hv, human_source=hs))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- pre-registered contrasts

ALL, QWEN = [m for m, _ in MODELS], ["qwen3-8b"]


def contrast_specs():
    """(finding, measure, (cell, wording) a, (cell, wording) b, models); effect = a minus b. First measure is primary."""
    s = [("H-inc", "report_rate", ("R10", "O"), ("F10", "O"), ALL)]
    for m in ("comm_decision", "formation"):
        s.append(("H-threat", m, ("T", "O"), ("C", "O"), ALL))
    for m in ("formation", "comm_decision"):
        s += [("H-word", m, (c, "O"), (c, "N"), ALL) for c in ("C", "F10", "T", "R10")]
        s.append(("H-incentive-N", m, ("F10", "N"), ("C", "N"), ALL))
    for m in ("comm_decision", "formation"):
        s += [("H-certainty", m, ("F02", "O"), ("F10", "O"), QWEN), ("H-certainty", m, ("L02", "O"), ("L10", "O"), QWEN)]
    s += [("H-ranking", "formation", a, b, QWEN) for a, b in
          [(("C", "O"), ("F10", "O")), (("F10", "O"), ("L10", "O")), (("F10", "O"), ("R10", "O"))]]
    return s


def verdict(llm, human):
    """Prereg criteria: same sign and LLM effect within 0.5x to 2x of the human effect."""
    if np.isnan(llm) or np.isnan(human):
        return "insufficient data" if np.isnan(llm) else "no human anchor"
    if np.sign(llm) != np.sign(human) or llm == 0:
        return "not replicated"
    return "replicated" if 0.5 <= llm / human <= 2 else "direction only"


def run_contrasts(df, human, rng, sample):
    rows, primary = [], {}
    for finding, m, (ca, wa), (cb, wb), models in contrast_specs():
        primary.setdefault(finding, m)
        for model in models:
            a, b = vals(df, model, ca, wa, m), vals(df, model, cb, wb, m)
            if not all(((df.model == model) & (df.cell == c) & (df.wording == w)).any() for c, w in ((ca, wa), (cb, wb))):
                continue               # cell not in this model's design (e.g. R10 under N for Olmo and gpt-oss)
            h = [human.get((m, c)) for c in (ca, cb)]
            if wa != wb or not all(h) or h[0][1] != h[1][1]:
                h = [None, None]       # no human wording arm (humans saw the original text); never mix sources
            herr = (h[0][0] - h[1][0]) if all(h) else np.nan
            ok = len(a) > 1 and len(b) > 1
            eff = a.mean() - b.mean() if len(a) and len(b) else np.nan
            ci = np.percentile(boot_means(a, rng) - boot_means(b, rng), [2.5, 97.5]) if ok else (np.nan, np.nan)
            rows.append(dict(sample=sample, finding=finding, model=model, measure=m, primary=m == primary[finding],
                             contrast=f"{ca}_{wa} minus {cb}_{wb}", n_a=len(a), n_b=len(b),
                             mean_a=a.mean() if len(a) else np.nan, mean_b=b.mean() if len(b) else np.nan, effect=eff,
                             ci_lo=ci[0], ci_hi=ci[1], p_perm=perm_p(a, b, rng) if ok else np.nan, human_effect=herr,
                             human_source=h[0][1] if all(h) else "", ratio=eff / herr if herr else np.nan,
                             verdict=verdict(eff, herr) if ok else "insufficient data"))
    t = pd.DataFrame(rows)
    # Holm within each finding: family = same finding, measure and model; the _all_models column pools the models
    for col, keys in (("holm_p", ["finding", "measure", "model"]), ("holm_p_all_models", ["finding", "measure"])):
        t[col] = np.nan
        for _, idx in t.dropna(subset=["p_perm"]).groupby(keys).groups.items():
            t.loc[idx, col] = holm(t.loc[idx, "p_perm"])
    return t


def replication_table(c: pd.DataFrame) -> pd.DataFrame:
    """Primary-measure contrasts with a human anchor, plus one joint row per model for the formation ranking."""
    r = c[c.primary & c.human_effect.notna()].copy()
    rk = r[r.finding == "H-ranking"]
    joint = []
    for (sample, model), g in rk.groupby(["sample", "model"]):
        v = set(g.verdict)
        joint.append(dict(sample=sample, finding="H-ranking", model=model, measure="formation", primary=True,
                          contrast="joint: C > F10 > L10, R10", n_a=g.n_a.min(), n_b=g.n_b.min(),
                          verdict="insufficient data" if "insufficient data" in v else "replicated" if v == {"replicated"} else
                          "not replicated" if "not replicated" in v else "direction only"))
    return pd.concat([r, pd.DataFrame(joint)], ignore_index=True)


def md_table(t: pd.DataFrame) -> str:
    f = lambda x, k=3: "" if pd.isna(x) else f"{x:.{k}f}"  # noqa: E731
    head = "| finding | model | contrast | measure | human | LLM effect [95% CI] | ratio | p | Holm p | Holm p (all models) | verdict |"
    lines = [head, "|" + "---|" * 11]
    for _, r in t.iterrows():
        ci = f"{f(r.effect)} [{f(r.ci_lo)}, {f(r.ci_hi)}]" if pd.notna(r.effect) else ""
        lines.append(f"| {r.finding} | {NAME[r.model]} | {r.contrast} | {r.measure} | {f(r.human_effect)} | {ci} | "
                     f"{f(r.ratio, 2)} | {f(r.p_perm, 4)} | {f(r.holm_p, 4)} | {f(r.holm_p_all_models, 4)} | {r.verdict} |")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- checks


def run_check(df, rng, n=B):
    """Do Qwen outcomes differ by Modal run? Run labels are permuted within each (cell, wording) stratum, because the
    runs cover different cells; the statistic is the between-run sum of squares of cell-centred group values."""
    q, out = df[df.model == "qwen3-8b"], []
    for m in ("formation", "comm_decision", "report_rate", "avg_price", "n_periods", "calls_per_pair_period"):
        s = q.dropna(subset=[m]).reset_index(drop=True)
        res = (s[m] - s.groupby(["cell", "wording"])[m].transform("mean")).to_numpy(float)
        runs = sorted(s.run.unique())
        lab = s.run.map({r: i for i, r in enumerate(runs)}).to_numpy()
        n_r = np.bincount(lab, minlength=len(runs))
        stat = lambda L: sum((res * (L == i)).sum(-1) ** 2 / n_r[i] for i in range(len(runs)))  # noqa: E731
        perm = np.tile(lab, (n, 1))
        for idx in s.groupby(["cell", "wording"]).indices.values():
            perm[:, idx] = lab[idx][np.argsort(rng.random((n, len(idx))), axis=1)]
        p = float((1 + np.sum(stat(perm) >= stat(lab) - 1e-12)) / (n + 1))
        for i, r in enumerate(runs):
            out.append(dict(measure=m, run=r, n_groups=int(n_r[i]), mean_cell_centred=res[lab == i].mean(),
                            mean_raw=s[m][lab == i].mean(), p_stratified_perm=p))
    return pd.DataFrame(out)


def quiz_check(groups):
    rows = []
    for model, _ in MODELS:
        for w in "ON":
            g = [d for d in groups if d["spec"]["model"] == model and d["spec"]["wording"] == w]
            sc = np.array([q["score"] for d in g for q in d["quiz"]], float)
            rows.append(dict(model=model, wording=w, agents=len(sc), mean_score=sc.mean(), share_below=(sc < QUIZ_MIN).mean(),
                             groups_with_any_below=sum(any(q["score"] < QUIZ_MIN for q in d["quiz"]) for d in g),
                             groups=len(g)))
    return pd.DataFrame(rows)


def recognisability(groups, rng):
    rows = []
    for model, _ in MODELS:
        shares = {}
        for w in "ON":
            g = [d for d in groups if d["spec"]["model"] == model and d["spec"]["wording"] == w]
            cls = [[classify_label(r["label"]) for r in d["recog"]] for d in g]
            shares[w] = {k: np.array([c.count(k) / len(c) for c in cls]) for k in ("coordination", "competition", "game", "other")}
            x = shares[w]["coordination"]
            lo, hi = boot_ci(x, rng)
            rows.append(dict(model=model, wording=w, groups=len(g), agents=sum(map(len, cls)), coordination_share=x.mean(),
                             ci_lo=lo, ci_hi=hi, **{f"{k}_share": v.mean() for k, v in shares[w].items() if k != "coordination"}))
        p = perm_p(shares["O"]["coordination"], shares["N"]["coordination"], rng)
        for r in rows[-2:]:
            r["p_O_vs_N"] = p
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- output


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else float(o)
    return int(o) if isinstance(o, np.integer) else (bool(o) if isinstance(o, np.bool_) else o)


def numbers(cm, ct, rep, rc, qc, rec, df, df_summary):
    n = {"meta": {"bootstrap_resamples": B, "permutations": B, "seed": SEED, "quiz_min_correct": QUIZ_MIN}}
    n["cell_mean"] = {f"{r['sample']}.{r.model}.{r.cell}_{r.wording}.{r.measure}": dict(
        mean=r["mean"], ci_lo=r.ci_lo, ci_hi=r.ci_hi, n_groups=r.n_groups, human=r.human) for _, r in cm.iterrows()}
    n["contrast"] = {f"{r['sample']}.{r.model}.{r.finding}.{r.measure}.{r.contrast.replace(' ', '_')}": {
        k: r[k] for k in ("effect", "ci_lo", "ci_hi", "p_perm", "holm_p", "holm_p_all_models", "human_effect", "ratio",
                          "verdict", "n_a", "n_b")} for _, r in ct.iterrows()}
    n["replication"] = {f"{r['sample']}.{r.model}.{r.finding}.{r.contrast.replace(' ', '_')}": r.verdict for _, r in rep.iterrows()}
    n["run_check"] = {f"{r.measure}.{r.run}": dict(n_groups=r.n_groups, mean_cell_centred=r.mean_cell_centred,
                                                   p_stratified_perm=r.p_stratified_perm) for _, r in rc.iterrows()}
    n["run_composition"] = {k: int(v) for k, v in df[df.model == "qwen3-8b"].run.value_counts().items()}
    n["quiz"] = {f"{r.model}.{r.wording}": dict(mean_score=r.mean_score, share_below=r.share_below, agents=r.agents,
                                               groups_with_any_below=r.groups_with_any_below) for _, r in qc.iterrows()}
    n["recognisability"] = {f"{r.model}.{r.wording}": {k: r[k] for k in rec.columns if k not in ("model", "wording")}
                            for _, r in rec.iterrows()}
    # headline sensitivity: recomputed definition vs the bigoni_summary.py definition, group-mean per model and cell
    n["definition_shift"] = {f"{m}.{c}_{w}": dict(formation=g.formation.mean(), formation_any=g.formation_any.mean(),
                                                  comm_decision=g.comm_decision.mean(), attempt_any=g.attempt_any.mean())
                             for (m, c, w), g in df.groupby(["model", "cell", "wording"])}
    n["summary_csv_agreement_max_abs_diff"] = dict(
        formation_any=float((df.set_index("group_id").formation_any - df_summary.set_index("group_id").formation_rate).abs().max()),
        attempt_any=float((df.set_index("group_id").attempt_any - df_summary.set_index("group_id").attempt_rate).abs().max()))
    return clean(n)


def main():
    rng = np.random.default_rng(SEED)
    groups, runs, human = load_groups(), run_labels(), load_human()
    summary = pd.read_csv(ROOT / "results/bigoni_main_summary.csv")
    OUT.mkdir(parents=True, exist_ok=True)
    cms, cts = [], []
    frames = {"all": measure_table(groups, runs), f"quiz{QUIZ_MIN}": measure_table(groups, runs, QUIZ_MIN)}
    for sample, df in frames.items():
        cms.append(cell_means(df, human, rng, sample))
        cts.append(run_contrasts(df, human, rng, sample))
    cm, ct = pd.concat(cms), pd.concat(cts)
    rep = replication_table(ct)
    rc, qc, rec = run_check(frames["all"], rng), quiz_check(groups), recognisability(groups, rng)
    frames["all"].to_csv(OUT / "group_measures.csv", index=False)
    cm.to_csv(OUT / "cell_means.csv", index=False)
    ct.to_csv(OUT / "contrasts.csv", index=False)
    rep.to_csv(OUT / "replication_table.csv", index=False)
    rc.to_csv(OUT / "run_check.csv", index=False)
    qc.to_csv(OUT / "quiz_check.csv", index=False)
    rec.to_csv(OUT / "recognisability.csv", index=False)
    md = ["# Replication verdicts", "", "Effect = first-named cell minus second-named cell, group means. Human effects are the "
          "published cell differences. Verdict rule (prereg): same sign and LLM effect within 0.5x to 2x of the human effect. "
          "Holm family: finding x measure x model.", ""]
    for sample in frames:
        md += [f"## Sample: {sample}", "", md_table(rep[rep["sample"] == sample]), ""]
    (OUT / "replication_table.md").write_text("\n".join(md))
    from bigoni_figures import fig1, fig2, fig3   # plotting kept out of this module's line budget
    fig1(cm[cm["sample"] == "all"], human, FIG)
    fig2(cm[cm["sample"] == "all"], FIG)
    import bigoni_prices as P   # price measures and contrasts live in their own module (line budget)
    pr = P.run(groups)
    nums = numbers(cm, ct, rep, rc, qc, rec, frames["all"], summary)
    nums["prices"] = clean(pr["numbers"])
    (OUT / "numbers.json").write_text(json.dumps(nums, indent=1))
    fig3(cm[(cm["sample"] == "all") & (cm.measure == "formation")], pr["cells"], pr["human_avg"], FIG)
    for f in OUT.iterdir():
        text = f.read_text()
        assert BANNED not in text.lower() and "\u2014" not in text, f"banned token in {f}"
    return ct, rep, rc, qc, rec


if __name__ == "__main__":
    import bigoni_analysis   # run the importable copy so bigoni_prices sees one module, not a second __main__ copy

    bigoni_analysis.main()
