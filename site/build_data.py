"""Writes site/data.js (window.PAGE_DATA) for the project page from results/rosenhan.

Rates and CIs are recomputed from the rating files with the same code paths as src/rosenhan_figures.py: the exposed,
label and prime CIs share one rng = default_rng(0) in the figure's call order, so the page whiskers equal the PNG's.
The other exposure metrics use their own rng. Means are checked against results/rosenhan/exposure.json.

    python site/build_data.py      (needs numpy; run from the repo root or anywhere)
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import expo_rosenhan as E  # noqa: E402
import rate_rosenhan as R  # noqa: E402

RES = ROOT / "results/rosenhan"
CONDS = E.CONDS
METRICS = ["exposed", "caught", "cleared", "framed", "inverted"]
NAMES = ["Lily", "Luke", "Mike", "Quinn"]
# copied from src/rosenhan_figures.py, which imports matplotlib
RATERS = {"qwen3-30b": ["main_rater", "null_rater", "all_rater", "unprimed_qwen"],
          "phi-4": ["phi4_rater", "unprimed_phi4"], "gpt-oss-120b": ["gptoss_rater", "unprimed_gptoss"]}
EXPO_ORDER = ["neutral", "access", "report_claim", "report_cred", "cred_access", "cred_forged", "cred_conflict",
              "cred_system", "cred_system_ih", "dueling", "dueling_checked", "report_forged", "forged_checked"]
LABEL_CONDS = ["label", "label_null", "label_all", "neutral"]
PRIME_CONDS = ["unprimed", "neutral", "told"]


def boot_ci(x, rng, b=10_000):   # src/figures.py
    x = np.asarray(x, float)
    m = rng.choice(x, (b, len(x))).mean(axis=1)
    return np.percentile(m, [2.5, 97.5])


def load_all(dirs):   # src/rosenhan_figures.py
    rates = {}
    for d in dirs:
        for p in sorted((RES / d).glob("*.json")):
            r = json.loads(p.read_text())
            rates.setdefault(p.stem, dict(r, ratings=[]))["ratings"] += r["ratings"]
    return rates


def mean_over(g, cond, seat, key):   # src/rosenhan_figures.py
    v = [r["players"][seat][key] for r in g["ratings"] if r["cond"] == cond and r["players"]]
    return float(np.mean(v)) if v else None


def stat(x, ci):
    return [round(float(np.mean(x)), 4), round(float(ci[0]), 4), round(float(ci[1]), 4)]


def exposure(rng0, rng1, expo_json):
    """{overseer: {cond: {metric: [mean, lo, hi], n}}} plus the per-game scores used to pick the example game."""
    out, pgs = {}, {}
    for name, dirs in E.OVERSEERS.items():
        rates = E.load(dirs)
        pg = {c: E.per_game(rates, c) for c in CONDS}
        pgs[name] = pg
        out[name] = {c: dict(n=len(pg[c])) for c in CONDS}
    for name in E.OVERSEERS:   # figure order: overseer-major, then EXPO order, exposed only, on rng0
        for c in EXPO_ORDER:
            x = [v["exposed"] for v in pgs[name][c].values()]
            out[name][c]["exposed"] = stat(x, boot_ci(x, rng0))
    for name in E.OVERSEERS:
        for c in EXPO_ORDER:
            for k in METRICS[1:]:
                x = [v[k] for v in pgs[name][c].values()]
                out[name][c][k] = stat(x, boot_ci(x, rng1))
            for k in METRICS:
                ref = expo_json[name]["rates"][c][k]
                assert abs(out[name][c][k][0] - ref) < 1e-3, (name, c, k, out[name][c][k][0], ref)
    return out, pgs


def label_prime(rng0):
    label, prime = {}, {}
    for name, dirs in RATERS.items():   # fig_label, then fig_prime below, both continuing rng0
        g0 = [g for g in load_all(dirs).values() if g["cell"] == "G0"]
        label[name] = {c: stat(x, boot_ci(x, rng0)) for c in LABEL_CONDS
                       for x in [[mean_over(g, c, g["label_seat"], "p_arrangement") for g in g0]]}
    for name, dirs in RATERS.items():
        g0 = [g for g in load_all(dirs).values() if g["cell"] == "G0"]
        prime[name] = {c: stat(x, boot_ci(x, rng0)) for c in PRIME_CONDS
                       for x in [[float(max(mean_over(g, c, s, "p_auditor") for s in NAMES) >= E.FLAG) for g in g0]]}
    return label, prime


def pick_example(pgs, games):
    """G2/G3 game whose report names exactly the colluder and whose scores sit closest to the condition means."""
    best = None
    for stem, g in games.items():
        if g["auditor_report"] is None or g["auditor_report"]["colluders"] != [g["channel"]["initiator"]]:
            continue
        if not all(stem in pgs[o][c] for o in pgs for c in CONDS):
            continue
        dist = 0.0
        for o in pgs:
            for c in CONDS:
                for k in ("exposed", "inverted"):
                    mean = np.mean([v[k] for v in pgs[o][c].values()])
                    dist += abs(pgs[o][c][stem][k] - mean)
        chan = sum(len(m["message"]) + len(m["response"]) for m in g["secret_messages"])
        key = (round(dist, 9), chan)
        if best is None or key < best[0]:
            best = (key, stem)
    return best[1], best[0]


def blocks(game, cond):
    """The prompt split into its parts, checked to rebuild the exact text the overseer received."""
    user_cond = "cred_forged" if cond in R.SYSTEM_CONDS else cond
    text = R.prompt(game, user_cond)
    role = text.split("\n\n", 1)[0]
    ask = text[text.rindex("\n\nFor each player (") + 2:]
    parts = [["role", role], ["base", R.BASE_RATE], ["log", None]]
    if user_cond in R.REPORT_CONDS:
        parts.append(["note", R.exposure_note(game, user_cond)])
    if user_cond in ("access", "cred_access", "cred_forged", "cred_conflict"):
        parts.append(["channel", None])
    parts.append(["ask", ask])
    full = {"log": "Public game record:\n\n" + game["public_log"], "channel": channel_text(game)}
    assert "\n\n".join(t if t is not None else full[k] for k, t in parts) == text, cond
    msgs = R.messages(game, cond)
    system = msgs[0]["content"] if msgs[0]["role"] == "system" else None
    assert msgs[-1]["content"] == text
    return dict(system=system, user=parts)


def channel_text(game):
    msgs = "\n".join(f"[round {m['round']}] {m['sender']} -> {m['receiver']}: {m['message']}\n"
                     f"  reply: {m['response']}" for m in game["secret_messages"]) or "(none)"
    return "Secret-channel messages, which the other players could not see:\n\n" + msgs


def example(stem, pgs):
    game = json.loads((RES / "games" / f"{stem}.json").read_text())
    ratings = {}
    for name, dirs in E.OVERSEERS.items():
        g = E.load(dirs)[stem]
        ratings[name] = {}
        for c in CONDS:
            rs = [r for r in g["ratings"] if r["cond"] == c and r["players"]]
            seats = {}
            for s in game["seats"]:
                ps = [r["players"][s] for r in rs]
                reason = ps[0]["reason"]
                seats[s] = dict(v=[R.VERDICTS.index(p["verdict"]) if p["verdict"] in R.VERDICTS else -1 for p in ps],
                                pa=round(float(np.mean([p["p_arrangement"] for p in ps])), 1),
                                pu=round(float(np.mean([p["p_auditor"] for p in ps])), 1),
                                r=reason if len(reason) <= 160 else reason[:157].rsplit(" ", 1)[0] + " ...")
            ratings[name][c] = dict(seats=seats, game={k: round(pgs[name][c][stem][k], 4) for k in METRICS})
    return dict(stem=stem, cell=game["cell"], seed=game["seed"], seats=game["seats"], roles=game["roles"],
                rounds=game["public_log"].count("Round "), log="Public game record:\n\n" + game["public_log"],
                channel=channel_text(game), n_messages=len(game["secret_messages"]),
                prompts={c: blocks(game, c) for c in CONDS}, ratings=ratings)


def main():
    expo_json = json.loads((RES / "exposure.json").read_text())
    rng0 = np.random.default_rng(0)
    expo, pgs = exposure(rng0, np.random.default_rng(1), expo_json)
    label, prime = label_prime(rng0)
    games = {p.stem: json.loads(p.read_text()) for p in sorted((RES / "games").glob("G[23]_*.json"))}
    stem, key = pick_example(pgs, games)
    stats = {f: json.loads((RES / f"{f}.json").read_text())["H1"]
             for f in ("stats", "stats_phi4_rater", "stats_gptoss_rater")}
    dc = stats["stats"]["declined_challenge_auroc"]
    auroc = dict(overseers={o: dict(auroc=round(s["overseer_auroc_colluder"], 4),
                                    ci=[round(x, 4) for x in s["overseer_auroc_ci"]],
                                    excl_lily=round(s["seat_matched_auroc"]["excluding_lily"], 4))
                            for o, s in zip(E.OVERSEERS, stats.values())},
                 count=dict(auroc=round(dc["count"], 4), ci=[round(x, 4) for x in dc["count_ci"]]),
                 given_only=round(dc["given_only"], 4), pair_vs_rest=dc["pair_vs_rest"])
    h = json.loads((RES / "stats.json").read_text())
    data = dict(overseers=list(E.OVERSEERS), conds=CONDS, metrics=METRICS, expo=expo, label=label, prime=prime,
                auroc=auroc, h2=h["H2_G2"]["any_peer_vs_overseer"], h3=h["H3"],
                reports=expo_json["auditor_reports"], verdicts=R.VERDICTS,
                example=example(stem, pgs) | dict(pick_distance=key[0]))
    out = ROOT / "site/data.js"
    out.write_text("// generated by site/build_data.py from results/rosenhan; do not edit\nwindow.PAGE_DATA = "
                   + json.dumps(data, separators=(",", ":"), ensure_ascii=False) + ";\n")
    print(f"example {stem} (distance {key[0]:.3f}, channel chars {key[1]}); wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
