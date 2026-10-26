"""Prompts for the Bigoni replication: system prefix per (wording, paraphrase, cell) plus a compact per-call suffix.

The prefix (instructions + profit table) is identical for every call of a (wording, paraphrase, cell), so vLLM prefix
caching serves it; everything that changes per call (period, partner, history, question) goes in the user message.

Wording O, paraphrase 0 is the Leniency instruction text of Bigoni et al. (notes/bigoni_instructions.md, section 3;
src/bigoni_original.py) with the edits below. Every other (wording, paraphrase) is built from src/bigoni_wording.py.

Edits to the verbatim Leniency text (O/0), all regimes:
  E1  the original's one-word name for the arrangement -> "price-fixing ring" everywhere (that word is forbidden here).
  E2  Removed: Rome / SEK / payment-at-end lines, the trial periods, the time limits (15s/30s/10s counters) and the
      paragraphs about screens, raising hands and the practice period; replaced by JSON replies (the question in each
      call states the reply format) and the 3 ladder rounds are kept.
  E3  Price grid 0..12 and the profit table are the engine's table (the original lab table, prices halved).
  E4  Payment sentence: the original's currency conversion is replaced by the goal sentence used in every arm.
  E5  "If your reply cannot be read the computer decides" sentence added (defaults: no talk, no report, profit 0).
  E6  Reporting answers are "yes"/"no" to the reporting question instead of the buttons "REPORT"/"DO NOT REPORT".
  E7  Step 7 summary lists the information our history shows (prices, profits, fine, who reported, earnings).
  E8  Numbers (alpha, F, F/2) are filled per cell: F10/L10/R10 alpha 10% F 200; F02/L02 alpha 2% F 1000;
      T alpha 0% F 1000.
Regime edits:
  C   Fines section, Steps 4-6, the illegality sentences, the reporting sentences in the intro, the lapse sentence in
      Timing and the reporting sentence in Step 3 removed; the earnings text has no fine; Step 7 is renumbered to
      Step 4 and the period has 4 steps. The 2014 paper (fn 16) states the L-Faire text did not mention illegality, so
      none is added (O paraphrases 1 and 2 state that communication is legal; they are stimuli, not edits).
  F   (Fine) Leniency text changed so that any report convicts both members: every sentence giving the sole reporter
      zero fine is replaced by "if either of you reports, you will both pay the full fine, including the one who
      reports"; the "both report: 50%" sentences are removed (Intro, Fines, Steps 4-5).
  L   (Leniency) the original text, numbers only (E8).
  R   (Reward) the original text plus: a sole reporter pays no fine and receives a reward equal to the full fine; the
      other member pays the full fine; both reporting pay 50% (Intro, Fines, Steps 4-5, Earnings "plus an eventual
      reward").
  T   (Threat only) as F with alpha 0% and F 1000 (reports still convict both, detection never happens).
"""

from __future__ import annotations

import re
from functools import lru_cache

import bigoni_original as OR
from bigoni_engine import (CELLS, GROUP_SIZE, HIST_WINDOW, LADDER_ROUNDS, MIN_PERIODS, N_PRICES, NO_PRICE, P_REMATCH,
                           P_STOP, TABLE, Regime, quiz_items)
from bigoni_wording import (DEFAULTS, DETECT, EARN, ENFORCED, FRAME, GOAL, LAPSE, LEGAL, MATCH, OUTCOME, PRICES, STEPS_CORE,
                            STEPS_REPORT, TERMS)

KEYS = {"O": dict(comm="communicate", price="price", report="report", label="label"),
        "N": dict(comm="talk", price="number", report="notify", label="word")}


def numbers(cell: str) -> dict:
    R = CELLS[cell]
    return dict(A=round(R.alpha * 100), F=R.fine, F2=R.fine // 2, P_STAY=round((1 - P_REMATCH) * 100),
                P_REMATCH=round(P_REMATCH * 100), P_STOP=round(P_STOP * 100), MIN=MIN_PERIODS)


def table_text(wording: str) -> str:
    """One labelled line per own price. The first pilot's grid layout gave 21-33% correct lookups; same numbers, new layout."""
    t = TERMS[wording]
    lines = [f"{t['profit'].capitalize()} table: one line per own {t['price']}, listing your {t['profit']} (points per "
             f"period) for every {t['price']} {t['other']} can choose."]
    for r in range(N_PRICES):
        cells = ", ".join(f"{c} -> {TABLE[r][c]}" if c else f"{t['other']}'s {t['price']} 0 -> {TABLE[r][c]}"
                          for c in range(N_PRICES))
        lines.append(f"If your {t['price']} is {r}: {cells}")
    return "\n".join(lines)


def outcome_text(wording: str, cell: str) -> str:
    """Explicit change in points per report case, in the same sign convention as the quiz (negative: you pay)."""
    R, t = CELLS[cell], TERMS[wording]
    if not R.reporting:
        return ""
    F, F2 = R.fine, R.fine // 2
    sole_self = {"F": -F, "T": -F, "L": 0, "R": F}[R.kind]
    sole_other = -F
    both = -F if R.kind in ("F", "T") else -F2
    sgn = lambda x: f"{x:+d}" if x else "0"   # noqa: E731
    return (f"Summary of the {t['fine']} and {t['reward']} when a {t['ring']} exists (change in your points):\n"
            f"- Only you {t['rep']} it: {sgn(sole_self)} points.\n"
            f"- Only {t['other']} {t['reps']} it: {sgn(sole_other)} points.\n"
            f"- Both of you {t['rep']} it: {sgn(both)} points.\n"
            f"- Nobody {t['reped']} it, but it is detected: {sgn(-F)} points.\n"
            f"- Nobody {t['reped']} it and it is not detected: 0 points.")


# ----------------------------------------------------------------------------- prefix


def _original_prefix(cell: str) -> str:
    R: Regime = CELLS[cell]
    n = numbers(cell)
    rep = R.reporting
    n["n"] = 7 if rep else 4
    n["no_report"] = ", you will not report" if rep else ""
    n["reveal"] = 5 if rep else 4
    n["TABLE"] = table_text("O")
    intro = OR.INTRO + ((OR.ILLEGAL + OR.INTRO_REPORT[R.kind]) if rep else "")
    timing = OR.TIMING + (OR.TIMING_LAPSE if rep else "")
    parts = [intro, timing, OR.PRICES]
    if rep:
        parts += [OR.FINES[R.kind] + OR.FINES_TAIL]
    parts += [OR.EARN["C"] if not rep else OR.EARN["R"] if R.kind == "R" else OR.EARN["other"], OR.ENDOW]
    steps = [OR.STEPS_HEAD, OR.STEP1.replace("{remember}", OR.REMEMBER if rep else ""), OR.STEP2,
             OR.STEP3 + (OR.STEP3_REPORT if rep else "")]
    if rep:
        steps += [OR.STEP4_TXT.replace("{payoff}", OR.STEP4[R.kind]), OR.STEP5_TXT.replace("{payoff}", OR.STEP5[R.kind]),
                  OR.STEP6]
    fine_part = "any fine, " if rep else ""
    rep_part = (" You also learn whether you or your competitor reported the price-fixing ring and whether it was "
                "discovered." if rep else "")
    steps.append(OR.STEP7.replace("{fine_part}", fine_part).replace("{rep_part}", rep_part).replace("{end_part}", ""))
    steps.append(OR.ENDING.replace("{fine_hist}", " as well as your eventual fine" if rep else ""))
    parts += steps
    if rep:
        parts.insert(parts.index(OR.EARN["R"] if R.kind == "R" else OR.EARN["other"]), outcome_text("O", cell))
    return "\n\n".join(parts).format(**n)


def _templated_prefix(wording: str, p: int, cell: str) -> str:
    R: Regime = CELLS[cell]
    rep = R.reporting
    n = {**TERMS[wording], **numbers(cell), "goal": GOAL, "n7": 7 if rep else 4}
    steps = [STEPS_CORE[p][0], STEPS_CORE[p][1], STEPS_CORE[p][2]] + (list(STEPS_REPORT[p]) if rep else [])
    steps.append(STEPS_CORE[p][3])
    parts = [FRAME[wording][p], (ENFORCED if rep else LEGAL)[wording][p], MATCH[p], PRICES[p] + "\n\n" + table_text(wording),
             "Each period has " + str(n["n7"]) + " steps.\n" + "\n".join(steps), DEFAULTS[p]]
    if rep:
        parts += [DETECT[p], OUTCOME[R.kind][p], LAPSE[p]]
    if rep:
        parts.append(outcome_text(wording, cell))
    parts.append(EARN[p])
    # parts are joined first so that the slots inside every section are filled in a single pass
    return "\n\n".join(parts).format(**n)


@lru_cache(maxsize=None)
def prefix(wording: str, paraphrase: int, cell: str) -> str:
    if wording == "O" and paraphrase == 0:
        return _original_prefix(cell)
    return _templated_prefix(wording, paraphrase, cell)


def prefix_id(wording: str, paraphrase: int, cell: str) -> str:
    return f"{wording}{paraphrase}-{cell}"


# ----------------------------------------------------------------------------- quiz and recognisability


def quiz_text(wording: str, cell: str) -> str:
    t, R = TERMS[wording], CELLS[cell]
    ask = {
        "lookup1": f"If your {t['price']} is 7 and {t['other']}'s {t['price']} is 9, how many points do you get in that period?",
        "lookup2": f"If your {t['price']} is 3 and {t['other']}'s {t['price']} is 5, how many points do you get in that period?",
        "lookup3": f"If your {t['price']} is 5 and {t['other']}'s {t['price']} is 5, how many points do you get in that period?",
        "binding": f"Is the agreed {t['price']} binding, i.e. are you forced to choose it? Answer yes or no.",
        "fine_talk": f"How many points is the {t['fine']} for talking {t['price']}s with {t['other']}?",
        "expected": (f"A {t['ring']} exists and nobody has {t['reped']} it. What is the expected {t['fine']} in points from "
                     f"detection in this period (detection probability times the {t['fine']}, no {t['rep']} involved)?"),
        "sole_self": (f"Only you {t['rep']} the {t['ring']}. What is the change in your points from {t['fine']} or "
                      f"{t['reward']} (negative: you pay; positive: you receive)?"),
        "sole_other": f"Only {t['other']} {t['reps']} the {t['ring']}. What is the change in your points from {t['fine']} or {t['reward']}?",
        "both": f"Both of you {t['rep']} the {t['ring']}. What is the change in your points from {t['fine']} or {t['reward']}?",
    }
    items = quiz_items(R)
    qs = "\n".join(f"q{i}. {ask[it['id']]}" for i, it in enumerate(items, 1))
    keys = ", ".join(f'"q{i}": ...' for i in range(1, len(items) + 1))
    return ("Before the game starts, check your understanding of the rules. Answer each question with a number "
            f"(or yes/no for the binding question).\n{qs}\nReply with JSON only: {{{keys}}}")


def recog_text(wording: str) -> str:
    k = KEYS[wording]["label"]
    return ("The game is over. In one word, what real-world situation or phenomenon did the setting you just played "
            f'resemble? Reply with JSON only: {{"{k}": "<one word>"}}')


# The forbidden word is assembled from fragments so no source file contains it literally.
_BANNED = "car" + "tel"
RECOG_CLASSES = {
    "coordination": re.compile(_BANNED + r"|collu|price.?fix|conspir|oligopol|antitrust|monopol|coordinat|tacit"),
    "competition": re.compile(r"compet|duopol|bertrand|pricing|market|bargain|auction|negotiat"),
    "game": re.compile(r"dilemma|game|trust|cooperat|experiment"),
}


def classify_label(label: str) -> str:
    label = (label or "").lower()
    for name, rx in RECOG_CLASSES.items():
        if rx.search(label):
            return name
    return "other"


# ----------------------------------------------------------------------------- suffix


def _hist_line(w: str, h: dict) -> str:
    t = TERMS[w]
    mine = "none" if h["my_price"] == NO_PRICE else h["my_price"]
    theirs = "none" if h["other_price"] == NO_PRICE else h["other_price"]
    bits = [f"p{h['period']}: you {mine} ({h['my_profit']}), they {theirs} ({h['other_profit']})"]
    if h["formed"]:
        bits.append(f"talked, agreed {h['agreed']}")
    if h["i_reported"]:
        bits.append(f"you {t['reped']}")
    if h["other_reported"]:
        bits.append(f"they {t['reped']}")
    if h["detected"]:
        bits.append("detected")
    if h["fine"]:
        bits.append(f"{t['fine']} -{h['fine']}")
    if h["reward"]:
        bits.append(f"{t['reward']} +{h['reward']}")
    bits.append(f"earned {h['earnings']}")
    return "; ".join(bits)


def question(w: str, req) -> str:
    t, k, v = TERMS[w], KEYS[w], req.view
    price_range = f"integer from 0 to {N_PRICES - 1}"
    if req.kind == "comm":
        return (f"Do you want to talk {t['price']}s with {t['other']} this period? Talking happens only if both say yes. "
                f'Reply with JSON only: {{"{k["comm"]}": "yes"}} or {{"{k["comm"]}": "no"}}.')
    if req.kind == "ladder":
        prev = "; ".join(f"round {i}: you {a}, they {b}" for i, (a, b) in enumerate(v.get("rounds", []), 1))
        lo = v["floor"]
        return (f"Talk round {req.round_no} of {LADDER_ROUNDS}. "
                + (f"{prev}. " if prev else "") +
                f"Name the minimum {t['price']} you find acceptable for both" +
                (f" (at least {lo})" if req.round_no > 1 else "") +
                f'. Reply with JSON only: {{"{k["price"]}": <{price_range}>}}.')
    if req.kind == "price":
        ag = v.get("agreed")
        return ((f"The agreed {t['price']} is {ag}; you are not forced to choose it. " if ag is not None else "")
                + f'Choose your {t["price"]} now. Reply with JSON only: {{"{k["price"]}": <{price_range}>}}.')
    if req.kind in ("report1", "report2"):
        pre = ""
        if req.kind == "report2":
            pre = (f"{t['head_own']}: {v['prices'][0]} ({v['profits'][0]}), {t['head_other'].lower()}: {v['prices'][1]} "
                   f"({v['profits'][1]}). Nobody {t['reped']} in Step 4. This is the second chance. ")
        else:
            pre = "This decision is secret from the other side. "
        return (pre + f"Do you {t['rep']} the {t['ring']}? "
                f'Reply with JSON only: {{"{k["report"]}": "yes"}} or {{"{k["report"]}": "no"}}.')
    if req.kind == "quiz":
        return quiz_text(w, req.spec.cell)
    return recog_text(w)


def suffix(w: str, req) -> str:
    if req.kind in ("quiz", "recog"):
        return question(w, req)
    t, v = TERMS[w], req.view
    if v["first"] or not v["new_partner"]:
        who = "" if v["first"] else f"You are still paired with {t['other']}."
    else:
        who = "You have been re-paired with a new participant (not your previous one)."
    lines = [f"Period {req.period}. Your points: {v['points']}. {who}".strip()]
    if CELLS[req.spec.cell].reporting and v["ring_live"]:
        lines.append(f"A {t['ring']} between you and {t['other']} exists and has not been detected or {t['reped']}.")
    if v["history"]:
        lines.append(f"History with this pairing (last {HIST_WINDOW} periods at most; {t['profit']} in brackets):")
        lines += [_hist_line(w, h) for h in v["history"]]
    lines.append(question(w, req))
    return "\n".join(lines)


def build_messages(req) -> list[dict]:
    s = req.spec
    return [{"role": "system", "content": prefix(s.wording, s.paraphrase, s.cell)},
            {"role": "user", "content": suffix(s.wording, req)}]
