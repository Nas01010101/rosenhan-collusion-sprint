"""One CSV row per Bigoni group from the raw group JSONs.

    python src/bigoni_summary.py results/raw/bigoni/main results/bigoni_main_summary.csv

Definitions (pair-period = one matched pair in one period):
  attempt_rate     share of pair-periods where at least one of the two said yes to talking
  formation_rate   share of pair-periods where both said yes
  ring_price       mean price of the two agents in pair-periods with a formation (valid prices only)
  report_rate_live share of pair-periods at a reporting stage (ring live) where someone reported; blank in C (no reporting)
  avg_price        mean valid price over all agent-periods
  parse_fail       share of game decisions (not quiz/recognisability) replaced by the default
  refusal          share of all decisions whose text matched the refusal regex
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

from bigoni_prompts import classify_label

FIELDS = ["model", "cell", "wording", "paraphrase", "group_id", "complete", "n_periods", "attempt_rate", "comm_yes_rate",
          "formation_rate", "ring_price", "report_rate_live", "avg_price", "quiz_score", "parse_fail", "refusal",
          "recog_coordination", "recog_competition", "recog_game", "recog_other", "recog_labels"]


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else ""


def group_row(d: dict) -> dict:
    s, per = d["spec"], d["periods"]
    live = [p for p in per if p.get("report4") is not None]
    ring_prices = [x for p in per if p["formed"] for x, df in zip(p["prices"], p["price_default"]) if not df]
    prices = [x for p in per for x, df in zip(p["prices"], p["price_default"]) if not df]
    play = [x for x in d["decisions"] if x["kind"] not in ("quiz", "recog")]
    labels = Counter(classify_label(r["label"]) for r in d["recog"])
    return {
        "model": s["model"], "cell": s["cell"], "wording": s["wording"], "paraphrase": s["paraphrase"],
        "group_id": s["group_id"], "complete": d["complete"], "n_periods": len(per),
        "attempt_rate": _mean(any(p["comm"]) for p in per), "comm_yes_rate": _mean(c for p in per for c in p["comm"]),
        "formation_rate": _mean(p["formed"] for p in per), "ring_price": _mean(ring_prices),
        "report_rate_live": _mean(p["reported"] for p in live), "avg_price": _mean(prices),
        "quiz_score": _mean(q["score"] for q in d["quiz"]),
        "parse_fail": _mean(x["default"] for x in play), "refusal": _mean(x["refusal"] for x in d["decisions"]),
        **{f"recog_{k}": labels.get(k, 0) for k in ("coordination", "competition", "game", "other")},
        "recog_labels": "|".join(sorted(str(r["label"]) for r in d["recog"])),
    }


def write(raw_dir: Path, out: Path) -> int:
    # the superseded assembled Qwen set stays on disk for the appendix only
    rows = [group_row(json.load(open(f))) for f in sorted(raw_dir.glob("*/groups/*.json"))
            if "assembled" not in f.parts[-3]]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


if __name__ == "__main__":
    print(write(Path(sys.argv[1]), Path(sys.argv[2])), "rows")
