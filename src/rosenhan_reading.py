"""How each overseer reads the auditor's diary and what it finally calls the labelled seat, per condition.

The diary reading is the Rosenhan "writing behavior" analog: the same notebook lines, rated under each header.
Writes results/rosenhan/reading.json and prints it.

    python src/rosenhan_reading.py
"""

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "results/rosenhan"
OVERSEERS = {"qwen3-30b": ["main_rater", "null_rater", "all_rater"], "phi-4": ["phi4_rater"],
             "gpt-oss-120b": ["gptoss_rater"]}
CONDS = ("neutral", "label", "label_null", "label_all", "access")


def load(d):
    return {p.stem: json.loads(p.read_text()) for p in sorted((ROOT / d).glob("*.json"))}


def shares(counter):
    n = sum(counter.values())
    return {str(k): round(v / n, 3) for k, v in counter.most_common()} | {"n": n} if n else None


def main():
    out = {}
    for name, dirs in OVERSEERS.items():
        rates = {}
        for d in dirs:
            for s, r in load(d).items():
                rates.setdefault(s, dict(r, ratings=[]))["ratings"] += r["ratings"]
        o = out[name] = {"notebook_G2": {}, "verdict": {}}
        for cond in CONDS:
            o["notebook_G2"][cond] = shares(Counter(
                r["notebook"] for g in rates.values() if g["cell"] == "G2"
                for r in g["ratings"] if r["cond"] == cond and r["players"]))
        for cell in ("G0", "G2", "G3"):
            for cond in CONDS[:4]:
                o["verdict"][f"{cell}_{cond}"] = shares(Counter(
                    r["players"][g["label_seat"]]["verdict"] for g in rates.values() if g["cell"] == cell
                    for r in g["ratings"] if r["cond"] == cond and r["players"]))
    (ROOT / "reading.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
