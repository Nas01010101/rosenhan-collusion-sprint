"""E0: does the paper's fixed-alpha rule survive LLM-scale sample sizes on scripted RPS?

For each (n_games, cp) cell, 300 independent datasets. Detectors:
  fixed      paper rule, plug-in CMI, alpha=0.05
  fixed_mm   paper rule with Miller-Madow correction
  perm       permutation test on gamma for every pair, Bonferroni over 3 pairs (p < 0.05/3),
             must flag exactly one pair; that pair must also have positive min net influence
Reports detection accuracy (correct pair) for cp>0 and false-alarm rate (any pair) for cp=0.
--state prev_joint conditions every estimate on the previous joint action (paper-faithful variant).
"""

import argparse

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from netinf import detect, pair_scores, permutation_pvalues, prev_joint_state  # noqa: E402
from rps import play  # noqa: E402

N_GAMES = [15, 30, 60, 120, 250, 1000]
CPS = [0.0, 0.2, 0.4, 0.6, 1.0]
REPS = 300
N_PERM = 200


def perm_detect(acts, seed, s=None):
    res = permutation_pvalues(acts, s, n_perm=N_PERM, rng=seed)
    sc = pair_scores(acts, s)
    hits = [p for p, (_, pv) in res.items() if pv < 0.05 / 3 and sc[p] > 0]
    return hits[0] if len(hits) == 1 else None


def main(out, state):
    rows = []
    for n in N_GAMES:
        for cp in CPS:
            dets = {"fixed": [], "fixed_mm": [], "perm": []}
            for r in range(REPS):
                seed = hash((n, cp, r)) % 2**32
                acts = play(n, cp, rng=seed)
                s = prev_joint_state(acts) if state == "prev_joint" else None
                dets["fixed"].append(detect(acts, s))
                dets["fixed_mm"].append(detect(acts, s, miller_madow=True))
                dets["perm"].append(perm_detect(acts, seed, s))
            for name, d in dets.items():
                if cp == 0:
                    rate = float(np.mean([x is not None for x in d]))
                    metric = "false_alarm"
                else:
                    rate = float(np.mean([x == (0, 1) for x in d]))
                    metric = "accuracy"
                rows.append(dict(n_games=n, cp=cp, state=state, detector=name, metric=metric, rate=rate, reps=REPS))
                print(f"n={n:5d} cp={cp:.1f} {name:9s} {metric:11s} {rate:.3f}", flush=True)
    Path(out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", choices=["none", "prev_joint"], default="none")
    ap.add_argument("--out")
    a = ap.parse_args()
    suffix = "" if a.state == "none" else f"_{a.state}"
    main(a.out or f"results/e0_rps_calibration{suffix}.json", a.state)
