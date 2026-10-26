"""S4 p-values on synthetic markets where sellers are independent given the public state (a true null).

Same generator as tests/test_cda_screens.py; writes results/s4_synthetic_null.json for Figure 4 and
results/s4_summary.json (flag counts, exact CIs, Fisher test). `--summary` redoes only the summary.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from cda_load import ROOT  # noqa: E402
from cda_screens import dependence_test  # noqa: E402
from test_cda_screens import common_cause_market  # noqa: E402

N_SIMS, N_PERM, ALPHA = 200, 1000, 0.05


def flags(p, alpha=ALPHA):
    """Markets flagged at p < alpha, with a Clopper-Pearson 95% CI on the flag rate."""
    k, n = int(np.sum(np.asarray(p) < alpha)), len(p)
    ci = stats.binomtest(k, n).proportion_ci(method="exact")
    return dict(k=k, n=n, rate=k / n, ci95=[float(ci.low), float(ci.high)])


def summary():
    """Flag rates on the synthetic null, the planted agreement and the no-channel LLM runs."""
    sims = json.loads((ROOT / "results/s4_synthetic_null.json").read_text())
    runs = pd.read_csv(ROOT / "results/cda_runs.csv")
    out = {"independent": flags(sims["independent"]), "private_agreement_0.3": flags(sims["private_agreement_0.3"]),
           "llm_no_channel": flags(runs.S4_p[runs.condition == "base"])}
    null, llm = out["independent"], out["llm_no_channel"]
    out["fisher_llm_no_channel_vs_independent_p"] = float(stats.fisher_exact(
        [[llm["k"], llm["n"] - llm["k"]], [null["k"], null["n"] - null["k"]]]).pvalue)
    (ROOT / "results/s4_summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def main():
    rng = np.random.default_rng(7)
    out = {}
    for name, agree in (("independent", 0.0), ("private_agreement_0.3", 0.3)):
        runs = [common_cause_market(rng, agree_prob=agree) for _ in range(N_SIMS)]
        # same window length as the real runs (rounds 10-29 = 20 decisions)
        out[name] = [dependence_test(r, w=slice(10, 30), n_perm=N_PERM, rng=k)[1] for k, r in enumerate(runs)]
        print(name, "flag rate p<0.05:", np.mean(np.array(out[name]) < 0.05))
    out["config"] = dict(n_sims=N_SIMS, n_perm=N_PERM, window=[10, 30])
    (ROOT / "results/s4_synthetic_null.json").write_text(json.dumps(out))
    summary()


if __name__ == "__main__":
    summary() if "--summary" in sys.argv else main()
