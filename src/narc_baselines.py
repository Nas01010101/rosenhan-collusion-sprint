"""Run the published NARCBench protocol on released Qwen3-32B activations; write results/narc/baselines.json.

Published numbers: arXiv 2604.01151 v5 Table 2 (Core/Transfer/Stego) and Table 1 (Verdict). Verdict and
Exploit-Share activations are not in the HF release, so those cells are null with the reason recorded.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from narc_load import LAYERS, ROOT, load_tier  # noqa: E402
from narc_probes import METHODS, auc_rows, boot_auc, held_out_scores, zero_shot_scores  # noqa: E402

OUT = ROOT / "results/narc/baselines.json"
FAMILIES = ("cicd", "code_review", "news", "pump_dump", "rag", "research")
# order: sorted_concat, activation_alignment, peak_suspicion, deception_split, asymmetry_probe
PUB = {
    "held_out_core": [1.00, 1.00, 1.00, 1.00, 1.00],
    "transfer_mean": [0.82, 0.48, 0.84, 0.88, 0.87],
    "cicd": [0.53, 0.28, 0.32, 0.73, 0.48],
    "code_review": [0.94, 0.39, 1.00, 0.99, 0.99],
    "news": [0.71, 0.71, 0.94, 0.96, 0.94],
    "pump_dump": [0.99, 0.32, 0.97, 0.83, 0.96],
    "rag": [0.77, 0.47, 0.82, 0.77, 0.83],
    "research": [0.99, 0.68, 1.00, 1.00, 1.00],
    "stego": [1.00, 1.00, 1.00, 0.99, 1.00],
    "verdict_reflection": [0.728, 0.638, 0.490, 0.512, 0.495],
}


def summarize(name, S_by_method, y, groups, rng, b, seeds, layers):
    out = {}
    for mi, m in enumerate(METHODS):
        S = S_by_method[m]
        if groups is None:
            point = float(auc_rows(S, y).mean())
            per_ls = auc_rows(S, y)
        else:
            per_ls = np.mean([auc_rows(S[:, groups == g], y[groups == g]) for g in np.unique(groups)], axis=0)
            point = float(per_ls.mean())
        d = boot_auc(S, y, groups, rng, b)
        grid = per_ls.reshape(len(layers), len(seeds))
        pub = PUB[name][mi]
        out[m] = {"published": pub, "ours": round(point, 4),
                  "ci95_run_bootstrap": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)],
                  "std_over_seeds": round(float(grid.mean(0).std()), 5), "std_over_layers": round(float(grid.mean(1).std()), 4),
                  "gap_vs_published": round(point - pub, 4), "gap_gt_0.03": bool(abs(point - pub) > 0.03)}
    return out


def main(b=2000, n_seeds=10):
    t0 = time.time()
    core, tr, st = (load_tier(n) for n in ("core", "transfer", "stego"))
    seeds = list(range(n_seeds))
    rng = np.random.default_rng(0)
    zs = zero_shot_scores(core, {"transfer": tr, "stego": st}, LAYERS, seeds)
    res = {}
    mt = zs["_meta"]["transfer"]
    y, fam = mt["y"], np.array(mt["family"])
    gi = np.array([FAMILIES.index(f) for f in fam])
    res["transfer_mean"] = summarize("transfer_mean", zs["transfer"], y, gi, rng, b, seeds, LAYERS)
    for k, f in enumerate(FAMILIES):
        msk = gi == k
        res[f] = summarize(f, {m: zs["transfer"][m][:, msk] for m in METHODS}, y[msk], None, rng, b, seeds, LAYERS)
    res["stego"] = summarize("stego", zs["stego"], zs["_meta"]["stego"]["y"], None, rng, b, seeds, LAYERS)
    ho = held_out_scores(core, LAYERS, seeds)
    res["held_out_core"] = {m: {"published": PUB["held_out_core"][i], "ours": round(float(ho[m].mean()), 4),
                                "std_over_layer_seed": round(float(ho[m].std()), 4),
                                "gap_vs_published": round(float(ho[m].mean()) - PUB["held_out_core"][i], 4),
                                "ci95_run_bootstrap": None, "note": "leave-domain-out folds; no run bootstrap"}
                            for i, m in enumerate(METHODS)}
    res["verdict_reflection"] = {m: {"published": PUB["verdict_reflection"][i], "ours": None,
                                     "reason": "Verdict activations not in aaronrose227/narcbench main (v3) or anon-review"}
                                 for i, m in enumerate(METHODS)}
    sha = subprocess.run(["git", "-C", str(Path.home() / "scratch/narcbench_src"), "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    doc = {"meta": {"model": "qwen3_32b", "layers": list(LAYERS), "seeds": n_seeds, "bootstrap_b": b,
                    "bootstrap": "stratified run-level; Transfer resamples runs within family x class, averages family AUROCs",
                    "n_runs": {"core": [31, 31], "transfer": [36, 36], "stego": [20, 20]},
                    "hf_commit": "19cb14b269e052b19ba6d2bd2f4519afd3b10c57", "narcbench_git": sha,
                    "published_source": "arXiv 2604.01151 Table 2 / Table 1", "seconds": round(time.time() - t0)},
           "results": res}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1))
    print(OUT)


if __name__ == "__main__":
    main(b=int(sys.argv[1]) if len(sys.argv) > 1 else 2000)
