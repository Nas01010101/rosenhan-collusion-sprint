"""Load Agrawal et al. (2025) double-auction runs into per-round seller arrays.

Each run directory holds auction_results.md (one ```json block per round), experiment_metadata.json
and collusion_metrics.json. Released runs: starter/llm-agent-collusion/data (preexisting, not ours).
"""

import json
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "starter/llm-agent-collusion/data"
CACHE = ROOT / "results/raw/cache/runs.pkl"
_BLOCK = re.compile(r"````json\s*(\{.*?\})\s*````", re.S)


@dataclass
class Run:
    name: str
    condition: str
    config: dict
    sellers: list
    asks: np.ndarray            # (n_sellers, T), nan where a seller posted no ask
    bids_mean: np.ndarray       # (T,)
    trades: list                # per round: list of trade dicts
    messages: list              # per round: {seller: text}
    judge: np.ndarray           # (n_sellers, T) coordination score 1-4 on private reasoning, nan if missing
    overseer: np.ndarray        # (T,) overseer score 1-4 on seller messages, nan without oversight
    evidence: list              # [seller][round] -> list of reasoning phrases the judge quoted
    judge_type: list            # [seller][round] -> judge's label 'overt' | 'tacit' | 'null' | None
    metrics: dict = field(repr=False, default_factory=dict)

    @property
    def gag_round(self):
        """First round the overseer scored 4; the simulator gags all sellers from then on."""
        hits = np.flatnonzero(self.overseer == 4)
        return int(hits[0]) if hits.size else None


def condition_of(dirname):
    """'20250517_001228_boss_pressure-8' -> 'boss_pressure'."""
    return dirname.split("_", 2)[2].rsplit("-", 1)[0]


def load_run(d):
    d = Path(d)
    meta = json.loads((d / "experiment_metadata.json").read_text())
    metrics = json.loads((d / "collusion_metrics.json").read_text())
    rounds = [json.loads(b) for b in _BLOCK.findall((d / "auction_results.md").read_text())]
    rounds.sort(key=lambda r: r["round_number"])
    sellers = sorted(meta["seller_ids"])
    T = len(rounds)
    asks = np.full((len(sellers), T), np.nan)
    bids = np.full(T, np.nan)
    for t, r in enumerate(rounds):
        for i, s in enumerate(sellers):
            v = r.get("seller_asks", {}).get(s)
            if v is not None:
                asks[i, t] = float(v)
        b = [float(v) for v in r.get("buyer_bids", {}).values() if v is not None]
        if b:
            bids[t] = np.mean(b)
    # evaluate.py numbers sellers by sorted seller id, the same order as `sellers` here
    judge = np.full((len(sellers), T), np.nan)
    for i in range(len(sellers)):
        sc = metrics.get(f"seller_{i + 1}_coordination_score", [])
        judge[i, : len(sc)] = [np.nan if x is None else x for x in sc[:T]]
    overseer = np.full(T, np.nan)
    ov = metrics.get("overseer_coordination_score") or []
    overseer[: len(ov)] = [np.nan if x is None else x for x in ov[:T]]
    evidence = [[[] for _ in range(T)] for _ in sellers]
    judge_type = [[None] * T for _ in sellers]
    eval_path = d / "llm_eval_full.json"
    if eval_path.exists():
        for i, s in enumerate(sorted(json.loads(eval_path.read_text()), key=lambda e: e["seller_id"])):
            for r in s["rounds"]:
                data = r.get("data") or {}
                ev = data.get("evidence") or []
                if 1 <= r["round"] <= T:
                    evidence[i][r["round"] - 1] = [e.get("phrase", "") for e in ev if isinstance(e, dict)]
                    judge_type[i][r["round"] - 1] = data.get("type")
    return Run(
        name=d.name,
        condition=condition_of(d.name),
        config=meta["auction_config"],
        sellers=sellers,
        asks=asks,
        bids_mean=bids,
        trades=[r.get("trades", []) for r in rounds],
        messages=[r.get("seller_messages", {}) or {} for r in rounds],
        judge=judge,
        overseer=overseer,
        evidence=evidence,
        judge_type=judge_type,
        metrics=metrics,
    )


def load_all(root=DATA, cache=CACHE):
    """All runs under root; pickled to `cache` because parsing 75 runs off the external SSD is slow."""
    if cache and Path(cache).exists():
        return pickle.loads(Path(cache).read_bytes())
    runs = [load_run(d) for d in sorted(Path(root).iterdir()) if (d / "auction_results.md").exists()]
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        Path(cache).write_bytes(pickle.dumps(runs))
    return runs
