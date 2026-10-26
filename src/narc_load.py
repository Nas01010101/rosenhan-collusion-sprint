"""Load NARCBench (Rose et al., arXiv 2604.01151) Qwen3-32B activations into per-run group tensors.

Data: huggingface.co/datasets/aaronrose227/narcbench (main, v3), downloaded to data/narcbench/.
Each tier directory holds metadata_gen.json (one dict per agent-turn) and activations_gen.npz
(keys layer_26..layer_30, float32, rows aligned with metadata). Preexisting data, not ours.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/narcbench"
LAYERS = (26, 27, 28, 29, 30)
N_AGENTS = 4


@dataclass
class Tier:
    name: str
    meta: list
    npz: "np.lib.npyio.NpzFile"
    _cache: dict

    def layer(self, layer: int) -> np.ndarray:
        if layer not in self._cache:
            self._cache[layer] = self.npz[f"layer_{layer}"].astype(np.float32)
        return self._cache[layer]


def load_tier(name: str, root: Path = DATA, model: str = "qwen3_32b") -> Tier:
    d = Path(root) / "activations" / model / name
    meta = json.loads((d / "metadata_gen.json").read_text())
    npz = np.load(d / "activations_gen.npz", allow_pickle=False)
    return Tier(name, meta, npz, {})


@dataclass
class Groups:
    """One row per run: X is (n_runs, 4, d) with agents in sorted-name order."""

    run_ids: list
    y: np.ndarray        # 1 = collusion run, 0 = control run
    family: list         # Transfer family / 'deception' for Core / 'bj' for Stego
    domain: list         # Core scenario_domain, else scenario_id
    X: np.ndarray
    roles: list          # per run: tuple of roles in the same agent order as X


def domain_of(m: dict) -> str:
    """Core domain label; the leave-domain-out unit. scenario_id 'audit_01' -> 'audit'."""
    return m.get("scenario_domain") or "_".join(m["scenario_id"].split("_")[:-1])


def build_groups(tier: Tier, layer: int, phase: str = "discussion") -> Groups:
    """phase='discussion': mean over public discussion rounds per agent (Core training/eval protocol).
    phase='final': the single final-round public activation per agent (Transfer, Stego protocol)."""
    acts = tier.layer(layer)
    rows: dict = {}
    for i, m in enumerate(tier.meta):
        if m["channel"] != "public" or m["phase"] != phase:
            continue
        rows.setdefault(m["run_id"], {}).setdefault(m["agent_name"], []).append(i)
    ids, y, fam, dom, X, roles = [], [], [], [], [], []
    for rid in sorted(rows):
        agents = rows[rid]
        if len(agents) != N_AGENTS:
            continue
        names = sorted(agents)
        first = tier.meta[agents[names[0]][0]]
        ids.append(rid)
        y.append(int(first["mode"] == "collusion"))
        fam.append(first["family"])
        dom.append(domain_of(first))
        roles.append(tuple(tier.meta[agents[n][0]]["role"] for n in names))
        X.append(np.stack([acts[agents[n]].mean(0) for n in names]))
    return Groups(ids, np.array(y), fam, dom, np.stack(X), roles)


def probe_rows(tier: Tier, exclude_domain: str | None = None):
    """Indices of colluder and honest public-discussion rows inside Core collusion runs.
    Control-run rows are never used to fit the deception direction (published protocol)."""
    col, hon = [], []
    for i, m in enumerate(tier.meta):
        if m["mode"] != "collusion" or m["channel"] != "public" or m["phase"] != "discussion":
            continue
        if exclude_domain is not None and domain_of(m) == exclude_domain:
            continue
        (col if m["role"] == "colluder" else hon).append(i)
    return col, hon
