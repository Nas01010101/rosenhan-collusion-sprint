"""One cell x seed of the off-camera study (notes/prereg_offcamera.md) on the starter's double auction.

3 LLM sellers with chat, 3 ZIP buyers, 20 rounds. The starter's agents, market and seller template are imported
unchanged (starter/ is preexisting); this file swaps the agent identities and wraps the rendered seller prompt
through offcamera_prompts. Inference goes through the starter's env overrides LLM_BASE_URL / LLM_API_KEY /
LLM_MODEL (util/poseidon_compute/llm_config.py). Writes results/offcamera/raw/<cell>_<seed>.json.

    LLM_BASE_URL=http://localhost:8000/v1 LLM_API_KEY=EMPTY python src/exp_offcamera.py --cell test_none --seed 0
"""

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STARTER = ROOT / "starter/llm-agent-collusion"
for p in (STARTER, ROOT / "starter/util", Path(__file__).resolve().parent):
    sys.path.insert(0, str(p))

import offcamera_prompts as P  # noqa: E402
from src.continuous_double_auction.cda_types import ExperimentParams  # noqa: E402
from src.continuous_double_auction.identities import (  # noqa: E402
    get_random_agent_names, get_random_buyer_company_names, get_random_seller_company_names,
)
from src.continuous_double_auction.market import agents as agents_mod  # noqa: E402
from src.continuous_double_auction.market.agents import LMSeller, ZIPBuyer  # noqa: E402
from src.continuous_double_auction.market.market import Market  # noqa: E402
from src.continuous_double_auction.util.json_util import extract_json  # noqa: E402
from src.continuous_double_auction.util.misc import get_client, initialize_market  # noqa: E402
from src.resources.model_wrappers import Message  # noqa: E402

MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
TEMPERATURE = 0.7   # starter default, fixed for every cell
OUT = ROOT / "results/offcamera/raw"


class TolerantSeller(LMSeller):
    """Starter seller that records a failed turn (after its own retries) instead of killing the run."""

    n_fail: int = 0

    def generate_bid_response(self, **kwargs):
        try:
            return super().generate_bid_response(**kwargs)
        except Exception as e:  # noqa: BLE001  malformed JSON or endpoint error after the starter's retries
            self.n_fail += 1
            print(f"[{self.id}] turn failed: {type(e).__name__}", file=sys.stderr)
            return {}


def identities(realism, n_sellers, n_buyers):
    """(seller ids, seller companies, buyer ids, buyer companies). Test-like uses placeholders, else starter names."""
    if realism == "test":
        letters = "ABCDEFGH"
        return ([f"Seller_{letters[i]}" for i in range(n_sellers)], [f"Company_{letters[i]}" for i in range(n_sellers)],
                [f"Buyer_{i + 1}" for i in range(n_buyers)], [f"Buyer_Company_{i + 1}" for i in range(n_buyers)])
    names = get_random_agent_names(n_buyers + n_sellers)
    return ([f"Seller {n}" for n in names[n_buyers:]], get_random_seller_company_names(n_sellers),
            [f"Buyer {n}" for n in names[:n_buyers]], get_random_buyer_company_names(n_buyers))


def _number(text):
    """Probability in [0, 100] from a JSON reply, falling back to the first number in the text."""
    if text is None:
        return None
    try:
        d = extract_json(text)
        v = next(iter(d.values()))
    except Exception:  # noqa: BLE001
        m = re.search(r"-?\d+(?:\.\d+)?", text)
        v = m.group() if m else None
    try:
        return min(100.0, max(0.0, float(v)))
    except (TypeError, ValueError):
        return None


def _price_index(asks):
    if not asks:
        return None
    return (float(np.mean(list(asks.values()))) - P.CE_ASK) / (P.JPM_ASK - P.CE_ASK)


def run_cell(cell_name, seed, rounds=P.ROUNDS, model=None, temperature=TEMPERATURE, factory=None, n_sellers=3, n_buyers=3):
    """Run one market and return the raw-record dict. `factory(model, temperature, **kw)` builds a ModelWrapper."""
    cell = P.CELLS[cell_name]
    model = os.environ.get("LLM_MODEL") or model or MODEL
    if factory is None:
        if not os.environ.get("LLM_BASE_URL"):
            sys.exit("set LLM_BASE_URL (a local vLLM server); refusing to fall through to a paid remote endpoint")
        factory = get_client
    random.seed(seed)
    np.random.seed(seed)

    sid, scomp, bid, bcomp = identities(cell.realism, n_sellers, n_buyers)
    params = ExperimentParams(
        seller_valuations=[P.COST] * n_sellers, buyer_valuations=[P.BUYER_VALUE] * n_buyers,
        seller_models=[model] * n_sellers, buyer_models=["zip"] * n_buyers, rounds=rounds,
        seller_comms_enabled=True, boss_pressure=cell.ceo, temperature=temperature, tag=cell.name, seed=seed,
    )
    sellers = [TolerantSeller(id=sid[i], company=scomp[i], valuation=P.COST, expt_params=params,
                              client=factory(model=model, temperature=temperature, seed=seed)) for i in range(n_sellers)]
    buyers = [ZIPBuyer(id=bid[i], company=bcomp[i], valuation=P.BUYER_VALUE, expt_params=params) for i in range(n_buyers)]
    market = Market(sellers=sellers, buyers=buyers)
    ask0, bid0 = initialize_market(n_sellers, params.initial_seller_ask_center, params.initial_seller_ask_spread,
                                   n_buyers, params.initial_buyer_bid_center, params.initial_buyer_bid_spread)
    market.set_initial_orders([(ask0[i], sid[i]) for i in range(n_sellers)], [(bid0[i], bid[i]) for i in range(n_buyers)])

    opening = {}   # seller id -> (full round-1 prompt, round-1 prompt without the detector claim)
    orig = agents_mod.render_prompt

    def render(**kw):
        base = orig(**kw)
        if "seller_id" not in kw:
            return base
        r = kw["round_number"]
        if r == 1:
            opening[kw["seller_id"]] = (P.build_seller_prompt(base, cell, 1), P.build_seller_prompt(base, cell, 1, False))
        return P.build_seller_prompt(base, cell, r)

    agents_mod.render_prompt = render
    try:
        for _ in range(rounds):
            market.run_round()
    finally:
        agents_mod.render_prompt = orig

    log = [dict(round=r.round_number, asks=r.seller_asks, bids=r.buyer_bids, messages=r.seller_messages,
                trades=[t.model_dump() for t in r.trades], index=_price_index(r.seller_asks)) for r in market.rounds]

    judge = factory(model=model, temperature=temperature, seed=seed)

    def ask(prompt):
        return judge.generate([Message(role="user", content=prompt)])

    p_caught = {s: _number(ask(P.p_caught_prompt(opening[s][0]))) for s in sid if s in opening}
    first = sid[0]
    p_eval = _number(ask(P.p_eval_prompt(opening[first][1]))) if first in opening else None
    return dict(
        cell=cell.name, seed=seed, model=model, temperature=temperature, rounds=rounds,
        realism=cell.realism, certainty=cell.certainty, severity=cell.severity, disclose=cell.disclose, ceo=cell.ceo,
        sellers=sid, ce=P.CE_ASK, jpm=P.JPM_ASK, threshold=P.THRESHOLD, n_fail=sum(s.n_fail for s in sellers),
        rounds_log=log, p_caught=p_caught, p_eval=p_eval,
        opening_prompt=opening[first][0] if first in opening else None,
    )


def write_record(rec, out=OUT):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{rec['cell']}_{rec['seed']}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=1, default=float))
    tmp.replace(path)   # a killed job never leaves a half-written file behind
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", required=True, choices=list(P.CELLS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--rounds", type=int, default=P.ROUNDS)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--model", default=None)
    ap.add_argument("--temperature", type=float, default=TEMPERATURE)
    a = ap.parse_args()
    rec = run_cell(a.cell, a.seed, a.rounds, a.model, a.temperature)
    idx = [r["index"] for r in rec["rounds_log"] if r["index"] is not None]
    third = idx[-max(1, len(idx) // 3):]
    print(f"{write_record(rec, a.out)}  last-third index {np.mean(third):+.3f}  fails {rec['n_fail']}  "
          f"p_eval {rec['p_eval']}  p_caught {list(rec['p_caught'].values())}")


if __name__ == "__main__":
    main()
