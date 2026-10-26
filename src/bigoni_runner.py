"""Runs Bigoni groups concurrently against an OpenAI-compatible server (vLLM), plus the grid, a local random policy,
run statistics and a throughput probe.

    python src/bigoni_runner.py --smoke --cells F10 --groups 2        # CPU, random policy, no network

All groups of all cells go into one event loop; a semaphore bounds the requests in flight. A soft deadline cancels the
unfinished groups and writes them with complete=false, so a run never overshoots its reserved GPU time.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import time
from pathlib import Path

from bigoni_engine import CELLS, GROUP_SIZE, NO_PRICE, Group, GroupSpec, Request, quiz_items
from bigoni_prompts import KEYS, build_messages

MODELS_PRIMARY = "qwen3-8b"
# (cell, wording) combinations from notes/prereg_replication.md; the primary table has 11 rows although the prereg
# text says "12 cells"; per the post-pilot amendment the primary grid is 11 x 8 = 88 groups, the two others 7 x 6 = 42.
PRIMARY_CELLS = [("C", "O"), ("C", "N"), ("F10", "O"), ("F10", "N"), ("F02", "O"), ("L10", "O"), ("L02", "O"),
                 ("R10", "O"), ("R10", "N"), ("T", "O"), ("T", "N")]
OTHER_CELLS = [("C", "O"), ("C", "N"), ("F10", "O"), ("F10", "N"), ("R10", "O"), ("T", "O"), ("T", "N")]
SEED_BASE = 20261024
MAX_TOKENS = {"comm": 32, "ladder": 32, "price": 32, "report1": 32, "report2": 32, "quiz": 96, "recog": 32}


def grid(model: str, tag: str, cells=None, n_groups=None) -> list[GroupSpec]:
    """Group k of every (cell, wording) shares the schedule seed, and its paraphrase is k % 3."""
    primary = model == MODELS_PRIMARY
    combos = cells or (PRIMARY_CELLS if primary else OTHER_CELLS)
    n = n_groups or (8 if primary else 6)
    # replicate-major order: when a deadline cuts a run, every row has lost the same number of replicates
    return [GroupSpec(f"{tag}_{c}{w}_g{k}", c, SEED_BASE + k, w, k % 3, model) for k in range(n) for c, w in combos]


# ----------------------------------------------------------------------------- policies


async def random_policy(req: Request):
    """Deterministic stand-in for a model: lets the whole pipeline run on CPU."""
    rng = random.Random(req.seed)
    w = req.spec.wording
    k = KEYS[w]
    yn = lambda p: "yes" if rng.random() < p else "no"   # noqa: E731
    if req.kind == "comm":
        return json.dumps({k["comm"]: yn(0.65)})
    if req.kind == "ladder":
        return json.dumps({k["price"]: rng.randint(max(req.view["floor"], 5), 10)})
    if req.kind == "price":
        ag = req.view.get("agreed")
        return json.dumps({k["price"]: ag if ag is not None and rng.random() < 0.6 else rng.randint(3, 9)})
    if req.kind in ("report1", "report2"):
        return json.dumps({k["report"]: yn(0.15)})
    if req.kind == "quiz":
        items = quiz_items(CELLS[req.spec.cell])
        return json.dumps({f"q{i}": it["answer"] if rng.random() < 0.8 else 0 for i, it in enumerate(items, 1)})
    return json.dumps({k["label"]: rng.choice(["market", "collusion", "game"])})


class LLMPolicy:
    """Chat-completions client with a global in-flight cap; keeps a sample of prompts for the throughput probe."""

    def __init__(self, base_url, model, extra_body=None, extra_tokens=0, concurrency=512, sample=300):
        from openai import AsyncOpenAI
        import httpx

        self.client = AsyncOpenAI(base_url=base_url, api_key="EMPTY", timeout=180, max_retries=2,
                                  http_client=httpx.AsyncClient(limits=httpx.Limits(max_connections=concurrency + 64)))
        self.model, self.extra_body, self.extra_tokens = model, extra_body or {}, extra_tokens
        self.sem, self.sample, self.max_sample = asyncio.Semaphore(concurrency), [], sample
        self.n_calls = self.prompt_tokens = self.cached_tokens = self.completion_tokens = 0

    async def complete(self, messages, max_tokens, seed):
        async with self.sem:
            t0 = time.perf_counter()
            r = await self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=1.0, max_tokens=max_tokens + self.extra_tokens,
                seed=seed, extra_body=self.extra_body)
            lat = time.perf_counter() - t0
        u = r.usage
        details = getattr(u, "prompt_tokens_details", None)
        self.n_calls += 1
        self.prompt_tokens += u.prompt_tokens
        self.completion_tokens += u.completion_tokens
        self.cached_tokens += (getattr(details, "cached_tokens", 0) or 0) if details else 0
        c = r.choices[0]
        return {"text": c.message.content or "", "pt": u.prompt_tokens, "ct": u.completion_tokens,
                "finish": c.finish_reason, "lat": round(lat, 3)}

    async def __call__(self, req: Request):
        messages = build_messages(req)
        if len(self.sample) < self.max_sample and req.seed % 7 == 0:
            self.sample.append((messages, MAX_TOKENS[req.kind]))
        try:
            return await self.complete(messages, MAX_TOKENS[req.kind], req.seed)
        except Exception as e:   # noqa: BLE001  server error after retries: logged as an unreadable reply
            return {"text": "", "error": f"{type(e).__name__}: {e}"[:200]}


class ReplayPolicy:
    """Resumes unfinished groups: calls already logged are answered from the log (same seed, same reply), so the engine
    reaches the exact state where the group was cut off and only the remaining calls go to the model."""

    def __init__(self, inner, partial_groups: list[dict]):
        self.inner, self.n_replayed = inner, 0
        self.replay = {f"{g['spec']['group_id']}:{d['seed']}":
                       {"text": d["raw"], **{k[2:]: v for k, v in d.items() if k.startswith("m_")}}
                       for g in partial_groups for d in g["decisions"]}

    async def __call__(self, req: Request):
        hit = self.replay.get(f"{req.spec.group_id}:{req.seed}")
        if hit is None:
            return await self.inner(req)
        self.n_replayed += 1
        return hit


async def throughput_probe(policy: LLMPolicy, n: int, deadline: float):
    """Replay logged prompts at full concurrency to measure saturated tokens/s (the main grid's regime)."""
    if not policy.sample:
        return {}
    t0, c0, p0 = time.perf_counter(), policy.completion_tokens, policy.prompt_tokens

    async def one(i):
        if time.time() > deadline:
            return
        m, mt = policy.sample[i % len(policy.sample)]
        try:
            await policy.complete(m, mt, 1_000_000 + i)
        except Exception:   # noqa: BLE001
            pass

    await asyncio.gather(*[one(i) for i in range(n)])
    dt = time.perf_counter() - t0
    return {"n": n, "wall_s": dt, "completion_tok": policy.completion_tokens - c0,
            "prompt_tok": policy.prompt_tokens - p0, "completion_tok_s": (policy.completion_tokens - c0) / dt,
            "req_s": n / dt}


# ----------------------------------------------------------------------------- run + stats


async def run_specs(specs, policy, out_dir: Path, deadline: float, quiz=True, recog=True, play=True):
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = [Group(s, policy, quiz=quiz, recog=recog, play=play) for s in specs]
    tasks = {asyncio.create_task(g.run()): g for g in groups}
    done, pending = await asyncio.wait(tasks, timeout=max(1.0, deadline - time.time()))
    for t in pending:
        t.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    results = []
    for t, g in tasks.items():
        if t in done and t.exception():
            g.error = repr(t.exception())
        d = g.to_dict()
        d["error"] = getattr(g, "error", None)
        (out_dir / f"{g.spec.group_id}.json").write_text(json.dumps(d))
        results.append(d)
    return results


def summarize(results: list[dict]) -> dict:
    """Pilot gate numbers: parse-failure and refusal rates, ring formation, calls per agent-period, tokens."""
    dec = [x for r in results for x in r["decisions"]]
    play = [x for x in dec if x["kind"] not in ("quiz", "recog")]
    periods = [p for r in results for p in r["periods"]]
    agent_periods = 2 * len(periods)
    by_kind = {}
    for k in sorted({x["kind"] for x in dec}):
        xs = [x for x in dec if x["kind"] == k]
        by_kind[k] = {"n": len(xs), "default": sum(x["default"] for x in xs) / len(xs)}
    per_cell = {}
    for r in results:
        c = per_cell.setdefault(f"{r['spec']['cell']}{r['spec']['wording']}",
                                {"pair_periods": 0, "formed": 0, "reports": 0, "ring_live": 0, "price_sum": 0.0, "n_price": 0})
        for p in r["periods"]:
            c["pair_periods"] += 1
            c["formed"] += p["formed"]
            c["reports"] += p["reported"]
            c["ring_live"] += p["ring_live_before"] or p["formed"]
            valid = [x for x, d in zip(p["prices"], p["price_default"]) if not d]
            c["price_sum"] += sum(valid)
            c["n_price"] += len(valid)
    ring_groups = {}
    for r in results:
        key = f"{r['spec']['cell']}{r['spec']['wording']}"
        ring_groups.setdefault(key, []).append(any(p["formed"] for p in r["periods"]))
    n = max(1, len(play))
    return {
        "groups": len(results), "complete": sum(r["complete"] for r in results), "decisions": len(dec),
        "play_calls": len(play), "periods": len(periods),
        "calls_per_agent_period": len(play) / max(1, agent_periods),
        "parse_failure_rate": sum(x["default"] for x in play) / n,
        "refusal_rate": sum(x["refusal"] for x in dec) / max(1, len(dec)),
        "server_errors": sum(1 for x in dec if x.get("m_error")), "by_kind": by_kind,
        "ring_formation_by_cell": {k: {**v, "form_rate": v["formed"] / max(1, v["pair_periods"]),
                                       "mean_price": v["price_sum"] / max(1, v["n_price"])} for k, v in per_cell.items()},
        "groups_with_a_ring": {k: f"{sum(v)}/{len(v)}" for k, v in ring_groups.items()},
        "quiz": quiz_accuracy(results),
        "prompt_tokens": sum(x.get("m_pt", 0) for x in dec), "completion_tokens": sum(x.get("m_ct", 0) for x in dec),
    }


def quiz_accuracy(results: list[dict]) -> dict:
    """Share of correct quiz answers per item id, overall and per (cell, wording) row."""
    per_item, per_row = {}, {}
    for r in results:
        key = f"{r['spec']['cell']}{r['spec']['wording']}"
        for q in r["quiz"]:
            for it, ok in zip(r["quiz_items"], q["correct"]):
                for d, k in ((per_item, it["id"]), (per_row, f"{key}:{it['id']}")):
                    a = d.setdefault(k, [0, 0])
                    a[0] += ok
                    a[1] += 1
    f = lambda d: {k: round(a / n, 3) for k, (a, n) in sorted(d.items())}   # noqa: E731
    return {"by_item": f(per_item), "by_row_item": f(per_row)}


def prefix_hit_rate(metrics_text: str) -> dict:
    """Prefix-cache counters from vLLM /metrics (names differ slightly across versions)."""
    out = {}
    for line in metrics_text.splitlines():
        m = re.match(r"(vllm:prefix_cache_(?:hits|queries)(?:_total)?)\{[^}]*\}\s+([0-9.e+]+)", line)
        if m:
            out[m.group(1).replace("_total", "")] = float(m.group(2))
    q, h = out.get("vllm:prefix_cache_queries"), out.get("vllm:prefix_cache_hits")
    return {**out, "hit_rate": (h / q) if q else None}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="random policy on CPU")
    ap.add_argument("--cells", default="C,F10", help="comma-separated cell names, wording O")
    ap.add_argument("--wording", default="O")
    ap.add_argument("--groups", type=int, default=2)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results/raw/bigoni_smoke"))
    a = ap.parse_args()
    assert a.smoke, "network runs go through src/modal_bigoni.py"
    specs = grid("smoke", "smoke", [(c, a.wording) for c in a.cells.split(",")], a.groups)
    t0 = time.time()
    res = asyncio.run(run_specs(specs, random_policy, Path(a.out), time.time() + 3600))
    s = summarize(res)
    print(json.dumps({k: v for k, v in s.items() if k not in ("by_kind",)}, indent=1))
    print(f"{len(specs)} groups in {time.time() - t0:.1f}s, {GROUP_SIZE} agents each, NO_PRICE={NO_PRICE}")
