"""Modal app for the Bigoni replication: weight download (CPU) and one H100 container per model running vLLM plus the
async group orchestrator (every group of every cell in flight at once).

Launch through this file so the spend guard reserves first (reserve(timeout_s / 3600, label), then `modal run`):

    python src/modal_bigoni.py --download --model qwen3-8b
    python src/modal_bigoni.py --model qwen3-8b --grid pilot --label pilot1 --timeout-s 1000
    python src/modal_bigoni.py --model qwen3-8b --grid main --confirm-main --label main_qwen3 --timeout-s 3600

Outputs go to the Volume `bigoni-out` under /<label> and are pulled to results/raw/bigoni/<label>/.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
VLLM_VERSION = "0.32.0"
PORT = 8000
MODAL_BIN = Path.home() / ".local/share/uv/tools/modal/bin/modal"
TIMEOUT_S = int(os.environ.get("BIGONI_TIMEOUT_S", "1000"))   # hard function timeout = the reserved GPU time
WRAP_UP_S = 60       # groups still running this long before the hard timeout are cancelled and written partial
PROBE_S = 90         # slice of the budget kept for the saturated-throughput probe
DOWNLOAD_RESERVE_H = 0.005   # CPU-only job; reserved at H100 rates anyway so the ledger stays an upper bound

# extra_body goes to every chat request; extra_tokens covers hidden reasoning tokens counted in max_tokens
MODELS = {
    "qwen3-8b": dict(hf="Qwen/Qwen3-8B", extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                     extra_tokens=0, ignore=[]),
    # vLLM 0.32 probe-renders the chat template at startup with `tools` undefined and crashes on Olmo's `tools | tojson`;
    # the patched copy (uploaded to the Volume) tests truthiness instead and renders our system+user chats identically.
    "olmo3-7b": dict(hf="allenai/Olmo-3-7B-Instruct", extra_body={}, extra_tokens=0, ignore=[],
                     chat_template="chat_template_patched.jinja"),
    "gpt-oss-20b": dict(hf="openai/gpt-oss-20b", extra_body={"reasoning_effort": "low"}, extra_tokens=640,
                        ignore=["original/*", "metal/*"]),
}

app = modal.App("bigoni-grid")
models = modal.Volume.from_name("bigoni-models", create_if_missing=True)
outputs = modal.Volume.from_name("bigoni-out", create_if_missing=True)

cpu_image = modal.Image.debian_slim(python_version="3.12").pip_install("huggingface_hub[hf_transfer]").env(
    {"HF_HUB_ENABLE_HF_TRANSFER": "1"})
gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(f"vllm=={VLLM_VERSION}", "openai", "httpx")
    .env({"HF_HUB_OFFLINE": "1"})
    .add_local_file(ROOT / "src/bigoni_engine.py", "/app/src/bigoni_engine.py")
    .add_local_file(ROOT / "src/bigoni_wording.py", "/app/src/bigoni_wording.py")
    .add_local_file(ROOT / "src/bigoni_original.py", "/app/src/bigoni_original.py")
    .add_local_file(ROOT / "src/bigoni_prompts.py", "/app/src/bigoni_prompts.py")
    .add_local_file(ROOT / "src/bigoni_runner.py", "/app/src/bigoni_runner.py")
)


def model_dir(key):
    return f"/models/{key}"


@app.function(image=cpu_image, volumes={"/models": models}, timeout=3600)
def download(key: str):
    """Fetch the weights into the Volume on a CPU box so no H100 time is spent downloading."""
    from huggingface_hub import snapshot_download

    cfg = MODELS[key]
    snapshot_download(cfg["hf"], local_dir=model_dir(key), ignore_patterns=cfg["ignore"] or None)
    models.commit()
    return sum(p.stat().st_size for p in Path(model_dir(key)).rglob("*") if p.is_file()) / 1e9


def _wait_healthy(proc, log, limit_s=600):
    t0 = time.time()
    while time.time() - t0 < limit_s:
        if proc.poll() is not None:
            raise RuntimeError(f"vllm exited {proc.returncode}: {Path(log).read_text()[-3000:]}")
        try:
            if urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3).status == 200:
                return time.time() - t0
        except Exception:  # noqa: BLE001  server not up yet
            pass
        time.sleep(2)
    raise TimeoutError(f"vllm not healthy after {limit_s}s: {Path(log).read_text()[-3000:]}")


def _metrics():
    return urllib.request.urlopen(f"http://127.0.0.1:{PORT}/metrics", timeout=10).read().decode()


@app.function(image=gpu_image, gpu="H100", timeout=TIMEOUT_S, volumes={"/models": models, "/out": outputs})
def run_grid(key: str, plan: dict, label: str):
    """Start vLLM, run the planned groups concurrently against it, write per-group JSON to the Volume."""
    import asyncio

    sys.path.insert(0, "/app/src")
    import bigoni_runner as RN

    t0 = time.time()
    # BIGONI_TIMEOUT_S is set only on the launching machine, so the container takes the limit from the plan
    hard_deadline = t0 + plan.get("timeout_s", TIMEOUT_S) - WRAP_UP_S
    cfg = MODELS[key]
    models.reload()
    out = Path("/out") / label
    out.mkdir(parents=True, exist_ok=True)
    vlog = "/tmp/vllm.log"
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", model_dir(key), "--served-model-name", key,
           "--host", "127.0.0.1", "--port", str(PORT), "--max-model-len", "8192", "--gpu-memory-utilization", "0.90",
           "--max-num-seqs", "512", "--enable-prefix-caching", "--generation-config", "vllm"]
    if cfg.get("chat_template"):
        cmd += ["--chat-template", f"{model_dir(key)}/{cfg['chat_template']}"]
    # The image has no nvcc, and the default FlashInfer sampler JIT-compiles at warmup and crashes without it.
    env = {**os.environ, "VLLM_USE_FLASHINFER_SAMPLER": "0"}
    server = subprocess.Popen(cmd, stdout=open(vlog, "w"), stderr=subprocess.STDOUT, env=env)
    res = {"model": key, "plan": plan}
    try:
        res["health_s"] = _wait_healthy(server, vlog)
        specs = RN.grid(key, plan.get("tag") or label, [tuple(c) for c in plan["cells"]] or None, plan.get("n_groups"))
        if plan.get("only_ids"):   # rerun of groups that did not finish before an earlier deadline
            specs = [sp for sp in specs if sp.group_id in set(plan["only_ids"])]
        policy = RN.LLMPolicy(f"http://127.0.0.1:{PORT}/v1", key, cfg["extra_body"], cfg["extra_tokens"],
                              concurrency=plan.get("concurrency", 512))
        m0 = _metrics()
        t_run = time.time()
        run_deadline = hard_deadline - (PROBE_S if plan.get("probe_n") else 0)
        if plan.get("resume_file"):   # partial group records uploaded to the Volume by the launcher
            outputs.reload()
            policy = RN.ReplayPolicy(policy, json.load(open(Path("/out") / plan["resume_file"])))
        results = asyncio.run(RN.run_specs(specs, policy, out / "groups", run_deadline, recog=not plan.get("quiz_only"),
                                           play=not plan.get("quiz_only")))
        res["run_s"] = time.time() - t_run
        res["summary"] = RN.summarize(results)
        llm = getattr(policy, "inner", policy)
        res["tokens"] = dict(calls=llm.n_calls, prompt=llm.prompt_tokens, cached=llm.cached_tokens,
                             completion=llm.completion_tokens, replayed=getattr(policy, "n_replayed", 0))
        res["tok_s_during_run"] = llm.completion_tokens / max(res["run_s"], 1e-9)
        res["prefix_cache"] = RN.prefix_hit_rate(_metrics())
        res["prefix_cache_before"] = RN.prefix_hit_rate(m0)
        if plan.get("probe_n"):
            res["probe"] = asyncio.run(RN.throughput_probe(llm, plan["probe_n"], hard_deadline))
        res["groups_complete"] = sum(r["complete"] for r in results)
        res["n_groups"] = len(results)
    finally:
        server.terminate()
        try:
            server.wait(30)
        except subprocess.TimeoutExpired:
            server.kill()
        res["elapsed_s"] = time.time() - t0
        res["vllm_tail"] = Path(vlog).read_text()[-2500:]
        (out / "summary.json").write_text(json.dumps(res, default=str))
        (out / "vllm.log").write_text(Path(vlog).read_text())
        outputs.commit()
    return res


@app.local_entrypoint()
def main(model: str = "qwen3-8b", grid: str = "pilot", label: str = "pilot", cells: str = "", n_groups: int = 0,
         probe_n: int = 0, concurrency: int = 512, download_only: bool = False, quiz_only: bool = False,
         tag: str = "", only_ids: str = "", resume_file: str = "",
         timeout_s: int = 1000):
    if os.environ.get("BIGONI_RESERVED") != "1":
        raise SystemExit("run via `python src/modal_bigoni.py ...` so the spend guard reserves first")
    if download_only:
        print(f"{model} on volume: {download.remote(model):.1f} GB")
        return
    plan = dict(grid=grid, quiz_only=quiz_only, timeout_s=timeout_s, tag=tag, resume_file=resume_file,
                only_ids=only_ids.split(",") if only_ids else [], probe_n=probe_n, concurrency=concurrency,
                n_groups=n_groups or None,
                cells=[c.split(":") for c in cells.split(",")] if cells else [])
    res = run_grid.remote(model, plan, label)
    dest = ROOT / "results/raw/bigoni" / label
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(MODAL_BIN), "volume", "get", "bigoni-out", f"/{label}", str(dest.parent), "--force"], check=False)
    print(json.dumps({k: v for k, v in res.items() if k not in ("vllm_tail",)}, indent=1, default=str))
    print("vllm tail:\n" + res["vllm_tail"][-800:])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-8b", choices=list(MODELS))
    ap.add_argument("--download", action="store_true", help="CPU-only weight download into the Volume")
    ap.add_argument("--grid", default="pilot", choices=["pilot", "main"])
    ap.add_argument("--confirm-main", action="store_true", help="required for the full grid")
    ap.add_argument("--cells", default="", help="cell:wording pairs, e.g. C:O,F10:O (default: the grid's own list)")
    ap.add_argument("--n-groups", type=int, default=0)
    ap.add_argument("--label", default="pilot")
    ap.add_argument("--timeout-s", type=int, default=TIMEOUT_S, help="hard function timeout, reserved in full")
    ap.add_argument("--probe-n", type=int, default=0, help="requests replayed at full concurrency after the run")
    ap.add_argument("--concurrency", type=int, default=512)
    ap.add_argument("--tag", default="", help="group-id prefix when it differs from the label (reruns)")
    ap.add_argument("--only-ids", default="", help="comma-separated group ids to run (reruns of unfinished groups)")
    ap.add_argument("--resume-file", default="", help="Volume path (under bigoni-out) of partial group records to resume")
    ap.add_argument("--quiz-only", action="store_true", help="comprehension check: quiz calls only, no game")
    a = ap.parse_args()
    if a.grid == "main" and not a.download and not a.confirm_main:
        raise SystemExit("the main grid needs --confirm-main")
    if a.grid == "pilot" and not a.download and not a.quiz_only and not (a.cells and a.n_groups):
        raise SystemExit("pilot needs --cells and --n-groups (e.g. --cells C:O,F10:O --n-groups 2)")
    env = {**os.environ, "BIGONI_TIMEOUT_S": str(a.timeout_s), "BIGONI_RESERVED": "1"}
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from spend_guard import reserve

    hours = DOWNLOAD_RESERVE_H if a.download else a.timeout_s / 3600
    print(f"reserved ${reserve(hours, ('download-' if a.download else '') + a.label):.2f} for {a.label}")
    flags = ["--model", a.model]
    if a.download:
        flags += ["--download-only"]
    else:
        flags += ["--timeout-s", str(a.timeout_s), "--grid", a.grid, "--label", a.label, "--n-groups", str(a.n_groups), "--probe-n", str(a.probe_n),
                  "--concurrency", str(a.concurrency)]
        if a.quiz_only:
            flags += ["--quiz-only"]
        if a.tag:
            flags += ["--tag", a.tag]
        if a.resume_file:
            flags += ["--resume-file", a.resume_file]
        if a.only_ids:
            flags += ["--only-ids", a.only_ids]
        if a.cells:
            flags += ["--cells", a.cells]
    sys.exit(subprocess.run([str(MODAL_BIN), "run", __file__, *flags], env=env).returncode)
