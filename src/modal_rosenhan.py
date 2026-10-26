"""Modal app for the planted-infiltrator study: one H100 container serving vLLM plus the Liar's Bar games.

Launch through this file so the spend guard runs first (it reserves gpu_hours >= the function timeout, then runs
`modal run`):

    python src/modal_rosenhan.py --label smoke_rosenhan --timeout-s 1200 --jobs G0:0,G0:1,G1:0,G1:1,G2:0,G2:1,G3:0,G3:1
    python src/modal_rosenhan.py --label main_rosenhan --timeout-s 3600 --jobs G2:0,G2:1,...

Jobs are `cell:seed` pairs run concurrently as subprocesses of src/exp_rosenhan.py against the container's
localhost vLLM. The weights come from the Volume `offcamera-models` (fetch once with `python src/modal_rosenhan.py
--download`). Outputs go to the Volume `rosenhan-out` and are returned (gzipped, one game is about 1 MB of raw calls)
to results/rosenhan/{raw,smoke}/.
"""

import argparse
import gzip
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
MODEL_DIR = "/models/qwen3-30b-a3b-instruct-2507-fp8"
VLLM_VERSION = "0.32.0"   # pinned/current; starter/util/poseidon_compute/modal_serve.py pins 0.6.6, too old for Qwen3
PORT = 8000
MODAL_BIN = shutil.which("modal") or Path.home() / ".local/share/uv/tools/modal/bin/modal"
# Hard function timeout in seconds; the launcher sets it and reserves at least this many GPU-hours first.
TIMEOUT_S = int(os.environ.get("ROSENHAN_TIMEOUT_S", "780"))
WRAP_UP_S = 90   # the runner stops jobs this long before the hard timeout so outputs can still be committed

app = modal.App("rosenhan-qwen3")
models = modal.Volume.from_name("offcamera-models", create_if_missing=True)
outputs = modal.Volume.from_name("rosenhan-out", create_if_missing=True)

cpu_image = modal.Image.debian_slim(python_version="3.12").pip_install("huggingface_hub[hf_transfer]").env(
    {"HF_HUB_ENABLE_HF_TRANSFER": "1"})
gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    # The paper's runs reused a cached build of this layer; a fresh resolve on 2026-10-24 pulled torch 2.13 and
    # vLLM failed to start, so a new build may need torch pinned to the version vLLM 0.32.0 was built against.
    .pip_install(f"vllm=={VLLM_VERSION}", "openai", "jinja2", "backoff", "pydantic", "python-dotenv", "numpy", "tqdm")
    # debian_slim has no nvcc, so JIT kernels fail at engine warmup (seen 2026-10-24): DeepGEMM, and the FlashInfer
    # top-k/top-p sampler, which was the fatal one. The non-JIT paths (Triton FP8 MoE, PyTorch sampler) need none.
    .env({"HF_HUB_OFFLINE": "1", "VLLM_USE_DEEP_GEMM": "0", "VLLM_USE_FLASHINFER_SAMPLER": "0"})
    # The engine reads prompts/ relative to its cwd; exp_rosenhan.py symlinks this dir into a scratch cwd per game.
    .add_local_dir(ROOT / "starter/llm-agent-collusion-2/Collusion_Liars_Bar/src",
                   "/app/starter/llm-agent-collusion-2/Collusion_Liars_Bar/src", ignore=["**/__pycache__"])
    .add_local_dir(ROOT / "starter/llm-agent-collusion-2/Collusion_Liars_Bar/prompts",
                   "/app/starter/llm-agent-collusion-2/Collusion_Liars_Bar/prompts")
    .add_local_dir(ROOT / "starter/util/poseidon_compute", "/app/starter/util/poseidon_compute",
                   ignore=["**/__pycache__"])
    .add_local_file(ROOT / "src/rosenhan_prompts.py", "/app/src/rosenhan_prompts.py")
    .add_local_file(ROOT / "src/rosenhan_stub.py", "/app/src/rosenhan_stub.py")
    .add_local_file(ROOT / "src/exp_rosenhan.py", "/app/src/exp_rosenhan.py")
    .add_local_file(ROOT / "src/rate_rosenhan.py", "/app/src/rate_rosenhan.py")
)


def model_dir(model):
    return MODEL_DIR if model == MODEL else "/models/" + model.split("/")[-1].lower()


@app.function(image=cpu_image, volumes={"/models": models}, timeout=3600)
def download(model=MODEL):
    """Fetch the weights into the Volume on a CPU box so no H100 time is spent downloading."""
    from huggingface_hub import snapshot_download

    # gpt-oss repos also carry original/ and metal/ copies of the weights; vLLM reads only the top-level safetensors.
    snapshot_download(model, local_dir=model_dir(model), ignore_patterns=["original/*", "metal/*"])
    models.commit()
    return sum(p.stat().st_size for p in Path(model_dir(model)).rglob("*") if p.is_file()) / 1e9


def _wait_healthy(proc, log, limit_s=840):
    t0 = time.time()
    while time.time() - t0 < limit_s:
        if proc.poll() is not None:
            # The tail is only the engine-startup traceback; the root cause sits earlier among the error lines.
            text = Path(log).read_text()
            Path("/out/vllm_fail.log").write_text(text)   # full log on the volume; the error lines drop the frames
            outputs.commit()
            errs = [ln for ln in text.splitlines() if "Error" in ln or "error" in ln or 'File "' in ln][:60]
            raise RuntimeError(f"vllm exited {proc.returncode}:\n" + "\n".join(errs) + "\n--- tail\n" + text[-1500:])
        try:
            if urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3).status == 200:
                return time.time() - t0
        except Exception:  # noqa: BLE001  server not up yet
            pass
        time.sleep(3)
    raise TimeoutError(f"vllm not healthy after {limit_s}s: {Path(log).read_text()[-3000:]}")


@app.function(image=gpu_image, gpu="H100", timeout=TIMEOUT_S, volumes={"/models": models, "/out": outputs})
def run_jobs(jobs, label, concurrency=16, extra=(), vllm_args=(), script="exp_rosenhan.py", timeout_s=TIMEOUT_S,
             model=MODEL):
    """Start vLLM, run every [cell, seed] job against it concurrently, return the raw JSONs, then exit."""
    t0 = time.time()
    # timeout_s comes from the caller: the container never sees ROSENHAN_TIMEOUT_S, so TIMEOUT_S there is the default.
    deadline = t0 + timeout_s - WRAP_UP_S
    models.reload()
    out = Path("/out") / label
    (out / "raw").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    vlog = "/tmp/vllm.log"
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", model_dir(model),
           "--served-model-name", model, "--host", "127.0.0.1", "--port", str(PORT), "--max-model-len", "32768",
           "--gpu-memory-utilization", "0.92", "--max-num-seqs", "64", *vllm_args]
    server = subprocess.Popen(cmd, stdout=open(vlog, "w"), stderr=subprocess.STDOUT)
    status, running, pending = {}, {}, [tuple(j) for j in jobs]
    env = {**os.environ, "LLM_BASE_URL": f"http://127.0.0.1:{PORT}/v1", "LLM_API_KEY": "EMPTY", "LLM_MODEL": model}
    try:
        health_s = _wait_healthy(server, vlog)
        while (pending or running) and time.time() < deadline:
            while pending and len(running) < concurrency:
                cell, seed = pending.pop(0)
                key = f"{cell}_{seed}"
                cmd = [sys.executable, f"/app/src/{script}", "--cell", cell, "--seed", str(seed),
                       "--out", str(out / "raw"), *extra]
                running[key] = subprocess.Popen(cmd, env=env, stdout=open(out / "logs" / f"{key}.log", "w"),
                                                stderr=subprocess.STDOUT)
            for key, p in list(running.items()):
                if p.poll() is not None:
                    status[key] = p.returncode
                    del running[key]
            time.sleep(2)
        for key, p in running.items():   # past the soft deadline: stop, never leave GPU time running
            p.terminate()
            status[key] = "terminated"
        for cell, seed in pending:
            status[f"{cell}_{seed}"] = "not started"
    finally:
        server.terminate()
        try:
            server.wait(30)
        except subprocess.TimeoutExpired:
            server.kill()
        outputs.commit()
    results = {f.name: gzip.compress(f.read_bytes()) for f in (out / "raw").glob("*.json")}
    logs = {f.name: f.read_text()[-1500:] for f in (out / "logs").glob("*.log")}
    return dict(results=results, status=status, logs=logs, health_s=health_s, elapsed_s=time.time() - t0,
                vllm_tail=Path(vlog).read_text()[-1500:])


@app.local_entrypoint()
def main(jobs: str = "", label: str = "run", concurrency: int = 16, extra: str = "", vllm_args: str = "",
         download_only: bool = False, script: str = "exp_rosenhan.py", model: str = MODEL):
    if os.environ.get("ROSENHAN_RESERVED") != "1" and not download_only:
        raise SystemExit("run via `python src/modal_rosenhan.py ...` so the spend guard reserves first")
    if download_only:
        print(f"model on volume: {download.remote(model):.1f} GB")
        return
    pairs = [[c, int(s)] for c, s in (j.split(":") for j in jobs.split(","))]
    res = run_jobs.remote(pairs, label, concurrency, extra.split(), vllm_args.split(), script, TIMEOUT_S, model)
    dest = ROOT / "results/rosenhan" / label
    dest.mkdir(parents=True, exist_ok=True)
    for name, blob in res["results"].items():
        (dest / name).write_bytes(gzip.decompress(blob))
    print(f"vllm healthy after {res['health_s']:.0f}s, container {res['elapsed_s']:.0f}s, status {res['status']}")
    for name, tail in res["logs"].items():
        print(f"--- {name}\n{tail[-600:]}")
    print(f"wrote {len(res['results'])} files to {dest}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true", help="CPU-only weight download into the Volume")
    ap.add_argument("--jobs", default="", help="comma-separated cell:seed")
    ap.add_argument("--label", default="run")
    ap.add_argument("--timeout-s", type=int, default=TIMEOUT_S, help="hard function timeout, reserved in full")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--extra", default="", help="extra args for exp_rosenhan.py, e.g. '--max-rounds 20'")
    ap.add_argument("--vllm-args", default="", help="extra vllm server args, e.g. '--enforce-eager'")
    ap.add_argument("--script", default="exp_rosenhan.py", help="job script: exp_rosenhan.py (games) or rate_rosenhan.py")
    ap.add_argument("--model", default=MODEL, help="HF repo served by vLLM, e.g. a second overseer model")
    a = ap.parse_args()
    env = {**os.environ, "ROSENHAN_TIMEOUT_S": str(a.timeout_s), "ROSENHAN_RESERVED": "1"}
    if not a.download:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from spend_guard import reserve

        print(f"reserved ${reserve(a.timeout_s / 3600, a.label):.2f} for {a.label}")
    flags = (["--download-only"] if a.download else
             ["--jobs", a.jobs, "--label", a.label, "--concurrency", str(a.concurrency), "--script", a.script])
    flags += ["--model", a.model]
    if a.extra:
        flags += ["--extra", a.extra]
    if a.vllm_args:
        flags += ["--vllm-args", a.vllm_args]
    sys.exit(subprocess.run([str(MODAL_BIN), "run", __file__, *flags], env=env).returncode)
