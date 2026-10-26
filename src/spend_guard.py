"""Hard cap (CAP_USD) on Modal spend for this sprint.

Modal budgets can only be set from the dashboard, so this is the in-code backstop. Billing data
lags by minutes, so the guard takes the larger of Modal's own number and a local ledger of GPU
seconds this project started, and refuses any job whose worst-case cost would cross the cap.
Usage: python spend_guard.py            -> prints spend so far
       python spend_guard.py 1.5        -> exits 1 if a job of up to 1.5 GPU-hours would cross the cap
"""

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

CAP_USD = 18.0  # +$8 authorised 2026-10-24 for the Rosenhan study
H100_USD_PER_S = 0.001097  # modal.com/pricing, Oct 2026
MODAL = shutil.which("modal") or Path.home() / ".local/share/uv/tools/modal/bin/modal"
LEDGER = Path(__file__).resolve().parent.parent / "results/spend_ledger.jsonl"


def modal_spend():
    out = subprocess.run([str(MODAL), "billing", "summary", "--json"], capture_output=True, text=True, check=True)
    s = json.loads(out.stdout)
    # breakdown is pre-allowance usage; metered_cost nets out the egress allowance
    return max(float(s["metered_cost"]), sum(float(v) for v in s["metered_cost_breakdown"].values()))


def ledger_spend():
    if not LEDGER.exists():
        return 0.0
    return sum(json.loads(line)["usd"] for line in LEDGER.read_text().splitlines() if line.strip())


def spent():
    return max(modal_spend(), ledger_spend())


def reserve(gpu_hours, label):
    """Record a job's worst-case cost before it starts; raises if it would cross the cap."""
    usd = gpu_hours * 3600 * H100_USD_PER_S
    now = spent()
    if now + usd > CAP_USD:
        raise SystemExit(f"spend guard: ${now:.2f} spent + ${usd:.2f} for {label} > ${CAP_USD:.0f} cap")
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a") as f:
        f.write(json.dumps({"t": time.time(), "label": label, "gpu_hours": gpu_hours, "usd": usd}) + "\n")
    return usd


if __name__ == "__main__":
    if len(sys.argv) > 1:
        h = float(sys.argv[1])
        need = h * 3600 * H100_USD_PER_S
        ok = spent() + need <= CAP_USD
        print(f"spent ${spent():.2f}, job ${need:.2f}, cap ${CAP_USD:.0f}: {'ok' if ok else 'BLOCKED'}")
        sys.exit(0 if ok else 1)
    print(f"modal ${modal_spend():.4f} · ledger ${ledger_spend():.2f} · cap ${CAP_USD:.0f}")
