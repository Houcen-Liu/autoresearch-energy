"""Gate G1 -- the highest-risk gate in the build.

Qwen3.6-35B-A3B at int4 is ~18 GB of weights on a 20 GB card. This script drives
the already-running serving endpoint with 40 requests of the REAL prompt shape
(a full program.md plus the current train.py, ~4-6k prompt tokens) and reports
whether it survives, how fast it is, and how close to the memory ceiling it runs.

Run this on DAY 2 of week 1. If it fails, the model choice changes, and that
decision needs to happen before anything else is built on top of it.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def gpu_mem(dev: int) -> tuple[float, float]:
    try:
        import pynvml
        pynvml.nvmlInit()
        info = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(dev))
        pynvml.nvmlShutdown()
        return info.used / 1e9, info.total / 1e9
    except Exception:                                              # noqa: BLE001
        return (float("nan"), float("nan"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--out", default="vram_check.json")
    a = ap.parse_args()

    # Real prompt shape: program.md template + a full train.py.
    program = (ROOT / "harness" / "templates" / "program.md.j2").read_text()
    train = (ROOT / "workload" / "train.py").read_text()
    system = (ROOT / "harness" / "templates" / "system.txt").read_text()
    user = program.replace("{{ current_source }}", train) + "\n" + train

    lats, fails, peak_used = [], 0, 0.0
    for i in range(a.n):
        t0 = time.time()
        try:
            r = requests.post(f"{a.endpoint.rstrip('/')}/chat/completions",
                              json={"model": a.model, "temperature": 0,
                                    "max_tokens": a.max_tokens,
                                    "messages": [{"role": "system", "content": system},
                                                 {"role": "user", "content": user}]},
                              timeout=600)
            r.raise_for_status()
            body = r.json()
            lats.append(time.time() - t0)
            used, total = gpu_mem(a.gpu)
            peak_used = max(peak_used, used)
            print(f"{i+1:3d}/{a.n}  {lats[-1]:6.1f}s  "
                  f"{body.get('usage',{}).get('completion_tokens','?')} tok  "
                  f"VRAM {used:.1f}/{total:.1f} GB")
        except Exception as e:                                     # noqa: BLE001
            fails += 1
            print(f"{i+1:3d}/{a.n}  FAILED: {e}")

    used, total = gpu_mem(a.gpu)
    result = {
        "model": a.model, "requests": a.n, "failures": fails,
        "latency_p50_s": statistics.median(lats) if lats else None,
        "latency_p95_s": (sorted(lats)[int(0.95 * len(lats)) - 1] if len(lats) > 3 else None),
        "peak_vram_gb": peak_used, "total_vram_gb": total,
        "headroom_gb": total - peak_used if total == total else None,
        "PASS": fails == 0,
    }
    Path(a.out).write_text(json.dumps(result, indent=2))
    print("\n" + json.dumps(result, indent=2))
    print("\nG1 " + ("PASSED" if result["PASS"] else
                     "FAILED -- fall back to llama.cpp GGUF (both arms) or the "
                     "Granite backup pair; see EXPERIMENT_PLAN.md D3"))
    return 0 if result["PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
