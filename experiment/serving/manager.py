"""Make sure the right proposer is being served before a run starts.

WHY THIS EXISTS. Both arms cannot be resident at once: the dense model is ~15 GB
at int4 and the MoE ~18 GB, on a single 20 GB card. But the Phase-1 run table is
deliberately SHUFFLED, so consecutive runs frequently switch arms.

The alternative -- blocking the run table by proposer so each model is served
once -- would align the factor with run order and hand every thermal or drift
effect straight to the headline comparison (D11). Randomisation is worth more
than the ~2 minutes it costs to swap models, so the swap is automated here.

    python serving/manager.py --arm moe --profile profiles/server.yaml
    python serving/manager.py --stop
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]


def healthy(endpoint: str, model: str | None = None, timeout: float = 3) -> bool:
    """True when the endpoint answers and, if asked, serves the named model."""
    try:
        r = requests.get(f"{endpoint.rstrip('/')}/models", timeout=timeout)
        if r.status_code != 200:
            return False
        if model is None:
            return True
        served = [m.get("id") for m in r.json().get("data", [])]
        return model in served or any(model in str(s) for s in served)
    except requests.RequestException:
        return False


def stop_all(pattern: str = "vllm serve") -> None:
    subprocess.run(["pkill", "-f", pattern], check=False)
    for _ in range(30):
        if subprocess.run(["pgrep", "-f", pattern],
                          capture_output=True).returncode != 0:
            return
        time.sleep(1)
    subprocess.run(["pkill", "-9", "-f", pattern], check=False)
    time.sleep(2)


def ensure(arm: str, cfg: dict, wait_s: float = 900) -> str:
    """Serve `arm` on the proposer GPU, swapping models if necessary.

    Returns the endpoint. Raises if the model does not become ready in time.
    """
    pcfg = cfg["proposer"]
    endpoint = pcfg["endpoints"][arm]
    served_name = pcfg.get("model_names", {}).get(arm, arm)

    if healthy(endpoint, served_name):
        return endpoint

    print(f"[serving] swapping to '{arm}' ...", flush=True)
    stop_all()

    port = endpoint.rstrip("/").rsplit(":", 1)[-1].split("/")[0]
    env = dict(os.environ, PROPOSER_GPU=str(cfg["gpus"]["proposer"]))
    log = Path(cfg.get("results_dir", ".")) / f"vllm_{arm}.log"
    log.parent.mkdir(parents=True, exist_ok=True)

    with log.open("a") as lf:
        subprocess.Popen(["bash", str(ROOT / "serving" / "serve_vllm.sh"), arm, port],
                         env=env, stdout=lf, stderr=subprocess.STDOUT,
                         start_new_session=True)

    t0 = time.time()
    while time.time() - t0 < wait_s:
        if healthy(endpoint, served_name):
            print(f"[serving] '{arm}' ready after {time.time() - t0:.0f}s", flush=True)
            return endpoint
        time.sleep(5)

    raise RuntimeError(
        f"'{arm}' did not become ready within {wait_s:.0f}s. Check {log}. "
        f"A model that never loads is usually OOM: see gate G1 and D3.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default=str(ROOT / "profiles" / "server.yaml"))
    ap.add_argument("--arm", choices=["dense", "moe"])
    ap.add_argument("--stop", action="store_true")
    ap.add_argument("--wait", type=float, default=900)
    a = ap.parse_args()

    if a.stop:
        stop_all()
        print("[serving] stopped")
        return 0
    if not a.arm:
        ap.error("--arm is required unless --stop")

    cfg = yaml.safe_load(Path(a.profile).read_text())
    print(ensure(a.arm, cfg, a.wait))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
