"""Environment checks. RunnerConfig refuses to start the experiment if this fails."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OK, FAIL, WARN = "[ ok ]", "[FAIL]", "[warn]"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default=str(ROOT / "profiles" / "server.yaml"))
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.profile).read_text())
    failures = 0

    def check(name, cond, hard=True, detail=""):
        nonlocal failures
        tag = OK if cond else (FAIL if hard else WARN)
        print(f"{tag} {name}" + (f"  {detail}" if detail else ""))
        if not cond and hard:
            failures += 1

    print(f"--- preflight ({cfg['name']}, attribution={cfg.get('attribution')}) ---")

    try:
        import torch
        check("torch + CUDA", torch.cuda.is_available(),
              detail=f"{torch.__version__}, {torch.cuda.device_count()} device(s)")
        n = torch.cuda.device_count()
    except ImportError:
        check("torch installed", False)
        n = 0

    need = {int(cfg["gpus"]["train"]), int(cfg["gpus"]["proposer"])} \
        if str(cfg["gpus"]["train"]).isdigit() else set()
    check("required GPU indices present", all(i < n for i in need),
          detail=f"need {sorted(need)}, have {n}")

    if cfg.get("attribution") == "per_device":
        check("two distinct GPUs for attribution",
              int(cfg["gpus"]["train"]) != int(cfg["gpus"]["proposer"]))

    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
        pynvml.nvmlShutdown()
        check("NVML energy counter readable", True)
    except Exception as e:                                         # noqa: BLE001
        check("NVML energy counter readable", False, detail=str(e))

    if cfg["energy"].get("energibridge"):
        check("energibridge on PATH",
              shutil.which(cfg["energy"]["energibridge_bin"]) is not None)

    data = Path(cfg["workload"]["data_dir"]) / "cifar10_splits.npz"
    check("CIFAR-10 splits prepared", data.exists(), hard=False,
          detail=f"run workload/prepare_cifar.py -> {data}")

    import requests
    for arm, url in cfg["proposer"]["endpoints"].items():
        try:
            r = requests.get(f"{url.rstrip('/')}/models", timeout=5)
            check(f"proposer endpoint '{arm}' reachable", r.status_code == 200,
                  hard=False, detail=url)
        except Exception:                                          # noqa: BLE001
            check(f"proposer endpoint '{arm}' reachable", False, hard=False, detail=url)

    free_gb = shutil.disk_usage(ROOT).free / 1e9
    check("disk space >= 50 GB", free_gb >= 50, hard=False, detail=f"{free_gb:.0f} GB free")

    print(f"--- {'PASS' if failures == 0 else f'{failures} FAILURE(S)'} ---")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
