"""One short session end to end. Gates G2 and G3.

  --calibrate   run only the baseline recipe and report whether its validation
                accuracy lands in the 0.60-0.85 headroom band (G2)
  (default)     run a short session and require at least one kept mutation (G3)
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def calibrate(cfg: dict, profile: str) -> int:
    from harness.agent_loop import run_training
    work = Path(tempfile.mkdtemp(prefix="calib_"))
    shutil.copy(ROOT / "workload" / "train.py", work / "train.py")
    shutil.copy(ROOT / "workload" / "prepare_cifar.py", work / "prepare_cifar.py")
    res = run_training(work, work, 0, cfg, "cuda")
    print(json.dumps(res, indent=2))
    if res.get("errored"):
        print("G2 FAILED: baseline does not run")
        return 1
    acc = res["val_acc"]
    ok = 0.60 <= acc <= 0.85
    print(f"\nbaseline val_acc = {acc:.4f} in {res['train_seconds']:.0f}s "
          f"({res['epochs_completed']} epochs)")
    print("G2 " + ("PASSED" if ok else
                   "FAILED -- " + ("weaken" if acc > 0.85 else "strengthen") +
                   " the baseline recipe so the agent has headroom"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default=str(ROOT / "profiles" / "pilot.yaml"))
    ap.add_argument("--proposer", default="dense", choices=["dense", "moe"])
    ap.add_argument("--iterations", type=int, default=5)
    ap.add_argument("--patience", type=int, default=1)
    ap.add_argument("--stub", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--run-dir", default=None)
    a = ap.parse_args()

    cfg = yaml.safe_load(Path(a.profile).read_text())
    if a.calibrate:
        return calibrate(cfg, a.profile)

    run_dir = Path(a.run_dir or tempfile.mkdtemp(prefix="smoke_"))
    cmd = [sys.executable, "-m", "harness.agent_loop",
           "--profile", a.profile, "--proposer", a.proposer,
           "--patience", str(a.patience), "--loop-budget", str(a.iterations),
           "--run-dir", str(run_dir), "--seed", "0"]
    if a.stub:
        cmd.append("--stub")
    print(" ".join(cmd))
    rc = subprocess.run(cmd, cwd=str(ROOT)).returncode
    if rc != 0:
        print("G3 FAILED: session did not complete")
        return 1

    s = json.loads((run_dir / "summary.json").read_text())
    print(json.dumps(s, indent=2))
    ok = s["kept"] >= 1
    print(f"\nrun dir: {run_dir}")
    print("G3 " + ("PASSED" if ok else
                   "FAILED -- no kept mutation in this session. Check the "
                   "rejection rate and the proposer's output contract compliance."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
