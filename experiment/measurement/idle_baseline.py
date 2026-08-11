"""Idle-draw characterisation (gate G6).

Measures both GPUs and the CPU package at rest for `--minutes`, so that:
  * standby draw of the non-active GPU can be subtracted if non-trivial,
  * drift across a night of batches can be bounded (<2 % is the gate),
  * Phase-2 CPU proposer energy has a host baseline to subtract.

Run it before the first batch and again after the last one; the two numbers going
into the report as a stability claim.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from nvml_sampler import NvmlSampler


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--out-dir", default="./idle_baseline")
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    s = NvmlSampler(out / "idle_nvml.csv", hz=10)
    s.start()
    t0 = time.time()
    time.sleep(a.minutes * 60)
    totals = s.stop()
    dur = time.time() - t0

    summary = {
        "duration_s": dur,
        "energy_j_per_device": totals["energy_j_per_device"],
        "mean_power_w_per_device": {
            d: j / dur for d, j in totals["energy_j_per_device"].items()},
    }
    (out / "idle_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
