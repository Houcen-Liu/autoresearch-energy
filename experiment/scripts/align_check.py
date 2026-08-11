"""Gates G4 and G5 -- the energy/iteration alignment proof.

G4: per-iteration energy + gap energy must reconstruct the session total.
G5: during a training phase, the training GPU must draw substantially more than
    the proposer GPU, and vice versa during a proposal phase. If this fails, the
    device indices are swapped somewhere and every attribution in the study is
    wrong.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.session_log import iterations_from_log, read_log      # noqa: E402
from measurement.energy_align import align                         # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--gpu-train", type=int, default=0)
    ap.add_argument("--gpu-prop", type=int, default=1)
    a = ap.parse_args()

    rd = Path(a.run_dir)
    summary = align(rd, a.gpu_train, a.gpu_prop)
    print(json.dumps(summary, indent=2, default=str))

    g4 = summary["gap_fraction"] <= 0.5
    print(f"\nG4 accounted={summary['E_accounted_J']:.0f} J  "
          f"gap={summary['E_gap_J']:.0f} J ({summary['gap_fraction']:.1%})  "
          + ("PASSED" if g4 else "FAILED"))

    nvml = pd.read_csv(rd / "nvml.csv")
    iters = iterations_from_log(read_log(rd / "session.jsonl"))
    checks = []
    for it in iters:
        for phase, lo, hi, hot, cold in (
                ("train", it["train_t0"], it["train_t1"], a.gpu_train, a.gpu_prop),
                ("propose", it["propose_t0"], it["propose_t1"], a.gpu_prop, a.gpu_train)):
            if lo is None or hi is None:
                continue
            w = nvml[(nvml.t_wall >= lo) & (nvml.t_wall <= hi)]
            if w.empty:
                continue
            p_hot = w[w.dev == hot].power_mw.mean()
            p_cold = w[w.dev == cold].power_mw.mean()
            checks.append({"iter": it["iter"], "phase": phase,
                           "hot_W": p_hot / 1000, "cold_W": p_cold / 1000,
                           "ok": bool(p_hot > 1.5 * p_cold)})
    df = pd.DataFrame(checks)
    if len(df):
        print("\n" + df.groupby("phase")[["hot_W", "cold_W", "ok"]].mean().to_string())
    g5 = bool(len(df)) and df.ok.mean() > 0.9
    print("G5 " + ("PASSED" if g5 else "FAILED -- device attribution looks wrong"))
    return 0 if (g4 and g5) else 1


if __name__ == "__main__":
    raise SystemExit(main())
