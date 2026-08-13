"""Exploratory long-horizon session: how far can the loop actually get?

Phase 1 answered a comparative question over 10-20 iterations. This asks a
different, descriptive one: given many more attempts, does the agent keep
improving, plateau, or start winning on validation without winning on test?

Context from Phase 1 that makes the question well-posed:

    baseline                                  0.7620
    best test accuracy seen in 24 sessions    0.8645
    hand-tuned reference (never shown)        0.9161

i.e. the agent already recovers ~67 % of the available headroom within 20
iterations. Three outcomes are worth distinguishing, and they imply different
things about the approach:

  1. it plateaus below the reference -> the loop has a ceiling, and that ceiling
     bounds the value of agentic AutoML on this workload;
  2. it keeps climbing -> the loop_budget result (+107.6 % energy per kept
     mutation from 10 to 20) is about diminishing EFFICIENCY, not diminishing
     returns, which is a different claim;
  3. validation accuracy rises while test accuracy does not -> the loop is
     overfitting its own selection metric. Phase 1 shows no sign of this
     (val-test gap flat at ~0.87 pp, r = -0.29 against kept count) but the most
     any session kept was 4, so there has been no real selection pressure.

This is n = 1 and descriptive. Report it as a trajectory, not as statistics.

    python scripts/long_horizon.py --proposer moe --iterations 100
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default=str(ROOT / "profiles" / "server.yaml"))
    ap.add_argument("--proposer", choices=["dense", "moe"], default="moe")
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--patience", type=int, default=1,
                    help="greedy by default: every non-improvement returns to "
                         "the best recipe, which is the right setting for a "
                         "'how high can it climb' question")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--thinking", choices=["on", "off"], default=None)
    a = ap.parse_args()

    run_dir = Path(a.run_dir or (ROOT.parent / "experiments" / "long_horizon" /
                                 f"{a.proposer}_{a.iterations}it_{int(time.time())}"))
    run_dir.mkdir(parents=True, exist_ok=True)

    # ~77 s/iteration for the MoE and ~112 s for dense, measured in Phase 1.
    # measured in Phase 1: ~32 s/proposal (MoE), ~62 s (dense), + 45 s training
    est = a.iterations * ((32 if a.proposer == "moe" else 62) + 50)
    print(f"long-horizon session: {a.proposer}, {a.iterations} iterations, "
          f"patience {a.patience}")
    print(f"estimated wall-clock ~{est/3600:.1f} h  ->  {run_dir}")
    print("serve the arm first; this script does not swap models.\n")

    cmd = [sys.executable, "-m", "harness.agent_loop",
           "--profile", a.profile,
           "--proposer", a.proposer,
           "--patience", str(a.patience),
           "--loop-budget", str(a.iterations),
           "--run-dir", str(run_dir),
           "--seed", str(a.seed)]
    if a.thinking:
        cmd += ["--thinking", a.thinking]

    rc = subprocess.run(cmd, cwd=str(ROOT)).returncode
    print(f"\nsession exited {rc}; artifacts in {run_dir}")
    print("next:")
    print(f"  python analysis/trajectory.py --run-dir {run_dir}")
    print(f"  python scripts/replay_keeps.py --run-dir {run_dir}   # val-test gap")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
