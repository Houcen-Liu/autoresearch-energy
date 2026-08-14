"""Exploratory long-horizon session: how far can the loop actually get?

Phase 1 answered a comparative question over 10-20 iterations. This asks a
different, descriptive one: given many more attempts, does the agent keep
improving, plateau, or start winning on validation without winning on test?

Phase 1 established that the loop can improve the baseline, but capped every
session at 20 iterations. Three long-horizon outcomes are worth distinguishing,
and they imply different things about the approach:

  1. it plateaus below the reference -> the loop has a ceiling, and that ceiling
     bounds the value of agentic AutoML on this workload;
  2. it keeps climbing -> the short-budget comparison is about search
     EFFICIENCY, not diminishing returns, which is a different claim;
  3. validation accuracy rises while test accuracy does not -> the loop is
     overfitting its own selection metric. Phase 1 shows no sign of this
     (val-test gap flat at ~0.87 pp, r = -0.29 against kept count) but the most
     any session kept was 4, so there has been no real selection pressure.

This is n = 1 and descriptive. Report it as a trajectory, not as statistics.

    python scripts/long_horizon.py --proposer moe --iterations 100
"""
from __future__ import annotations

import argparse
import json
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
    ap.add_argument("--max-tokens", type=int, default=8192,
                    help="completion allowance; 8192 leaves half of the 16384-token "
                         "server context for the prompt")
    ap.add_argument("--history-max-rows", type=int, default=20,
                    help="recent history rows included in each prompt; the full "
                         "history is still recorded in session.jsonl")
    a = ap.parse_args()

    run_dir = Path(a.run_dir or (ROOT.parent / "experiments" / "long_horizon" /
                                 f"{a.proposer}_{a.iterations}it_{int(time.time())}"))
    existing_session_artifacts = [
        run_dir / name for name in ("session.jsonl", "summary.json")
        if (run_dir / name).exists()
    ]
    if existing_session_artifacts:
        names = ", ".join(p.name for p in existing_session_artifacts)
        print(f"[FAIL] refusing to reuse run directory {run_dir}: found {names}")
        print("choose a new --run-dir so session logs are never appended or overwritten")
        return 2
    run_dir.mkdir(parents=True, exist_ok=True)

    # ~77 s/iteration for the MoE and ~112 s for dense, measured in Phase 1.
    # measured in Phase 1: ~32 s/proposal (MoE), ~62 s (dense), + 45 s training
    est = a.iterations * ((32 if a.proposer == "moe" else 62) + 50)
    print(f"long-horizon session: {a.proposer}, {a.iterations} iterations, "
          f"patience {a.patience}")
    print(f"prompt safety: max_tokens={a.max_tokens}, "
          f"history_max_rows={a.history_max_rows}")
    print(f"estimated wall-clock ~{est/3600:.1f} h  ->  {run_dir}")
    print("serve the arm first; this script does not swap models.\n")

    cmd = [sys.executable, "-m", "harness.agent_loop",
           "--profile", a.profile,
           "--proposer", a.proposer,
           "--patience", str(a.patience),
           "--loop-budget", str(a.iterations),
           "--run-dir", str(run_dir),
           "--seed", str(a.seed),
           "--max-tokens", str(a.max_tokens),
           "--history-max-rows", str(a.history_max_rows)]
    if a.thinking:
        cmd += ["--thinking", a.thinking]

    rc = subprocess.run(cmd, cwd=str(ROOT)).returncode
    if rc == 0:
        summary_path = run_dir / "summary.json"
        try:
            summary = json.loads(summary_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"\n[FAIL] session exited 0 but no readable summary exists: {exc}")
            return 3
        if not summary.get("valid", False):
            print(f"\n[FAIL] session completed but is scientifically invalid: "
                  f"{summary.get('invalid_reason', 'no reason recorded')}")
            return 3

    print(f"\nsession exited {rc}; artifacts in {run_dir}")
    print("next:")
    print(f"  python analysis/trajectory.py --run-dir {run_dir}")
    print(f"  python scripts/replay_keeps.py --run-dir {run_dir}   # val-test gap")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
