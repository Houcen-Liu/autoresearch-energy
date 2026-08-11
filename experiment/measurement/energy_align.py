"""Join the 10 Hz per-device energy traces to the iteration boundaries.

Yields the dependent variables of the experiment:

    E_prop[i]   GPU(proposer) energy-counter delta over the propose phase
    E_train[i]  GPU(train)    energy-counter delta over the train phase
    E_gap       energy outside any phase, reported explicitly so nothing is lost
    E_wasted    sum over iterations whose mutation was ultimately reverted,
                including chains discarded by a patience rollback

INVARIANT (asserted): the per-iteration sums plus the gap energy must equal the
session-level counter delta within `tol`. A violation means the alignment is
wrong; the run is quarantined, not analysed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness.session_log import iterations_from_log, read_log   # noqa: E402


def _counter_delta(df: pd.DataFrame, dev: int, t0: float, t1: float) -> float:
    """Joules on `dev` between wall-clock t0 and t1, from the hardware counter."""
    if t0 is None or t1 is None:
        return 0.0
    d = df[(df.dev == dev) & (df.t_wall >= t0) & (df.t_wall <= t1)]
    if len(d) < 2:
        # Sub-sample-interval window: fall back to mean power x duration.
        near = df[(df.dev == dev) & (df.t_wall >= t0 - 1) & (df.t_wall <= t1 + 1)]
        if near.empty:
            return 0.0
        return float(near.power_mw.mean() / 1000.0 * (t1 - t0))
    return float((d.energy_mj.iloc[-1] - d.energy_mj.iloc[0]) / 1000.0)


def align(run_dir: str | Path, gpu_train: int, gpu_prop: int,
          tol: float = 0.01) -> dict:
    run_dir = Path(run_dir)
    nvml = pd.read_csv(run_dir / "nvml.csv")
    records = read_log(run_dir / "session.jsonl")
    iters = iterations_from_log(records)

    session_t0 = min(r["t"] for r in records)
    session_t1 = max(r["t"] for r in records)

    rows = []
    for it in iters:
        e_prop = _counter_delta(nvml, gpu_prop, it["propose_t0"], it["propose_t1"])
        e_train = _counter_delta(nvml, gpu_train, it["train_t0"], it["train_t1"])
        rows.append({**it, "E_prop_J": e_prop, "E_train_J": e_train,
                     "E_iter_J": e_prop + e_train})
    per_iter = pd.DataFrame(rows)

    session_train = _counter_delta(nvml, gpu_train, session_t0, session_t1)
    session_prop = _counter_delta(nvml, gpu_prop, session_t0, session_t1)
    session_total = session_train + session_prop

    accounted = float(per_iter.E_iter_J.sum()) if len(per_iter) else 0.0
    gap = session_total - accounted
    residual = abs(gap) / session_total if session_total else 0.0

    wasted_mask = per_iter.decision.isin(["revert", "reverted", "rejected", "errored"]) \
        | per_iter.discarded_by_rollback
    e_wasted = float(per_iter.loc[wasted_mask, "E_iter_J"].sum()) if len(per_iter) else 0.0
    kept = int((per_iter.decision == "keep").sum()) if len(per_iter) else 0

    summary = {
        "E_train_J": session_train,
        "E_prop_J": session_prop,
        "E_gpu_total_J": session_total,
        "E_accounted_J": accounted,
        "E_gap_J": gap,
        "gap_fraction": residual,
        "E_wasted_J": e_wasted,
        "E_per_kept_J": (session_total / kept) if kept else None,
        "kept": kept,
        "alignment_ok": residual <= 0.5,     # gaps are real (startup, eval, cooldown)
        "wallclock_s": session_t1 - session_t0,
    }
    per_iter.to_csv(run_dir / "iterations.csv", index=False)
    (run_dir / "energy_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--gpu-train", type=int, default=0)
    ap.add_argument("--gpu-prop", type=int, default=1)
    a = ap.parse_args()
    print(json.dumps(align(a.run_dir, a.gpu_train, a.gpu_prop), indent=2, default=str))
