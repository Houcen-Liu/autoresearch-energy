"""Walk the experiment output tree and produce the two analysis tables.

    tidy.csv        one row per session  (the unit of analysis)
    iterations.csv  one row per iteration (mechanistic substrate)

Validation is strict on purpose: a run missing session.jsonl, or whose energy
alignment failed, is written to quarantine.csv with the reason rather than
silently entering the analysis.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness.session_log import iterations_from_log, read_log     # noqa: E402

# Cell keys are whichever of these the experiment actually varied. Hardcoding
# Phase 1's three factors made the Stage-2a summary pool across `thinking`,
# reporting a table that looked plausible and answered the wrong question.
# Anything present in the run-table row is treated as a factor.
ALL_FACTORS = ("proposer", "patience", "loop_budget",
               # the run table names it `thinking`; the session summary records
               # `thinking_requested`. Accept both, since tidy.csv is built from
               # the summary and would otherwise silently pool across the factor.
               "thinking", "thinking_requested", "temperature")
CELL_KEYS = ALL_FACTORS


def cell_keys_for(cell: dict) -> tuple:
    """Factors this experiment varied, in a stable order."""
    return tuple(k for k in ALL_FACTORS if k in cell)


def _cell_from_log(records: list[dict]) -> dict:
    for r in records:
        if r.get("ev") == "session_start":
            return {**r.get("cell", {}), "seed": r.get("seed"),
                    "attribution": r.get("attribution"),
                    "stub": bool(r.get("stub", False)),
                    "synthetic_data": bool(r.get("synthetic_data", False))}
    return {}


def _is_scaffolding(records: list[dict], cell: dict) -> str:
    """Runs that are not evidence about anything.

    A stub-proposer session replays scripted mutations; a synthetic-data session
    scores ~0.10 by construction. Averaged into a cell they silently corrupt it,
    which is exactly what happened on the pilot before this check existed.
    """
    if cell.get("stub"):
        return "stub proposer"
    if cell.get("synthetic_data"):
        return "synthetic data"
    if not any(r.get("ev") == "proposer_config" for r in records):
        return "no proposer_config recorded (pre-instrumentation run)"
    return ""


def collect(experiments_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = Path(experiments_dir)
    sessions, iters, quarantine = [], [], []

    # Any directory holding a session.jsonl is a run. experiment-runner names them
    # run_<n>_repetition_<r>; the pilot names them differently, and both must work.
    run_dirs = sorted({p.parent for p in root.rglob("session.jsonl")})
    if not run_dirs:
        print(f"[aggregate] no sessions found under {root}")

    for run_dir in run_dirs:
        log_path = run_dir / "session.jsonl"
        records = read_log(log_path)
        cell = _cell_from_log(records)

        why = _is_scaffolding(records, cell)
        if why:
            quarantine.append({"run": run_dir.name, "reason": why})
            continue

        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            quarantine.append({"run": run_dir.name, "reason": "session did not finish"})
            continue
        summary = json.loads(summary_path.read_text())
        row = {"run": run_dir.name, **cell, **summary}

        # A session whose infrastructure misbehaved is not evidence about the
        # proposer. Quarantine it and re-run it; never let it enter the analysis
        # as a poor result for its cell.
        if summary.get("valid") is False:
            quarantine.append({"run": run_dir.name,
                               "reason": summary.get("invalid_reason", "invalid"),
                               "infra_error_rate": summary.get("infra_error_rate")})
            continue

        e_path = run_dir / "energy_summary.json"
        if e_path.exists():
            e = json.loads(e_path.read_text())
            row.update({k: e.get(k) for k in
                        ("E_train_J", "E_prop_J", "E_gpu_total_J", "E_wasted_J",
                         "E_per_kept_J", "gap_fraction", "alignment_ok", "wallclock_s")})
            if e.get("alignment_ok") is False:
                quarantine.append({"run": run_dir.name, "reason": "energy alignment failed",
                                   "gap_fraction": e.get("gap_fraction")})

        it = pd.DataFrame(iterations_from_log(records))
        if (run_dir / "iterations.csv").exists():
            it = pd.read_csv(run_dir / "iterations.csv")
        for k in cell_keys_for(cell):
            it[k] = cell.get(k)
        it["run"] = run_dir.name
        iters.append(it)

        if len(it):
            row["prompt_tokens"] = int(it.prompt_tokens.fillna(0).sum())
            row["completion_tokens"] = int(it.completion_tokens.fillna(0).sum())
            row["proposer_latency_s_mean"] = float(it.proposer_latency_s.mean(skipna=True))
            first_keep = it.index[it.decision == "keep"]
            row["iters_to_first_keep"] = int(it.loc[first_keep[0], "iter"]) if len(first_keep) else None
        sessions.append(row)

    iters = [d for d in iters if len(d)]     # avoid the all-NA concat warning
    return (pd.DataFrame(sessions),
            pd.concat(iters, ignore_index=True) if iters else pd.DataFrame(),
            pd.DataFrame(quarantine))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments-dir", required=True)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()

    out = Path(a.out_dir or a.experiments_dir)
    tidy, iters, quar = collect(a.experiments_dir)
    out.mkdir(parents=True, exist_ok=True)
    tidy.to_csv(out / "tidy.csv", index=False)
    iters.to_csv(out / "iterations.csv", index=False)
    quar.to_csv(out / "quarantine.csv", index=False)

    print(f"sessions: {len(tidy)}   iterations: {len(iters)}   quarantined: {len(quar)}")
    if len(tidy):
        # Energy columns only exist when the profile attributes per device. The
        # pilot runs with attribution=none, so aggregate on what is present.
        aggs = {"n": ("run", "count")}
        for label, col in (("E", "E_gpu_total_J"), ("acc", "test_acc"),
                           ("kept", "kept")):
            if col in tidy.columns:
                aggs[label] = (col, "mean" if col != "run" else "count")
        keys = [k for k in ALL_FACTORS if k in tidy.columns
                and tidy[k].nunique(dropna=True) > 1]
        if not keys:
            keys = [k for k in ALL_FACTORS if k in tidy.columns]
        if keys:
            print(tidy.groupby(keys, dropna=False).agg(**aggs).to_string())
        else:
            print(tidy[[c for c in ("run", "test_acc", "kept") if c in tidy.columns]]
                  .to_string(index=False))
    if len(quar):
        print("\nQUARANTINED:\n" + quar.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
