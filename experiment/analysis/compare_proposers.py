"""Compare proposer models across pilot sessions.

The pilot's purpose is not energy -- one GPU cannot attribute it. Its purpose is
BEHAVIOURAL: does a given model produce well-formed, rule-respecting, actually
better recipes, and how expensive is it in tokens and time? Those properties
decide whether a model is usable as an experimental subject at all, and they are
measurable on a laptop.

    python analysis/compare_proposers.py --experiments-dir ../experiments/pilot
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

METRICS = [
    ("model", "proposer model"),
    ("valid", "session valid"),
    ("iterations", "iterations"),
    ("kept", "kept"),
    ("reverted", "reverted"),
    ("rejected", "guard-rejected"),
    ("errored", "errored"),
    ("infra_errors", "  of which infra"),
    ("contract_violation_rate", "contract violation rate"),
    ("guard_rejection_rate", "guard rejection rate"),
    ("baseline_val_acc", "baseline val_acc"),
    ("best_val_acc", "best val_acc"),
    ("delta_val_acc_pp", "improvement (pp)"),
    ("test_acc", "test_acc"),
    ("mean_proposer_latency_s", "mean proposal latency (s)"),
    ("total_prompt_tokens", "prompt tokens"),
    ("total_completion_tokens", "completion tokens"),
    ("tokens_per_kept", "completion tokens per kept"),
    ("wallclock_min", "session wall clock (min)"),
]


def summarise(run_dir: Path) -> dict | None:
    log_path = run_dir / "session.jsonl"
    if not log_path.exists():
        return None
    recs = read_log(log_path)
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return None
    s = json.loads(summary_path.read_text())

    model = "unknown"
    stub = synthetic = False
    for r in recs:
        if r["ev"] == "proposer_config":
            model = r.get("model", "unknown")
        if r["ev"] == "session_start":
            stub = bool(r.get("stub", False))
            synthetic = bool(r.get("synthetic_data", False))

    iters = pd.DataFrame(iterations_from_log(recs))
    n = max(1, int(s.get("iterations", 1)))
    t0 = min(r["t"] for r in recs)
    t1 = max(r["t"] for r in recs)

    row = {
        "run": run_dir.name, "model": model, "stub": stub, "synthetic": synthetic,
        **{k: s.get(k) for k in ("valid", "iterations", "kept", "reverted",
                                 "rejected", "errored", "infra_errors",
                                 "baseline_val_acc", "best_val_acc", "test_acc",
                                 "no_progress", "invalid_reason", "eps")},
        "contract_violation_rate": round(s.get("err_contract_violation", 0) / n, 3),
        "guard_rejection_rate": round(s.get("err_guard_rejection", 0) / n, 3),
        "wallclock_min": round((t1 - t0) / 60, 1),
    }
    if s.get("best_val_acc") is not None and s.get("baseline_val_acc") is not None:
        row["delta_val_acc_pp"] = round(
            100 * (s["best_val_acc"] - s["baseline_val_acc"]), 2)

    if len(iters):
        row["total_prompt_tokens"] = int(iters.prompt_tokens.fillna(0).sum())
        row["total_completion_tokens"] = int(iters.completion_tokens.fillna(0).sum())
        row["mean_proposer_latency_s"] = round(
            float(iters.proposer_latency_s.mean(skipna=True)), 1)
        kept = max(1, int(s.get("kept", 0)))
        row["tokens_per_kept"] = int(row["total_completion_tokens"] / kept) \
            if s.get("kept") else None
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments-dir", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--include-scaffolding", action="store_true",
                    help="also show stub, synthetic-data and unknown-model runs")
    a = ap.parse_args()

    root = Path(a.experiments_dir)
    rows = [r for r in (summarise(p.parent) for p in root.rglob("session.jsonl")) if r]
    if not rows:
        print(f"no sessions under {root}")
        return 1

    df = pd.DataFrame(rows)

    # Scaffolding runs are not proposers. A stub session or a synthetic-data
    # session scores ~0.10 by construction and would read as a catastrophic
    # model; an "unknown" model is a pre-instrumentation leftover. Keep them out
    # of the comparison unless explicitly asked for.
    is_scaffold = (df.get("stub", False) | df.get("synthetic", False)
                   | (df.model == "unknown"))
    scaffold = df[is_scaffold]
    if not a.include_scaffolding:
        df = df[~is_scaffold]

    # An invalid session is infrastructure evidence, not model evidence.
    invalid = df[df.valid == False]                                    # noqa: E712
    df = df[df.valid != False].sort_values("model")                    # noqa: E712

    if not len(df):
        print("no valid model sessions to compare")
        if len(scaffold):
            print(f"({len(scaffold)} scaffolding run(s) skipped; "
                  f"--include-scaffolding to see them)")
        return 1
    out = Path(a.out or root / "proposer_comparison.csv")
    df.to_csv(out, index=False)

    print("\n=== proposer comparison " + "=" * 46)
    width = max(len(lbl) for _, lbl in METRICS) + 2
    models = df.model.tolist()
    print(" " * width + "".join(f"{m[:22]:>24s}" for m in models))
    for key, label in METRICS:
        if key == "model" or key not in df.columns:
            continue
        vals = []
        for v in df[key].tolist():
            if v is None or (isinstance(v, float) and pd.isna(v)):
                vals.append("—")
            elif isinstance(v, (bool,)):
                vals.append(str(v))
            elif isinstance(v, float) and float(v).is_integer() and abs(v) < 1e6:
                vals.append(str(int(v)))
            elif isinstance(v, float):
                vals.append(f"{v:.4f}" if abs(v) < 10 else f"{v:.1f}")
            else:
                vals.append(str(v))
        print(f"{label:<{width}}" + "".join(f"{v:>24s}" for v in vals))

    print("\nReading this table:")
    print("  * contract violation rate -- can the model even produce a parseable")
    print("    reply? A high rate makes a model unusable as a subject regardless")
    print("    of how efficient it is.")
    print("  * guard rejection rate -- does it respect the task rules, especially")
    print("    the fixed time budget it is told not to touch?")
    print("  * improvement (pp) -- did it beat the baseline by more than EPS?")
    print("  * completion tokens per kept -- the behavioural proxy for the energy")
    print("    cost of real progress. On the two-GPU server this becomes joules.")
    if len(invalid):
        print("\nEXCLUDED -- invalid (infrastructure, not evidence about the model):")
        for _, r in invalid.iterrows():
            print(f"  {r['run']} ({r['model']}): {r['invalid_reason']}")
    if len(scaffold):
        print("\nEXCLUDED -- scaffolding (not proposer sessions):")
        for _, r in scaffold.iterrows():
            why = ("stub proposer" if r.get("stub") else
                   "synthetic data" if r.get("synthetic") else
                   "no proposer_config recorded")
            print(f"  {r['run']}: {why}")
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
