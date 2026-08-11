"""Is the inner workload's accuracy sensitive to training compute at all?

NOTE: this is a DIAGNOSTIC, not the gate. It asks whether the BASELINE responds
to compute. The decision of which training budget to use is made by
`scripts/headroom_check.py` (gate G11), which measures the signal-to-noise ratio
an IMPROVED recipe gets at each budget -- the quantity the agent actually
operates on. Run this one when you want to understand the baseline's behaviour;
run headroom_check.py when you need to choose. See EXPERIMENT_PLAN.md D12.

THE PROBLEM THIS EXISTS TO DETECT.

The pilot's noise-floor run produced, at a 45 s budget, step counts of 8500,
7652, 7159, 6486 and 5780 -- a 37 % spread from thermal throttling -- and
validation accuracies of 0.7520, 0.7668, 0.7614, 0.7434, 0.7664. The correlation
between steps and accuracy was -0.13: none. Cutting training compute by more than
a third changed nothing.

That is a saturated workload, and it is dangerous for this experiment:

  * `loop_budget` (10 vs 20) moves ENERGY but cannot move ACCURACY, so the
    energy/accuracy Pareto frontier degenerates to "the smallest budget wins" --
    a true statement about a broken measurement instrument, not a finding about
    agentic AutoML;
  * every mutation lands inside the noise band, so keep/revert carries little
    signal, and `E/kept` measures coin flips;
  * a longer training budget (240 s vs 45 s) costs 5x the joules for no accuracy,
    which would make the whole study's energy axis a measure of elapsed time.

This script measures the accuracy-versus-compute curve directly: one run per
budget, shortest first. What you want to see is accuracy still climbing at the
budget you intend to use. Where the curve flattens is where the workload stops
being able to answer the experiment's question.

    python scripts/budget_sensitivity.py --profile profiles/pilot.yaml \
        --budgets 5,10,20,45,90

If the curve is flat at your intended budget, fix the WORKLOAD before freezing
it: widen the baseline model, or shorten the budget to where compute still
matters. Do not proceed to Phase 1 with a saturated baseline.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness.agent_loop import run_training                        # noqa: E402


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
    return num / den if den else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default=str(ROOT / "profiles" / "pilot.yaml"))
    ap.add_argument("--budgets", default="5,10,20,45,90",
                    help="comma-separated training budgets in seconds")
    ap.add_argument("--cooldown", type=float, default=30,
                    help="seconds between runs, so throttling does not "
                         "masquerade as a compute effect")
    ap.add_argument("--out", default="budget_sensitivity.json")
    a = ap.parse_args()

    cfg = yaml.safe_load(Path(a.profile).read_text())
    budgets = [float(b) for b in a.budgets.split(",")]

    work = Path(tempfile.mkdtemp(prefix="budget_"))
    shutil.copy(ROOT / "workload" / "train.py", work / "train.py")
    shutil.copy(ROOT / "workload" / "prepare_cifar.py", work / "prepare_cifar.py")

    total = sum(budgets) + a.cooldown * len(budgets)
    print(f"Accuracy vs training compute: {len(budgets)} runs, "
          f"~{total / 60:.0f} min\n")
    print(f"  {'budget':>8s} {'val_acc':>9s} {'epochs':>7s} {'steps':>7s} {'delta':>8s}")

    rows = []
    prev = None
    for b in sorted(budgets):
        if rows:
            time.sleep(a.cooldown)
        cfg["workload"]["train_seconds"] = b
        res = run_training(work, work, len(rows), cfg, "cuda")
        if res.get("errored"):
            print(f"  {b:8.0f}   FAILED: {str(res.get('error'))[:120]}")
            continue
        delta = "" if prev is None else f"{100 * (res['val_acc'] - prev):+7.2f}pp"
        print(f"  {b:8.0f} {res['val_acc']:9.4f} {res['epochs_completed']:7d} "
              f"{res['steps']:7d} {delta:>8s}")
        rows.append({"train_seconds": b, **res})
        prev = res["val_acc"]

    if len(rows) < 3:
        print("\nToo few successful runs.")
        return 1

    accs = [r["val_acc"] for r in rows]
    steps = [float(r["steps"]) for r in rows]
    r_steps = _pearson(steps, accs)

    # Marginal gain over the top half of the range: is the curve still moving?
    mid = len(rows) // 2
    tail_gain_pp = 100 * (accs[-1] - accs[mid])
    tail_compute_ratio = steps[-1] / steps[mid] if steps[mid] else float("nan")

    report = {
        "profile": a.profile, "budgets": budgets, "cooldown_s": a.cooldown,
        "corr_steps_vs_acc": round(r_steps, 3),
        "total_gain_pp": round(100 * (accs[-1] - accs[0]), 2),
        "tail_gain_pp": round(tail_gain_pp, 2),
        "tail_compute_ratio": round(tail_compute_ratio, 2),
        "saturated": bool(tail_gain_pp < 1.0),
        "runs": rows,
    }
    Path(a.out).write_text(json.dumps(report, indent=2))

    print(f"""
--- accuracy vs compute ---
  correlation(steps, val_acc)   {r_steps:+.2f}
  gain across the whole range   {report['total_gain_pp']:+.2f} pp
  gain over the top half        {tail_gain_pp:+.2f} pp for {tail_compute_ratio:.1f}x the compute""")

    if report["saturated"]:
        print("""
  *** WORKLOAD SATURATED ***
  Accuracy stops responding to compute inside the range you tested. As it
  stands, `loop_budget` will move energy but not accuracy, and the
  energy/accuracy Pareto frontier degenerates to "smallest budget wins".

  Fix the workload before freezing it, in this order of preference:
    1. WIDEN THE BASELINE MODEL (more channels or depth) so it needs the
       compute. Preferred: it keeps the agent's improvement headroom intact.
    2. SHORTEN THE BUDGET to where the curve is still climbing. Cheaper in
       GPU-hours, but check the noise floor again -- shorter runs are noisier.
    3. Adding augmentation to the baseline also delays saturation, but it
       spends one of the agent's most obvious moves, so prefer 1 or 2.

  Then re-run this script and the noise floor.""")
    else:
        print("""
  Accuracy is still responding to compute at the top of the tested range.
  The workload can answer the loop-budget question. Choose the budget at the
  knee of this curve and record the curve in the report.""")

    print(f"\n  written to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
