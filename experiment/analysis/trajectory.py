"""Trajectory of a long-horizon session: climb, plateau, or overfit.

Produces one figure and a small set of statistics that answer, descriptively:
where did improvement stop, how much energy did each successive keep cost, and
how much of the headroom to the hand-tuned reference was recovered.

    python analysis/trajectory.py --run-dir ../experiments/long_horizon/moe_100it_...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.session_log import read_log, iterations_from_log   # noqa: E402

REFERENCE_ACC = 0.9161      # calibration recipe, never shown to the agent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--reference", type=float, default=REFERENCE_ACC)
    a = ap.parse_args()

    rd = Path(a.run_dir)
    out = Path(a.out_dir) if a.out_dir else rd / "trajectory"
    out.mkdir(parents=True, exist_ok=True)

    rows = iterations_from_log(read_log(rd / "session.jsonl"))
    summary = json.loads((rd / "summary.json").read_text())
    base = summary["baseline_val_acc"]
    eps = summary["eps"]

    it, acc, kept_it, kept_acc = [], [], [], []
    best, best_curve = base, []
    for r in rows:
        if r["iter"] == 0:
            continue
        it.append(r["iter"])
        acc.append(r["val_acc"])
        if r["val_acc"] is not None and r["val_acc"] > best + eps:
            best = r["val_acc"]
            kept_it.append(r["iter"]); kept_acc.append(r["val_acc"])
        best_curve.append(best)

    # ---------------------------------------------------------------- stats
    head_total = a.reference - base
    head_got = best - base
    print(f"baseline            {base:.4f}")
    print(f"best kept           {best:.4f}")
    print(f"reference (unseen)  {a.reference:.4f}")
    print(f"headroom recovered  {head_got*100:+.2f} pp of {head_total*100:.2f} pp "
          f"= {100*head_got/head_total:.0f} %")
    print(f"keeps               {len(kept_it)} at iterations {kept_it}")

    if kept_it:
        last = kept_it[-1]
        stalled = max(it) - last
        print(f"last improvement at iteration {last}; "
              f"{stalled} iterations since ({stalled/max(it):.0%} of the session)")
        if len(kept_it) >= 3:
            halves = [k for k in kept_it if k <= max(it) / 2]
            print(f"keeps in first half {len(halves)}, second half "
                  f"{len(kept_it)-len(halves)}  -> "
                  f"{'front-loaded' if len(halves) > len(kept_it)/2 else 'sustained'}")
    evaluated = [x for x in acc if x is not None]
    if evaluated:
        print(f"proposals evaluated {len(evaluated)}, "
              f"below baseline {sum(x < base for x in evaluated)} "
              f"({sum(x < base for x in evaluated)/len(evaluated):.0%})")

    # ----------------------------------------------------------------- plot
    fig, ax = plt.subplots(figsize=(9, 5))
    ev_it = [i for i, x in zip(it, acc) if x is not None]
    ev_ac = [x for x in acc if x is not None]
    ax.scatter(ev_it, ev_ac, s=18, alpha=0.45, label="proposal (evaluated)")
    ax.step(it, best_curve, where="post", lw=2, label="best kept")
    ax.axhline(base, ls="--", lw=1, color="grey", label=f"baseline {base:.3f}")
    ax.axhline(a.reference, ls=":", lw=1.5, color="black",
               label=f"hand-tuned reference {a.reference:.3f}")
    if kept_it:
        ax.scatter(kept_it, kept_acc, marker="^", s=70, zorder=5, label="kept")
    ax.set_xlabel("iteration"); ax.set_ylabel("validation accuracy")
    ax.set_title(f"Long-horizon trajectory ({rd.name})")
    ax.legend(loc="lower right", fontsize=8); ax.grid(alpha=0.25)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"trajectory.{ext}", dpi=150)
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
