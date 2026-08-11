"""Choose the training budget by measuring the signal-to-noise ratio it gives the agent.

WHAT THIS CORRECTS.

`budget_sensitivity.py` asks whether the BASELINE's accuracy responds to compute.
On the pilot it does not: 4.4x the steps (7 115 -> 31 375, a 45 s vs 240 s budget)
bought +0.91 pp against a pooled noise SD of 0.96 pp -- a signal-to-noise ratio of
0.95, i.e. nothing.

But a saturated baseline does not by itself condemn a budget. The baseline is a
deliberately weak recipe; what matters is whether an IMPROVED recipe can exploit
the budget, because that is the accuracy range the agent actually explores.
Augmentation especially only repays over many epochs, so a budget that looks
generous for the baseline can be far too short for the agent's single best move.

So this script measures the quantity that actually decides the budget:

    headroom  = mean(reference recipe) - mean(baseline recipe)
    SNR       = headroom / pooled SD of the two

`workload/train_reference.py` is a competent-but-unexotic recipe (batch norm,
random crop and flip, one-cycle schedule, wider channels) that stands in for what
a good agent would find. It is calibration scaffolding and is never shown to the
agent.

Pick by SNR, not by headroom: a budget yielding +10 pp with 5 pp of noise is
worse than one yielding +6 pp with 1 pp of noise. Below SNR ~3 the agent cannot
reliably tell its own improvements from noise, and keep/revert degrades toward
coin flipping.

Among budgets that clear that bar, take the CHEAPEST. Every extra second is
multiplied by loop_budget x 24 sessions, so a budget with a marginally better SNR
can easily cost ten GPU-hours for signal the agent cannot use.

WHY THE BASELINE'S OWN CURVE CANNOT DECIDE THIS. The pilot's baseline plateaus at
~0.75 from a 20 s budget onward (0.7516 at 20 s, 0.7598 at 45 s, 0.7464 at 90 s,
0.7671 at 240 s -- flat inside a ~1 pp noise band). Reading "pick 20 s" off that
curve would be a mistake: the plateau is a property of a deliberately weak recipe
with no augmentation and no normalisation. Augmented recipes keep improving for
far longer, so a 20 s budget would cap what the agent can ever achieve and
compress the headroom this experiment depends on.

    python scripts/headroom_check.py --budgets 45,240 --repeats 2
"""
from __future__ import annotations

import argparse
import json
import shutil
import statistics as st
import sys
import tempfile
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness.agent_loop import run_training                        # noqa: E402


def _run_recipe(recipe: Path, work: Path, cfg: dict, tag: str, repeats: int,
                cooldown: float) -> list[dict]:
    shutil.copy(recipe, work / "train.py")
    out = []
    for i in range(repeats):
        if out or tag != "baseline":
            time.sleep(cooldown)
        res = run_training(work, work, i, cfg, "cuda")
        if res.get("errored"):
            print(f"    {tag} {i+1}: FAILED -- {str(res.get('error'))[:160]}")
            continue
        out.append(res)
        print(f"    {tag} {i+1}: val_acc={res['val_acc']:.4f}  "
              f"epochs={res['epochs_completed']}  steps={res['steps']}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default=str(ROOT / "profiles" / "pilot.yaml"))
    ap.add_argument("--budgets", default="45,240")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--cooldown", type=float, default=30)
    ap.add_argument("--min-snr", type=float, default=3.0,
                    help="the SNR the agent needs to tell improvement from noise")
    ap.add_argument("--out", default="headroom.json")
    a = ap.parse_args()

    cfg = yaml.safe_load(Path(a.profile).read_text())
    budgets = [float(b) for b in a.budgets.split(",")]

    work = Path(tempfile.mkdtemp(prefix="headroom_"))
    shutil.copy(ROOT / "workload" / "prepare_cifar.py", work / "prepare_cifar.py")

    est = sum(b * 2 * a.repeats for b in budgets) + \
        a.cooldown * 2 * a.repeats * len(budgets)
    print(f"Headroom at {len(budgets)} budget(s), {a.repeats} repeat(s) each, "
          f"~{est / 60:.0f} min\n")

    results = []
    for b in budgets:
        cfg["workload"]["train_seconds"] = b
        print(f"  budget {b:.0f}s")
        base = _run_recipe(ROOT / "workload" / "train.py", work, cfg,
                           "baseline", a.repeats, a.cooldown)
        ref = _run_recipe(ROOT / "workload" / "train_reference.py", work, cfg,
                          "reference", a.repeats, a.cooldown)
        if not base or not ref:
            print("    insufficient data at this budget")
            continue

        ba = [r["val_acc"] for r in base]
        ra = [r["val_acc"] for r in ref]
        headroom = st.mean(ra) - st.mean(ba)
        sds = [st.stdev(v) for v in (ba, ra) if len(v) > 1]
        pooled = (sum(s ** 2 for s in sds) / len(sds)) ** 0.5 if sds else float("nan")
        snr = headroom / pooled if pooled and pooled == pooled else float("nan")

        # Validity check on the instrument itself. If the reference cannot get
        # through a comparable number of steps, the comparison measures
        # THROUGHPUT, not recipe quality, and the headroom number is meaningless.
        # The pilot's first reference did 1352 steps against 33735 and scored
        # -3.8 pp; that is not "no headroom", it is a broken measuring stick.
        base_steps = st.mean([r["steps"] for r in base])
        ref_steps = st.mean([r["steps"] for r in ref])
        step_ratio = ref_steps / base_steps if base_steps else float("nan")
        ref_epochs = st.mean([r["epochs_completed"] for r in ref])
        instrument_ok = step_ratio >= 0.25 and ref_epochs >= 3

        row = {"train_seconds": b,
               "step_ratio": round(step_ratio, 3),
               "reference_epochs": round(ref_epochs, 1),
               "instrument_ok": bool(instrument_ok),
               "baseline_mean": round(st.mean(ba), 4),
               "reference_mean": round(st.mean(ra), 4),
               "headroom_pp": round(100 * headroom, 2),
               "pooled_sd_pp": round(100 * pooled, 2) if pooled == pooled else None,
               "snr": round(snr, 2) if snr == snr else None,
               "baseline_runs": base, "reference_runs": ref}
        results.append(row)
        print(f"    -> headroom {row['headroom_pp']:+.2f} pp, "
              f"noise {row['pooled_sd_pp']} pp, SNR {row['snr']}")
        if not instrument_ok:
            print(f"       INVALID: reference did {step_ratio:.0%} of the baseline's "
                  f"steps ({ref_epochs:.0f} epochs). This measures throughput,\n"
                  f"       not recipe quality -- the reference is too expensive "
                  f"per step for this budget.")
        print()

    if not results:
        print("No usable results.")
        return 1

    usable = [r for r in results
              if r["snr"] is not None and r["instrument_ok"] and r["headroom_pp"] > 0]
    invalid = [r for r in results if not r["instrument_ok"] or r["headroom_pp"] <= 0]
    max_snr = max(usable, key=lambda r: r["snr"]) if usable else None
    # Economics: every extra second of budget is multiplied by (loop_budget x
    # sessions), so the right choice is the CHEAPEST budget that clears the SNR
    # the agent needs -- not the one with the highest SNR outright.
    adequate = sorted((r for r in usable if r["snr"] >= a.min_snr),
                      key=lambda r: r["train_seconds"])
    best = adequate[0] if adequate else max_snr
    report = {"profile": a.profile, "repeats": a.repeats, "min_snr": a.min_snr,
              "recommended_train_seconds": best["train_seconds"] if best else None,
              "recommendation_rule": ("cheapest budget clearing min_snr" if adequate
                                      else "max SNR (nothing cleared min_snr)"),
              "max_snr_train_seconds": max_snr["train_seconds"] if max_snr else None,
              "budgets": results}
    Path(a.out).write_text(json.dumps(report, indent=2))

    print("--- headroom by budget ---")
    print(f"  {'budget':>8s} {'baseline':>9s} {'reference':>10s} "
          f"{'headroom':>10s} {'noise':>8s} {'SNR':>6s} {'valid':>6s}")
    for r in results:
        print(f"  {r['train_seconds']:8.0f} {r['baseline_mean']:9.4f} "
              f"{r['reference_mean']:10.4f} {r['headroom_pp']:9.2f}pp "
              f"{r['pooled_sd_pp']:7.2f}pp {r['snr']:6.2f} "
              f"{'yes' if r['instrument_ok'] and r['headroom_pp'] > 0 else 'NO':>6s}")

    if invalid and not usable:
        print("""
  *** NO VALID MEASUREMENT AT ANY BUDGET ***
  The reference recipe never beat the baseline, which means the instrument is
  wrong, not that the workload lacks headroom. Under a fixed WALL-CLOCK budget a
  recipe that is expensive per step cannot finish training, and loses to a cheap
  one regardless of its quality. Shrink the reference recipe until its step
  count is within ~2x of the baseline's, then re-run.

  Worth keeping as a finding: the fixed-time budget systematically penalises
  added capacity. That is a real property of the autoresearch design and belongs
  in the report.""")
        print(f"\n  written to {a.out}")
        return 1

    if best:
        print(f"""
  RECOMMENDED BUDGET: {best['train_seconds']:.0f}s   ({report['recommendation_rule']})
    headroom {best['headroom_pp']:+.2f} pp against {best['pooled_sd_pp']:.2f} pp of noise (SNR {best['snr']:.1f})""")
        if max_snr and max_snr["train_seconds"] != best["train_seconds"]:
            ratio = max_snr["train_seconds"] / best["train_seconds"]
            print(f"""    {max_snr['train_seconds']:.0f}s has the higher SNR ({max_snr['snr']:.1f}) but costs {ratio:.1f}x the
    joules per inner experiment, multiplied by loop_budget x 24 sessions. Take it
    only if the extra signal is worth that.""")
        if best["snr"] < a.min_snr:
            print("""
  WARNING: SNR below 3 at every budget tested. The agent cannot reliably
  distinguish its own improvements from run-to-run noise, so keep/revert
  degrades toward coin flipping and `E/kept` measures luck. Before Phase 1,
  either reduce the noise (average val_acc over repeats inside train.py, or
  evaluate on a larger validation split) or widen the gap (weaken the baseline
  further). Report whichever you choose.""")
        elif best["headroom_pp"] < 3:
            print("""
  NOTE: headroom under 3 pp. The agent has little room to improve anything,
  which will produce many no-progress sessions. Consider weakening the
  baseline before freezing it.""")

    print(f"\n  written to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
