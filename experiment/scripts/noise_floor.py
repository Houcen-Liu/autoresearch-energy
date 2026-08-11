"""Measure the noise floor of the inner workload, and set EPS from it.

Why this exists. Three runs of the identical baseline on the pilot machine gave
validation accuracies of 0.7632, 0.7670 and 0.7590 -- a spread of 0.8 accuracy
points at a fixed seed. That is not a bug: `autoresearch` budgets training by
WALL CLOCK, so the number of optimisation steps depends on machine state
(8461 vs 8620 steps across those runs). No seed can remove it.

The consequence is severe. The agent's keep/revert rule used EPS = 0.001, four
times SMALLER than the run-to-run standard deviation. Under that threshold a
mutation that changes nothing at all is kept roughly half the time, so
`kept mutations`, `E_per_kept` and the entire patience factor would have been
measuring noise.

This script repeats the unmodified baseline N times and reports:

  * mean, SD and range of val_acc            -- the raw spread
  * DETRENDED SD and a drift test            -- the two are not the same thing
  * a suggested EPS (2 x detrended SD)       -- the smallest defensible threshold
  * step and epoch variability               -- how much the wall-clock budget moves
  * GPU utilisation, power and temperature   -- whether E_train reflects work or idling

Why the drift test exists. The first pilot measurement ran five repeats back to
back with no cooldown and produced steps of 8500, 7652, 7159, 6486, 5780 and mean
power of 115, 91, 81, 76, 72 W, at a constant 88 % utilisation: a 32 % throughput
collapse from thermal and power-limit throttling, not from randomness. Reporting
that as "noise" would inflate EPS and hide a systematic run-order confound. So
repeats are now separated by a cooldown, and any residual monotonic trend is
reported separately from the scatter around it.

Run it on the SERVER, per candidate budget, before Phase 1. The chosen EPS is a
fixed variable of the experiment and belongs in the report next to the noise
floor that justifies it.

    python scripts/noise_floor.py --profile profiles/server.yaml --repeats 8
    python scripts/noise_floor.py --profile profiles/server.yaml --repeats 5 --train-seconds 60
"""
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness.agent_loop import run_training                        # noqa: E402


def _gpu_stats(nvml_path: Path, dev: int, t0: float, t1: float) -> dict:
    try:
        import pandas as pd
        df = pd.read_csv(nvml_path)
        w = df[(df.dev == dev) & (df.t_wall >= t0) & (df.t_wall <= t1)]
        if w.empty:
            return {}
        out = {"mean_power_w": round(float(w.power_mw.mean()) / 1000, 1),
               "mean_util_pct": round(float(w.util_gpu.mean()), 1),
               "max_util_pct": int(w.util_gpu.max())}
        if "temp_c" in w.columns:
            out["mean_temp_c"] = round(float(w.temp_c.mean()), 1)
            out["max_temp_c"] = int(w.temp_c.max())
        return out
    except Exception:                                              # noqa: BLE001
        return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default=str(ROOT / "profiles" / "server.yaml"))
    ap.add_argument("--repeats", type=int, default=8)
    ap.add_argument("--train-seconds", type=float, default=None,
                    help="override the profile's budget (to compare budgets)")
    ap.add_argument("--cooldown", type=float, default=None,
                    help="seconds between repeats; defaults to the profile's "
                         "loop.cooldown_s. Set 0 only to MEASURE drift.")
    ap.add_argument("--out", default="noise_floor.json")
    a = ap.parse_args()

    cfg = yaml.safe_load(Path(a.profile).read_text())
    if a.train_seconds:
        cfg["workload"]["train_seconds"] = a.train_seconds
    budget = cfg["workload"]["train_seconds"]

    work = Path(tempfile.mkdtemp(prefix="noise_"))
    shutil.copy(ROOT / "workload" / "train.py", work / "train.py")
    shutil.copy(ROOT / "workload" / "prepare_cifar.py", work / "prepare_cifar.py")

    sampler = None
    if cfg.get("energy", {}).get("nvml"):
        try:
            from measurement.nvml_sampler import NvmlSampler
            sampler = NvmlSampler(work / "nvml.csv", hz=10)
            sampler.start()
        except Exception as e:                                     # noqa: BLE001
            print(f"[noise] NVML unavailable ({e}); skipping utilisation stats")

    cooldown = a.cooldown if a.cooldown is not None else \
        float(cfg.get("loop", {}).get("cooldown_s", 60))
    total_min = a.repeats * (budget + cooldown) / 60
    print(f"Repeating the baseline {a.repeats}x at a {budget:.0f}s budget, "
          f"{cooldown:.0f}s cooldown between repeats (~{total_min:.0f} min)\n")
    if cooldown == 0:
        print("  cooldown 0: this measures DRIFT, not noise\n")

    runs = []
    for i in range(a.repeats):
        if i and cooldown:
            time.sleep(cooldown)
        t0 = time.time()
        res = run_training(work, work, i, cfg, "cuda")
        t1 = time.time()
        if res.get("errored"):
            print(f"  run {i+1}: FAILED -- {str(res.get('error'))[:200]}")
            continue
        gpu = _gpu_stats(work / "nvml.csv", int(cfg["gpus"]["train"]), t0, t1) \
            if sampler and str(cfg["gpus"]["train"]).isdigit() else {}
        runs.append({**res, **gpu})
        print(f"  run {i+1}: val_acc={res['val_acc']:.4f}  "
              f"epochs={res['epochs_completed']}  steps={res['steps']}"
              + (f"  util={gpu.get('mean_util_pct')}%  {gpu.get('mean_power_w')}W"
                 if gpu else ""))

    if sampler:
        sampler.stop()
    if len(runs) < 3:
        print("\nToo few successful runs to estimate a noise floor.")
        return 1

    accs = [r["val_acc"] for r in runs]
    steps = [r["steps"] for r in runs]
    sd = statistics.stdev(accs)

    # Separate systematic drift (thermal, power limits) from random scatter.
    # A least-squares line against run index is enough at these sample sizes;
    # the residual SD is what EPS should be built on.
    def _spearman_vs_order(ys):
        """Rank correlation against run order.

        A least-squares slope alone is not enough: the 240 s pilot series
        (32837, 26400, 30355, 32510, 34771) dipped once and recovered, and a
        linear fit called that a +13 % 'drift'. Requiring a MONOTONIC trend as
        well kills that false positive -- the 45 s series was perfectly
        monotonic (rho = -1.00), the 240 s series was not (rho = +0.40).
        """
        n = len(ys)
        rx = list(range(1, n + 1))
        order = sorted(range(n), key=lambda i: ys[i])
        ry = [0] * n
        for rank, i in enumerate(order, start=1):
            ry[i] = rank
        mx, my = statistics.mean(rx), statistics.mean(ry)
        num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
        den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
        return num / den if den else 0.0

    def _trend(ys):
        n = len(ys)
        xs = list(range(n))
        mx, my = statistics.mean(xs), statistics.mean(ys)
        denom = sum((x - mx) ** 2 for x in xs)
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom if denom else 0.0
        resid = [y - (my + slope * (x - mx)) for x, y in zip(xs, ys)]
        return slope, resid

    acc_slope, acc_resid = _trend(accs)
    step_slope, _ = _trend([float(x) for x in steps])
    powers = [r.get("mean_power_w") for r in runs if r.get("mean_power_w")]
    power_slope = _trend(powers)[0] if len(powers) == len(runs) else None

    sd_detrended = statistics.stdev(acc_resid) if len(acc_resid) > 2 else sd
    suggested_eps = round(2 * sd_detrended, 4)
    step_drift_pct = 100 * step_slope * (len(steps) - 1) / statistics.mean(steps)
    step_rho = _spearman_vs_order([float(x) for x in steps])
    # Drift requires BOTH a material magnitude and a monotonic trend. Either one
    # alone produces false alarms at these sample sizes.
    drift_suspected = abs(step_drift_pct) > 10 and abs(step_rho) >= 0.8

    report = {
        "profile": a.profile, "train_seconds": budget, "repeats": len(runs),
        "cooldown_s": cooldown,
        "val_acc_sd_detrended": round(sd_detrended, 5),
        "acc_slope_pp_per_run": round(100 * acc_slope, 3),
        "step_drift_pct_total": round(step_drift_pct, 1),
        "step_drift_spearman": round(step_rho, 2),
        "power_slope_w_per_run": (round(power_slope, 2) if power_slope is not None else None),
        "drift_suspected": bool(drift_suspected),
        "val_acc_mean": round(statistics.mean(accs), 4),
        "val_acc_sd": round(sd, 5),
        "val_acc_min": round(min(accs), 4), "val_acc_max": round(max(accs), 4),
        "val_acc_range_pp": round(100 * (max(accs) - min(accs)), 2),
        "steps_mean": round(statistics.mean(steps), 1),
        "steps_cv_pct": round(100 * statistics.stdev(steps) / statistics.mean(steps), 2),
        "epochs": sorted({r["epochs_completed"] for r in runs}),
        "suggested_eps": suggested_eps,
        "mean_util_pct": (round(statistics.mean([r["mean_util_pct"] for r in runs]), 1)
                          if "mean_util_pct" in runs[0] else None),
        "mean_power_w": (round(statistics.mean([r["mean_power_w"] for r in runs]), 1)
                         if "mean_power_w" in runs[0] else None),
        "runs": runs,
    }
    Path(a.out).write_text(json.dumps(report, indent=2))

    print(f"""
--- noise floor at a {budget:.0f}s budget, n={len(runs)}, cooldown {cooldown:.0f}s ---
  val_acc        {report['val_acc_mean']:.4f}
  raw SD         {sd:.4f}   (includes any drift)
  detrended SD   {sd_detrended:.4f}   <- the actual noise floor
  range          {report['val_acc_min']:.4f} .. {report['val_acc_max']:.4f} """
          f"""({report['val_acc_range_pp']:.2f} pp)
  steps          {report['steps_mean']:.0f}, CV {report['steps_cv_pct']:.1f}%, """
          f"""trend {step_drift_pct:+.1f}% (rho {step_rho:+.2f})
  epochs         {report['epochs']}

  SUGGESTED EPS = {suggested_eps}  (2 x detrended SD)

  An improvement smaller than this cannot be distinguished from re-running the
  same recipe. Put this number, and the noise floor behind it, in the report.""")

    if not drift_suspected and abs(step_drift_pct) > 10:
        print(f"""
  Throughput moved {step_drift_pct:+.0f}% across the series but NOT monotonically
  (rho {step_rho:+.2f}), so this is scatter or a one-off disturbance rather than
  thermal drift. The cooldown is doing its job.""")

    if drift_suspected:
        print(f"""
  *** SYSTEMATIC DRIFT DETECTED (rho {step_rho:+.2f}, monotonic) ***
  Throughput moved {step_drift_pct:+.0f}% across the series"""
              + (f" and mean power {power_slope:+.1f} W per run" if power_slope else "")
              + f""".
  That is thermal or power-limit behaviour, not randomness. Consequences:
    * identical work costs different joules depending on run order, so the
      ENERGY comparison between cells is confounded, not just the accuracy one;
    * a longer cooldown is needed (currently {cooldown:.0f}s), and the run table
      must stay randomised so drift cannot align with a factor;
    * record temperature per run and report it.
  Re-run with a longer --cooldown and confirm the drift shrinks before trusting
  either the noise floor or any energy number from this machine.""")

    if report["mean_util_pct"] is not None:
        print(f"""
  GPU during training: {report['mean_util_pct']:.0f}% utilisation, "
                       {report['mean_power_w']:.0f} W mean""")
        if report["mean_util_pct"] < 50:
            print("""  WARNING: the training GPU is mostly idle. E_train is then dominated by
  idle power x time rather than by work done, which weakens the energy
  contrast between kept and reverted mutations. Consider a baseline that
  actually loads the card (larger batch or wider model).""")

    print(f"\n  written to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
