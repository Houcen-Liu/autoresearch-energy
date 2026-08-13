"""Matched-idle subtraction for the exploratory long-horizon session.

This is a sensitivity analysis, not the primary energy outcome. It subtracts
the mean of matched pre/post idle powers from each measured component for the
session wall-clock duration. Gross measured energy remains primary.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from measurement.energy_align import _host_energy


def _number(value: object) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _gpu_idle_power(summary: dict, device: int) -> float | None:
    """Read a device's mean power, with energy/duration as a fallback."""
    powers = summary.get("mean_power_w_per_device") or {}
    power = _number(powers.get(str(device), powers.get(device)))
    if power is not None:
        return power

    duration = _number(summary.get("duration_s"))
    energies = summary.get("energy_j_per_device") or {}
    energy = _number(energies.get(str(device), energies.get(device)))
    if duration is None or duration <= 0 or energy is None:
        return None
    return energy / duration


def _host_idle_power(idle_dir: Path) -> tuple[float | None, float | None]:
    """Return package/DRAM mean powers over an idle EnergiBridge trace."""
    path = idle_dir / "energibridge.csv"
    if not path.exists():
        return None, None
    try:
        trace = pd.read_csv(path)
        if "Time" not in trace.columns:
            return None, None
        times = pd.to_numeric(trace["Time"], errors="coerce").dropna() / 1000.0
        if len(times) < 2:
            return None, None
        t0, t1 = float(times.min()), float(times.max())
        duration = t1 - t0
        if duration <= 0:
            return None, None
        cpu_energy, dram_energy = _host_energy(idle_dir, t0, t1)
        return (
            None if cpu_energy is None else cpu_energy / duration,
            None if dram_energy is None else dram_energy / duration,
        )
    except (OSError, ValueError, KeyError, pd.errors.ParserError):
        return None, None


def _idle_powers(idle_dir: Path, gpu_train: int, gpu_prop: int) -> dict:
    summary_path = idle_dir / "idle_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing matched idle summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    cpu_power, dram_power = _host_idle_power(idle_dir)
    return {
        f"gpu{gpu_train}": _gpu_idle_power(summary, gpu_train),
        f"gpu{gpu_prop}": _gpu_idle_power(summary, gpu_prop),
        "cpu_package": cpu_power,
        "dram": dram_power,
    }


def subtract_idle(run_dir: str | Path, gpu_train: int = 0,
                  gpu_prop: int = 1) -> dict:
    """Compute a matched-idle sensitivity result and write it beside the run."""
    run_dir = Path(run_dir)
    if gpu_train == gpu_prop:
        raise ValueError("gpu-train and gpu-prop must be distinct for per-device energy")

    energy_path = run_dir / "energy_summary.json"
    if not energy_path.exists():
        raise FileNotFoundError(
            f"missing {energy_path}; run measurement/energy_align.py first"
        )
    gross_summary = json.loads(energy_path.read_text(encoding="utf-8"))
    wallclock = _number(gross_summary.get("wallclock_s"))
    if wallclock is None or wallclock <= 0:
        raise ValueError("energy_summary.json has no positive wallclock_s")

    before = _idle_powers(run_dir / "idle_before", gpu_train, gpu_prop)
    after = _idle_powers(run_dir / "idle_after", gpu_train, gpu_prop)
    for component in (f"gpu{gpu_train}", f"gpu{gpu_prop}"):
        if before[component] is None or after[component] is None:
            raise ValueError(f"matched idle power unavailable for required {component}")

    gross = {
        f"gpu{gpu_train}": _number(gross_summary.get("E_train_J")),
        f"gpu{gpu_prop}": _number(gross_summary.get("E_prop_J")),
        "cpu_package": _number(gross_summary.get("E_cpu_pkg_J")),
        "dram": _number(gross_summary.get("E_dram_J")),
    }
    roles = {
        f"gpu{gpu_train}": "training GPU board",
        f"gpu{gpu_prop}": "proposer GPU board",
        "cpu_package": "CPU package",
        "dram": "DRAM",
    }

    components = {}
    adjusted, unadjusted = [], []
    for name, gross_energy in gross.items():
        before_power, after_power = before[name], after[name]
        mean_power = (
            (before_power + after_power) / 2.0
            if before_power is not None and after_power is not None else None
        )
        idle_energy = None if mean_power is None else mean_power * wallclock
        # Deliberately do not clamp: a negative result is a diagnostic that the
        # idle baseline exceeds the session's gross component measurement.
        net_energy = (
            None if gross_energy is None or idle_energy is None
            else gross_energy - idle_energy
        )
        components[name] = {
            "role": roles[name],
            "gross_energy_J": gross_energy,
            "idle_power_before_W": before_power,
            "idle_power_after_W": after_power,
            "idle_power_mean_W": mean_power,
            "idle_energy_for_session_J": idle_energy,
            "idle_subtracted_energy_J": net_energy,
        }
        if gross_energy is not None:
            (adjusted if net_energy is not None else unadjusted).append(name)

    available_net_total = sum(
        components[name]["idle_subtracted_energy_J"] for name in adjusted
    )
    available_idle_total = sum(
        components[name]["idle_energy_for_session_J"] for name in adjusted
    )
    gross_measured_total = _number(gross_summary.get("E_measured_total_J"))
    if gross_measured_total is None:
        gross_measured_total = sum(value for value in gross.values() if value is not None)

    result = {
        "analysis_type": "matched-idle sensitivity analysis",
        "primary_outcome": False,
        "interpretation": (
            "Sensitivity result only; gross measured energy remains the primary "
            "auditable outcome. Negative component results are retained, not clamped."
        ),
        "run_dir": str(run_dir),
        "session_wallclock_s": wallclock,
        "gpu_train_device": gpu_train,
        "gpu_prop_device": gpu_prop,
        "gross_measured_total_J": gross_measured_total,
        "components": components,
        "adjusted_components": adjusted,
        "unadjusted_measured_components": unadjusted,
        "idle_energy_subtracted_available_total_J": available_idle_total,
        "idle_subtracted_available_total_J": available_net_total,
        "idle_subtracted_measured_total_J": (
            available_net_total if not unadjusted else None
        ),
    }
    out = run_dir / "idle_subtracted_summary.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Matched-idle sensitivity analysis for a long-horizon run"
    )
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--gpu-train", type=int, default=0)
    ap.add_argument("--gpu-prop", type=int, default=1)
    args = ap.parse_args()
    try:
        result = subtract_idle(args.run_dir, args.gpu_train, args.gpu_prop)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"[FAIL] {exc}")
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
