"""Synthetic coverage for host-energy reconstruction from EnergiBridge."""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.aggregate import collect                            # noqa: E402
from measurement.energy_align import align                        # noqa: E402
from scripts.make_synthetic_runs import make_run                  # noqa: E402


def _run(tmp_path: Path) -> tuple[Path, float, float]:
    run_dir = tmp_path / "run_0_repetition_0"
    make_run(run_dir, "dense", "greedy", 1, 0, seed=3,
             train_seconds=0.2)
    records = [
        json.loads(line)
        for line in (run_dir / "session.jsonl").read_text().splitlines()
    ]
    return run_dir, min(row["t"] for row in records), max(row["t"] for row in records)


def _write_trace(run_dir: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(run_dir / "energibridge.csv", index=False)


def _time_ms(timestamp_s: float) -> int:
    return round(timestamp_s * 1000)


def test_amd_cpu_energy_is_the_package_total(tmp_path):
    run_dir, t0, t1 = _run(tmp_path)
    _write_trace(run_dir, [
        {"Time": _time_ms(t0 + 0.01), "CPU_ENERGY (J)": 100.0,
         "CORE0_ENERGY (J)": 1_000.0},
        {"Time": _time_ms(t1 - 0.01), "CPU_ENERGY (J)": 140.0,
         "CORE0_ENERGY (J)": 9_000.0},
    ])

    result = align(run_dir, 0, 1)

    assert result["E_cpu_pkg_J"] == pytest.approx(40.0)
    assert result["E_dram_J"] is None
    assert result["E_measured_total_J"] == pytest.approx(
        result["E_gpu_total_J"] + 40.0
    )


def test_intel_package_and_dram_are_preserved_separately(tmp_path):
    run_dir, t0, t1 = _run(tmp_path)
    _write_trace(run_dir, [
        {"Time": _time_ms(t0 + 0.01), "PACKAGE_ENERGY (J)": 50.0,
         "DRAM_ENERGY (J)": 10.0},
        {"Time": _time_ms(t1 - 0.01), "PACKAGE_ENERGY (J)": 80.0,
         "DRAM_ENERGY (J)": 17.0},
    ])

    result = align(run_dir, 0, 1)

    assert result["E_cpu_pkg_J"] == pytest.approx(30.0)
    assert result["E_dram_J"] == pytest.approx(7.0)
    assert result["E_measured_total_J"] == pytest.approx(
        result["E_gpu_total_J"] + 37.0
    )

    tidy, _, _ = collect(tmp_path)
    assert tidy.loc[0, "E_cpu_pkg_J"] == pytest.approx(30.0)
    assert tidy.loc[0, "E_dram_J"] == pytest.approx(7.0)
    assert tidy.loc[0, "E_measured_total_J"] == pytest.approx(
        result["E_measured_total_J"]
    )


def test_missing_host_trace_keeps_gpu_as_the_measured_total(tmp_path):
    run_dir, _, _ = _run(tmp_path)

    result = align(run_dir, 0, 1)

    assert result["E_cpu_pkg_J"] is None
    assert result["E_dram_J"] is None
    assert result["E_measured_total_J"] == pytest.approx(result["E_gpu_total_J"])


def test_host_counters_use_session_timestamp_bounds(tmp_path):
    run_dir, t0, t1 = _run(tmp_path)
    _write_trace(run_dir, [
        {"Time": _time_ms(t0 - 1), "CPU_ENERGY (J)": 0.0},
        {"Time": _time_ms(t0 + 0.01), "CPU_ENERGY (J)": 1_000.0},
        {"Time": _time_ms(t1 - 0.01), "CPU_ENERGY (J)": 1_025.0},
        {"Time": _time_ms(t1 + 1), "CPU_ENERGY (J)": 9_000.0},
    ])

    result = align(run_dir, 0, 1)

    assert result["E_cpu_pkg_J"] == pytest.approx(25.0)


def test_core_counters_alone_are_not_treated_as_package_energy(tmp_path):
    run_dir, t0, t1 = _run(tmp_path)
    _write_trace(run_dir, [
        {"Time": _time_ms(t0 + 0.01), "CORE0_ENERGY (J)": 10.0,
         "CORE1_ENERGY (J)": 20.0},
        {"Time": _time_ms(t1 - 0.01), "CORE0_ENERGY (J)": 110.0,
         "CORE1_ENERGY (J)": 220.0},
    ])

    result = align(run_dir, 0, 1)

    assert result["E_cpu_pkg_J"] is None
    assert result["E_dram_J"] is None
    assert result["E_measured_total_J"] == pytest.approx(result["E_gpu_total_J"])
