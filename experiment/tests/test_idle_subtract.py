"""Synthetic tests for matched-idle sensitivity accounting."""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from measurement.idle_subtract import subtract_idle  # noqa: E402


def _write_idle(idle_dir: Path, gpu0_w: float, gpu1_w: float,
                cpu_w: float | None = None, dram_w: float | None = None) -> None:
    idle_dir.mkdir(parents=True)
    duration = 10.0
    (idle_dir / "idle_summary.json").write_text(json.dumps({
        "duration_s": duration,
        "energy_j_per_device": {"0": gpu0_w * duration,
                                "1": gpu1_w * duration},
        "mean_power_w_per_device": {"0": gpu0_w, "1": gpu1_w},
    }), encoding="utf-8")
    rows = [{"Time": 1_000_000}, {"Time": 1_010_000}]
    if cpu_w is not None:
        rows[0]["CPU_ENERGY (J)"] = 100.0
        rows[1]["CPU_ENERGY (J)"] = 100.0 + cpu_w * duration
    if dram_w is not None:
        rows[0]["DRAM_ENERGY (J)"] = 20.0
        rows[1]["DRAM_ENERGY (J)"] = 20.0 + dram_w * duration
    pd.DataFrame(rows).to_csv(idle_dir / "energibridge.csv", index=False)


def _write_gross(run_dir: Path, **overrides) -> None:
    summary = {
        "wallclock_s": 10.0,
        "E_train_J": 1_000.0,
        "E_prop_J": 600.0,
        "E_cpu_pkg_J": 400.0,
        "E_dram_J": 100.0,
        "E_measured_total_J": 2_100.0,
    }
    summary.update(overrides)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "energy_summary.json").write_text(json.dumps(summary),
                                                   encoding="utf-8")


def test_subtracts_matched_mean_power_per_component(tmp_path):
    run_dir = tmp_path / "run"
    _write_gross(run_dir)
    _write_idle(run_dir / "idle_before", 10.0, 20.0, 3.0, 1.0)
    _write_idle(run_dir / "idle_after", 12.0, 18.0, 5.0, 3.0)

    result = subtract_idle(run_dir)

    assert result["primary_outcome"] is False
    assert result["components"]["gpu0"]["idle_power_mean_W"] == 11.0
    assert result["components"]["gpu1"]["idle_power_mean_W"] == 19.0
    assert result["components"]["cpu_package"]["idle_power_mean_W"] \
        == pytest.approx(4.0)
    assert result["components"]["dram"]["idle_power_mean_W"] \
        == pytest.approx(2.0)
    assert result["components"]["gpu0"]["idle_subtracted_energy_J"] == 890.0
    assert result["components"]["gpu1"]["idle_subtracted_energy_J"] == 410.0
    assert result["components"]["cpu_package"]["idle_subtracted_energy_J"] \
        == pytest.approx(360.0)
    assert result["components"]["dram"]["idle_subtracted_energy_J"] \
        == pytest.approx(80.0)
    assert result["idle_subtracted_measured_total_J"] == pytest.approx(1_740.0)
    assert (run_dir / "idle_subtracted_summary.json").exists()


def test_negative_component_result_is_not_clamped(tmp_path):
    run_dir = tmp_path / "run"
    _write_gross(run_dir, wallclock_s=1.0, E_train_J=5.0,
                 E_prop_J=100.0, E_cpu_pkg_J=None, E_dram_J=None,
                 E_measured_total_J=105.0)
    _write_idle(run_dir / "idle_before", 10.0, 1.0)
    _write_idle(run_dir / "idle_after", 10.0, 1.0)

    result = subtract_idle(run_dir)

    assert result["components"]["gpu0"]["idle_subtracted_energy_J"] == -5.0
    assert result["idle_subtracted_measured_total_J"] == pytest.approx(94.0)


def test_missing_host_idle_keeps_host_out_of_adjusted_total(tmp_path):
    run_dir = tmp_path / "run"
    _write_gross(run_dir, E_dram_J=None, E_measured_total_J=2_000.0)
    _write_idle(run_dir / "idle_before", 10.0, 20.0)
    _write_idle(run_dir / "idle_after", 12.0, 18.0)

    result = subtract_idle(run_dir)

    assert result["unadjusted_measured_components"] == ["cpu_package"]
    assert result["idle_subtracted_measured_total_J"] is None
    assert result["idle_subtracted_available_total_J"] == pytest.approx(1_300.0)


def test_requires_both_matched_gpu_idle_summaries(tmp_path):
    run_dir = tmp_path / "run"
    _write_gross(run_dir)
    _write_idle(run_dir / "idle_before", 10.0, 20.0)

    with pytest.raises(FileNotFoundError, match="idle_after"):
        subtract_idle(run_dir)


def test_cli_writes_sensitivity_summary(tmp_path):
    run_dir = tmp_path / "run"
    _write_gross(run_dir, E_cpu_pkg_J=None, E_dram_J=None,
                 E_measured_total_J=1_600.0)
    _write_idle(run_dir / "idle_before", 10.0, 20.0)
    _write_idle(run_dir / "idle_after", 12.0, 18.0)

    proc = subprocess.run([
        sys.executable, str(ROOT / "measurement" / "idle_subtract.py"),
        "--run-dir", str(run_dir), "--gpu-train", "0", "--gpu-prop", "1",
    ], cwd=ROOT, capture_output=True, text=True)

    assert proc.returncode == 0, proc.stderr
    written = json.loads((run_dir / "idle_subtracted_summary.json").read_text())
    assert written["analysis_type"] == "matched-idle sensitivity analysis"
    assert written["primary_outcome"] is False
