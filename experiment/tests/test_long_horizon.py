"""Safety checks for the one-off long-horizon launcher."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import long_horizon  # noqa: E402


@pytest.mark.parametrize("artifact", ["session.jsonl", "summary.json"])
def test_refuses_run_directory_with_session_artifacts(
        tmp_path, monkeypatch, artifact):
    run_dir = tmp_path / "long-run"
    run_dir.mkdir()
    (run_dir / artifact).write_text("existing", encoding="utf-8")
    called = False

    def should_not_start(*args, **kwargs):
        nonlocal called
        called = True
        return subprocess.CompletedProcess(args[0], 0)

    monkeypatch.setattr(long_horizon.subprocess, "run", should_not_start)
    monkeypatch.setattr(sys, "argv", [
        "long_horizon.py", "--run-dir", str(run_dir), "--iterations", "1",
    ])

    assert long_horizon.main() == 2
    assert called is False


def test_allows_external_measurement_files_in_new_run_directory(
        tmp_path, monkeypatch):
    run_dir = tmp_path / "long-run"
    run_dir.mkdir()
    (run_dir / "energibridge.csv").write_text("Time,CPU_ENERGY (J)\n",
                                               encoding="utf-8")
    (run_dir / "console.log").write_text("measurement wrapper started\n",
                                          encoding="utf-8")
    commands = []

    def fake_start(cmd, **kwargs):
        commands.append(cmd)
        (run_dir / "summary.json").write_text(
            json.dumps({"valid": True}), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(long_horizon.subprocess, "run", fake_start)
    monkeypatch.setattr(sys, "argv", [
        "long_horizon.py", "--run-dir", str(run_dir), "--iterations", "1",
    ])

    assert long_horizon.main() == 0
    assert len(commands) == 1
    assert str(run_dir) in commands[0]


def test_returns_failure_for_scientifically_invalid_session(tmp_path, monkeypatch):
    run_dir = tmp_path / "long-run"

    def fake_start(cmd, **kwargs):
        (run_dir / "summary.json").write_text(json.dumps({
            "valid": False,
            "invalid_reason": "infra error rate exceeded",
        }), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(long_horizon.subprocess, "run", fake_start)
    monkeypatch.setattr(sys, "argv", [
        "long_horizon.py", "--run-dir", str(run_dir), "--iterations", "1",
    ])

    assert long_horizon.main() == 3
