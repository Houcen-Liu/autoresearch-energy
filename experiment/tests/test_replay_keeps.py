"""Focused tests for post-hoc replay selection."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.replay_keeps import kept_shas, main  # noqa: E402


def test_kept_shas_uses_harness_keep_decision():
    records = [
        {"ev": "train_start", "iter": 1, "sha": "sha-kept"},
        {"ev": "decision", "iter": 1, "decision": "keep"},
        {"ev": "train_start", "iter": 2, "sha": "sha-reverted"},
        {"ev": "decision", "iter": 2, "decision": "revert"},
    ]

    assert kept_shas(records) == [(1, "sha-kept")]


def test_no_keeps_still_writes_replay_artifact(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "session.jsonl").write_text('{"ev":"session_start"}\n')
    (run_dir / "recipe_history.bundle").write_bytes(b"placeholder")
    monkeypatch.setattr(sys, "argv", ["replay_keeps.py", "--run-dir", str(run_dir)])

    assert main() == 0
    assert (run_dir / "replay_keeps.json").read_text() == "[]\n"
