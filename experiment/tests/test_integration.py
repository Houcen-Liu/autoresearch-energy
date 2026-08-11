"""End-to-end harness integration, torch-free.

Runs a real session: real subprocess execution, real git history, real guards,
real session log, real energy alignment. Only the recipe is swapped for a numpy
stand-in and the data for a synthetic cache, so this runs anywhere.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "workload"))

from harness.agent_loop import run_session                        # noqa: E402
from harness.session_log import iterations_from_log, read_log     # noqa: E402
from prepare_cifar import prepare_synthetic                       # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "train_numpy.py"


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    prepare_synthetic(d)
    return d


def _cfg(data_dir):
    return {
        "name": "integration", "attribution": "none",
        "gpus": {"train": "cpu", "proposer": "cpu"},
        "workload": {"train_seconds": 2.0, "train_timeout_s": 90,
                     "data_dir": str(data_dir)},
        "proposer": {"endpoints": {"dense": "http://unused", "moe": "http://unused"},
                     "temperature": 0.0, "top_p": 1.0, "max_tokens": 256,
                     "request_timeout_s": 5, "max_retries": 1},
        "energy": {"nvml": False, "energibridge": False},
        "loop": {"eps": 0.001, "cooldown_s": 0},
    }


def test_full_session_runs(tmp_path, data_dir):
    s = run_session(_cfg(data_dir), proposer_arm="dense", patience=1, loop_budget=4,
                    run_dir=tmp_path, seed=1, stub=True, baseline_path=FIXTURE)

    assert s["iterations"] == 4
    assert s["kept"] + s["reverted"] + s["rejected"] + s["errored"] >= 4
    assert 0.0 <= s["baseline_val_acc"] <= 1.0
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "recipe_history.bundle").exists()


def test_git_history_records_every_iteration(tmp_path, data_dir):
    run_session(_cfg(data_dir), proposer_arm="dense", patience=1, loop_budget=3,
                run_dir=tmp_path, seed=2, stub=True, baseline_path=FIXTURE)
    import subprocess
    log = subprocess.run(["git", "-C", str(tmp_path / "recipe"), "log", "--oneline"],
                         capture_output=True, text=True).stdout
    assert "baseline" in log
    assert log.count("\n") >= 2


def test_guard_rejects_budget_tampering_in_a_live_session(tmp_path, data_dir):
    """The stub proposer emits a budget-cheating proposal ~15 % of the time."""
    seen = False
    for seed in range(6):
        rd = tmp_path / f"s{seed}"
        run_session(_cfg(data_dir), proposer_arm="dense", patience=1, loop_budget=4,
                    run_dir=rd, seed=seed, stub=True, baseline_path=FIXTURE)
        recs = read_log(rd / "session.jsonl")
        if any(r["ev"] == "guard" and not r["ok"] for r in recs):
            seen = True
            bad = [r for r in recs if r["ev"] == "guard" and not r["ok"]][0]
            assert any("TRAIN_SECONDS" in v for v in bad["violations"])
            break
    assert seen, "stub never produced a rule-violating proposal in 6 sessions"


def test_iterations_have_complete_phase_boundaries(tmp_path, data_dir):
    run_session(_cfg(data_dir), proposer_arm="dense", patience=3, loop_budget=4,
                run_dir=tmp_path, seed=3, stub=True, baseline_path=FIXTURE)
    iters = iterations_from_log(read_log(tmp_path / "session.jsonl"))
    assert len(iters) == 4
    for it in iters:
        assert it["propose_t0"] and it["propose_t1"]
        assert it["decision"] is not None
        if it["guard_ok"]:
            assert it["train_t0"] and it["train_t1"]
            assert it["train_t1"] > it["train_t0"]
