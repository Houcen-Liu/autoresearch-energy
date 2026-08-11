"""The patience and rollback semantics are the design; they get real tests.

Training is replaced by a scripted accuracy sequence, so these run in
milliseconds with no torch, no GPU and no model.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness import agent_loop                                    # noqa: E402
from harness.session_log import iterations_from_log, read_log     # noqa: E402
from harness.proposer import ProposerResponse                     # noqa: E402

CFG = {
    "name": "test", "attribution": "none",
    "gpus": {"train": "cpu", "proposer": "cpu"},
    "workload": {"train_seconds": 1.0, "train_timeout_s": 10, "data_dir": "./data"},
    "proposer": {"endpoints": {"dense": "http://x", "moe": "http://x"},
                 "temperature": 0.0, "top_p": 1.0, "max_tokens": 128,
                 "request_timeout_s": 5, "max_retries": 1},
    "energy": {"nvml": False, "energibridge": False},
    "loop": {"eps": 0.001, "cooldown_s": 0},
}


class ScriptedProposer:
    def __init__(self, n):
        self.i = 0
        self.n = n

    def request_manifest(self):
        return {"endpoint": "scripted://", "model": "scripted", "params": {}}

    def complete(self, system, user):
        self.i += 1
        import time
        return ProposerResponse(source=f"# proposal {self.i}\n", rationale=f"p{self.i}",
                                prompt_tokens=10, completion_tokens=5, latency_s=0.0,
                                t_start=time.time(), t_end=time.time(), attempts=1,
                                raw="", attempt_log=[])


def _install(monkeypatch, accs):
    """Guard always passes; training returns the scripted accuracies."""
    monkeypatch.setattr(agent_loop.guards, "check",
                        lambda src, ts: type("G", (), {"ok": True, "violations": []})())
    seq = iter(accs)

    def fake_train(workdir, run_dir, iteration, cfg, device):
        if iteration == 0:
            return {"errored": False, "exit": 0, "val_acc": 0.70,
                    "train_seconds": 1.0, "epochs_completed": 1, "steps": 1}
        try:
            acc = next(seq)
        except StopIteration:
            acc = 0.5
        if acc is None:
            return {"errored": True, "exit": 1, "error": "boom", "val_acc": None}
        return {"errored": False, "exit": 0, "val_acc": acc,
                "train_seconds": 1.0, "epochs_completed": 1, "steps": 1}

    monkeypatch.setattr(agent_loop, "run_training", fake_train)
    monkeypatch.setattr(agent_loop, "StubProposer", lambda **kw: ScriptedProposer(10))
    monkeypatch.setattr(agent_loop, "_final_eval", lambda *a, **k: 0.66)
    monkeypatch.setattr(agent_loop.shutil, "copy", lambda *a, **k: None)


def _run(tmp_path, patience, budget):
    return agent_loop.run_session(CFG, proposer_arm="dense", patience=patience,
                                  loop_budget=budget, run_dir=tmp_path, seed=0, stub=True)


def test_greedy_reverts_every_regression(tmp_path, monkeypatch):
    # baseline 0.70; three regressions in a row
    _install(monkeypatch, [0.69, 0.68, 0.67])
    s = _run(tmp_path, patience=1, budget=3)
    assert s["kept"] == 0
    assert s["reverted"] == 3
    assert s["no_progress"] is True


def test_greedy_keeps_improvements(tmp_path, monkeypatch):
    _install(monkeypatch, [0.72, 0.71, 0.75])
    s = _run(tmp_path, patience=1, budget=3)
    assert s["kept"] == 2                      # 0.72 and 0.75
    assert s["reverted"] == 1                  # 0.71 regressed against 0.72
    assert abs(s["best_val_acc"] - 0.75) < 1e-9


def test_patience3_tolerates_two_then_rolls_back(tmp_path, monkeypatch):
    _install(monkeypatch, [0.69, 0.68, 0.67])
    s = _run(tmp_path, patience=3, budget=3)
    assert s["kept"] == 0
    # all three provisional iterations are discarded by one rollback
    assert s["reverted"] == 3
    recs = read_log(tmp_path / "session.jsonl")
    rollbacks = [r for r in recs if r["ev"] == "rollback"]
    assert len(rollbacks) == 1
    assert rollbacks[0]["discarded_iters"] == [1, 2, 3]


def test_patience3_resets_counter_on_improvement(tmp_path, monkeypatch):
    _install(monkeypatch, [0.69, 0.68, 0.80, 0.79])
    s = _run(tmp_path, patience=3, budget=4)
    assert s["kept"] == 1
    assert abs(s["best_val_acc"] - 0.80) < 1e-9
    recs = read_log(tmp_path / "session.jsonl")
    rollbacks = [r for r in recs if r["ev"] == "rollback"]
    # iterations 1,2 provisional -> discarded at budget exhaustion together with 4
    assert rollbacks[-1]["discarded_iters"] == [4]


def test_errors_count_against_the_budget(tmp_path, monkeypatch):
    _install(monkeypatch, [None, None, 0.75])
    s = _run(tmp_path, patience=1, budget=3)
    assert s["errored"] == 2
    assert s["kept"] == 1
    assert s["iterations"] == 3


def test_eps_blocks_noise_level_improvement(tmp_path, monkeypatch):
    _install(monkeypatch, [0.7005])            # +0.0005 < eps
    s = _run(tmp_path, patience=1, budget=1)
    assert s["kept"] == 0


def test_log_folds_into_iterations(tmp_path, monkeypatch):
    _install(monkeypatch, [0.72, 0.71])
    _run(tmp_path, patience=1, budget=2)
    iters = iterations_from_log(read_log(tmp_path / "session.jsonl"))
    assert [i["iter"] for i in iters] == [1, 2]
    assert iters[0]["decision"] == "keep"
    assert iters[1]["decision"] in ("revert", "reverted")
    assert all(i["propose_t0"] and i["train_t0"] for i in iters)
