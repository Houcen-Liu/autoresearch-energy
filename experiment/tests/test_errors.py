"""Error taxonomy and session validity.

These exist because of what the pilot did: four Ollama read timeouts were
recorded as `errored`, which would have entered the run table as evidence that
the proposer produced four useless mutations.
"""
import sys
import time
from pathlib import Path

import pytest
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "workload"))

from harness import agent_loop                                    # noqa: E402
from harness.errors import ErrorClass, ProposerTimeout            # noqa: E402
from harness.proposer import Proposer                             # noqa: E402
from harness.session_log import read_log                          # noqa: E402
from prepare_cifar import prepare_synthetic                       # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "train_numpy.py"


def test_infra_and_agent_errors_are_distinguished():
    assert ErrorClass.INFRA_TIMEOUT.is_infra
    assert ErrorClass.INFRA_TRANSPORT.is_infra
    assert not ErrorClass.CONTRACT_VIOLATION.is_infra
    assert ErrorClass.CONTRACT_VIOLATION.is_agent
    assert ErrorClass.GUARD_REJECTION.is_agent
    assert ErrorClass.TRAIN_CRASH.is_agent


def test_timeout_raises_a_classified_error(monkeypatch):
    p = Proposer("http://127.0.0.1:9/v1", "m", timeout_s=1, max_retries=1)

    def boom(*a, **k):
        raise requests.Timeout("read timed out")

    monkeypatch.setattr(requests, "post", boom)
    with pytest.raises(ProposerTimeout) as e:
        p.complete("sys", "user")
    assert e.value.error_class == ErrorClass.INFRA_TIMEOUT
    assert e.value.attempts, "failed attempts must still be recorded"


def test_time_budget_is_enforced_across_retries(monkeypatch):
    """The pilot burned 3 x 300 s per failed iteration. That must be impossible."""
    p = Proposer("http://x/v1", "m", timeout_s=10, max_retries=5, time_budget_s=1.0)
    calls = {"n": 0}

    class Resp:
        status_code = 200

        @staticmethod
        def raise_for_status():
            pass

        @staticmethod
        def json():
            return {"choices": [{"message": {"content": "no code block here"}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 3}}

    def slow(*a, **k):
        calls["n"] += 1
        time.sleep(0.5)
        return Resp()

    monkeypatch.setattr(requests, "post", slow)
    t0 = time.time()
    with pytest.raises(ProposerTimeout):
        p.complete("sys", "user")
    assert time.time() - t0 < 3.0, "time budget was not enforced"
    assert calls["n"] <= 3


def test_failed_attempts_still_report_tokens(monkeypatch):
    p = Proposer("http://x/v1", "m", timeout_s=5, max_retries=1, time_budget_s=30)

    class Resp:
        status_code = 200

        @staticmethod
        def raise_for_status():
            pass

        @staticmethod
        def json():
            return {"choices": [{"message": {"content": "prose, no fenced block"}}],
                    "usage": {"prompt_tokens": 1234, "completion_tokens": 56}}

    monkeypatch.setattr(requests, "post", lambda *a, **k: Resp())
    from harness.errors import ProposerContract
    with pytest.raises(ProposerContract) as e:
        p.complete("sys", "user")
    assert sum(a.prompt_tokens for a in e.value.attempts) == 2468   # 2 attempts
    assert all(a.outcome == "contract" for a in e.value.attempts)


def test_session_with_all_infra_errors_is_marked_invalid(tmp_path, monkeypatch):
    """Exactly the pilot's failure mode: it must not look like a result."""
    prepare_synthetic(tmp_path / "data")
    cfg = {
        "name": "t", "attribution": "none",
        "gpus": {"train": "cpu", "proposer": "cpu"},
        "workload": {"train_seconds": 1.0, "train_timeout_s": 30,
                     "data_dir": str(tmp_path / "data")},
        "proposer": {"endpoints": {"dense": "http://127.0.0.1:9/v1"},
                     "temperature": 0.0, "top_p": 1.0, "max_tokens": 32,
                     "request_timeout_s": 1, "max_retries": 0, "time_budget_s": 2},
        "energy": {"nvml": False, "energibridge": False},
        "loop": {"eps": 0.001, "cooldown_s": 0, "max_infra_error_rate": 0.25},
    }

    def boom(*a, **k):
        raise requests.Timeout("read timed out")

    monkeypatch.setattr(requests, "post", boom)
    s = agent_loop.run_session(cfg, proposer_arm="dense", patience=1, loop_budget=3,
                               run_dir=tmp_path / "run", seed=0,
                               baseline_path=FIXTURE)

    assert s["errored"] == 3
    assert s["err_infra_timeout"] == 3
    assert s["infra_error_rate"] == 1.0
    assert s["valid"] is False
    assert "infra error rate" in s["invalid_reason"]

    recs = read_log(tmp_path / "run" / "session.jsonl")
    errs = [r for r in recs if r["ev"] == "propose_error"]
    assert all(r["is_infra"] for r in errs)


def test_guard_rejections_do_not_invalidate_a_session(tmp_path):
    """A model that breaks the rules is a RESULT, not a broken session."""
    prepare_synthetic(tmp_path / "data")
    cfg = {
        "name": "t", "attribution": "none",
        "gpus": {"train": "cpu", "proposer": "cpu"},
        "workload": {"train_seconds": 1.0, "train_timeout_s": 30,
                     "data_dir": str(tmp_path / "data")},
        "proposer": {"endpoints": {"dense": "http://unused"},
                     "temperature": 0.0, "top_p": 1.0, "max_tokens": 32,
                     "request_timeout_s": 5, "max_retries": 0},
        "energy": {"nvml": False, "energibridge": False},
        "loop": {"eps": 0.001, "cooldown_s": 0},
    }
    s = agent_loop.run_session(cfg, proposer_arm="dense", patience=1, loop_budget=4,
                               run_dir=tmp_path / "run", seed=0, stub=True,
                               baseline_path=FIXTURE)
    assert s["infra_errors"] == 0
    assert s["valid"] is True
    assert s["evaluated_iterations"] >= 1


def test_proposer_config_is_recorded(tmp_path):
    """Sampling params and thinking mode are fixed variables; they must be logged."""
    prepare_synthetic(tmp_path / "data")
    cfg = {
        "name": "t", "attribution": "none",
        "gpus": {"train": "cpu", "proposer": "cpu"},
        "workload": {"train_seconds": 1.0, "train_timeout_s": 30,
                     "data_dir": str(tmp_path / "data")},
        "proposer": {"endpoints": {"dense": "http://unused"}, "temperature": 0.0,
                     "top_p": 1.0, "max_tokens": 32, "request_timeout_s": 5,
                     "max_retries": 0},
        "energy": {"nvml": False, "energibridge": False},
        "loop": {"eps": 0.001, "cooldown_s": 0},
    }
    agent_loop.run_session(cfg, proposer_arm="dense", patience=1, loop_budget=1,
                           run_dir=tmp_path / "run", seed=0, stub=True,
                           baseline_path=FIXTURE)
    recs = read_log(tmp_path / "run" / "session.jsonl")
    assert any(r["ev"] == "proposer_config" for r in recs)


def test_dead_endpoint_aborts_early(tmp_path, monkeypatch):
    """The pilot burned 6 iterations discovering a wrong model name."""
    prepare_synthetic(tmp_path / "data")
    cfg = {
        "name": "t", "attribution": "none",
        "gpus": {"train": "cpu", "proposer": "cpu"},
        "workload": {"train_seconds": 1.0, "train_timeout_s": 30,
                     "data_dir": str(tmp_path / "data")},
        "proposer": {"endpoints": {"dense": "http://127.0.0.1:9/v1"},
                     "temperature": 0.0, "top_p": 1.0, "max_tokens": 32,
                     "request_timeout_s": 1, "max_retries": 0, "time_budget_s": 2},
        "energy": {"nvml": False, "energibridge": False},
        "loop": {"eps": 0.0073, "cooldown_s": 0, "max_infra_error_rate": 0.25},
    }
    calls = {"n": 0}

    def boom(*a, **k):
        calls["n"] += 1
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr(requests, "post", boom)
    s = agent_loop.run_session(cfg, proposer_arm="dense", patience=1,
                               loop_budget=20, run_dir=tmp_path / "run", seed=0,
                               baseline_path=FIXTURE)

    assert s["aborted_early"] is True
    assert s["errored"] <= 4, "should stop within a few iterations, not run all 20"
    assert s["valid"] is False
    recs = read_log(tmp_path / "run" / "session.jsonl")
    assert any(r["ev"] == "session_abort_early" for r in recs)


def test_healthy_session_is_not_aborted(tmp_path):
    """A session that produces results must never trip the early abort."""
    prepare_synthetic(tmp_path / "data")
    cfg = {
        "name": "t", "attribution": "none",
        "gpus": {"train": "cpu", "proposer": "cpu"},
        "workload": {"train_seconds": 1.0, "train_timeout_s": 30,
                     "data_dir": str(tmp_path / "data")},
        "proposer": {"endpoints": {"dense": "http://unused"}, "temperature": 0.0,
                     "top_p": 1.0, "max_tokens": 32, "request_timeout_s": 5,
                     "max_retries": 0},
        "energy": {"nvml": False, "energibridge": False},
        "loop": {"eps": 0.0073, "cooldown_s": 0},
    }
    s = agent_loop.run_session(cfg, proposer_arm="dense", patience=1, loop_budget=5,
                               run_dir=tmp_path / "run", seed=0, stub=True,
                               baseline_path=FIXTURE)
    assert s["aborted_early"] is False
    assert s["iterations"] == 5


def test_endpoint_must_be_a_url():
    """A config bug once put the model name in the endpoint slot."""
    with pytest.raises(ValueError, match="http"):
        Proposer("qwen3:4b", "qwen3:4b")
    Proposer("http://127.0.0.1:11434/v1", "qwen3:4b")   # must not raise


def test_rejected_extra_params_are_dropped_and_retried(monkeypatch, capsys):
    """Ollama and vLLM disagree on non-standard fields; a 4xx must not end the run."""
    p = Proposer("http://x/v1", "m", timeout_s=5, max_retries=2, time_budget_s=30,
                 extra_params={"think": False, "options": {"num_ctx": 16384}})
    seen = []

    class Resp:
        def __init__(self, code, body):
            self.status_code, self._b = code, body

        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.HTTPError(f"{self.status_code}")

        def json(self):
            return self._b

    good = {"choices": [{"message": {"content":
            "RATIONALE: x\n```python\nTRAIN_SECONDS = 1\ndef main():\n    pass\n```"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5}}

    def post(url, json=None, timeout=None):
        seen.append(dict(json))
        if "think" in json:
            return Resp(400, {"error": "unknown field"})
        return Resp(200, good)

    monkeypatch.setattr(requests, "post", post)
    r = p.complete("sys", "user")

    assert "TRAIN_SECONDS" in r.source
    assert "think" in seen[0] and "think" not in seen[1]
    assert p.request_manifest()["params_reduced"] is True
    assert "NOT in effect" in capsys.readouterr().out


def test_truncated_reply_is_diagnosed_as_truncation(monkeypatch):
    """The pilot reported 'no fenced block' when the real cause was max_tokens."""
    from harness.errors import ProposerContract
    p = Proposer("http://x/v1", "m", timeout_s=5, max_retries=0, time_budget_s=20,
                 max_tokens=2048)

    class Resp:
        status_code = 200

        @staticmethod
        def raise_for_status():
            pass

        @staticmethod
        def json():
            return {"choices": [{"finish_reason": "length",
                                 "message": {"content": "RATIONALE: x\n```python\n"
                                                        "TRAIN_SECONDS = 45.0\n# cut off"}}],
                    "usage": {"prompt_tokens": 2500, "completion_tokens": 2048}}

    monkeypatch.setattr(requests, "post", lambda *a, **k: Resp())
    with pytest.raises(ProposerContract) as e:
        p.complete("sys", "user")

    detail = e.value.attempts[-1].detail
    assert "truncated" in detail and "2048" in detail
    assert e.value.attempts[-1].finish_reason == "length"
    assert e.value.attempts[-1].raw, "the rejected reply must be kept for inspection"


def test_training_crash_is_fed_back_to_the_proposer(tmp_path, monkeypatch):
    """The pilot's 14B model repeated one hallucinated API call four times."""
    prepare_synthetic(tmp_path / "data")
    cfg = {
        "name": "t", "attribution": "none",
        "gpus": {"train": "cpu", "proposer": "cpu"},
        "workload": {"train_seconds": 1.0, "train_timeout_s": 30,
                     "data_dir": str(tmp_path / "data")},
        "proposer": {"endpoints": {"dense": "http://unused"}, "temperature": 0.0,
                     "top_p": 1.0, "max_tokens": 64, "request_timeout_s": 5,
                     "max_retries": 0},
        "energy": {"nvml": False, "energibridge": False},
        "loop": {"eps": 0.0073, "cooldown_s": 0},
    }
    prompts = []

    class Recorder:
        def request_manifest(self):
            return {"endpoint": "stub://", "model": "stub", "params": {}}

        def complete(self, system, user):
            prompts.append(user)
            import time as _t
            from harness.proposer import Attempt, ProposerResponse
            return ProposerResponse(
                source="x", rationale="r", prompt_tokens=1, completion_tokens=1,
                latency_s=0.0, t_start=_t.time(), t_end=_t.time(), attempts=1,
                raw="", attempt_log=[Attempt(1, _t.time(), _t.time(), 0.0)])

    monkeypatch.setattr(agent_loop, "StubProposer", lambda **kw: Recorder())
    monkeypatch.setattr(agent_loop.guards, "check",
                        lambda src, ts: type("G", (), {"ok": True, "violations": []})())

    def crashing(workdir, run_dir, i, cfg_, device):
        if i == 0:
            return {"errored": False, "exit": 0, "val_acc": 0.70,
                    "train_seconds": 1.0, "epochs_completed": 1, "steps": 1}
        return {"errored": True, "exit": 1, "val_acc": None,
                "error": "TypeError: erase() got an unexpected keyword argument 'p'",
                "error_class": "train_crash"}

    monkeypatch.setattr(agent_loop, "run_training", crashing)
    monkeypatch.setattr(agent_loop.shutil, "copy", lambda *a, **k: None)

    agent_loop.run_session(cfg, proposer_arm="dense", patience=1, loop_budget=3,
                           run_dir=tmp_path / "run", seed=0, stub=True,
                           baseline_path=FIXTURE)

    assert len(prompts) == 3
    assert "CRASHED" not in prompts[0], "nothing had crashed yet"
    assert "CRASHED" in prompts[1], "the traceback must reach the next prompt"
    assert "erase()" in prompts[1]


def test_prompt_suffix_goes_on_the_user_turn(monkeypatch):
    """Qwen3 honours /no_think only in the user message."""
    p = Proposer("http://x/v1", "m", timeout_s=5, max_retries=0,
                 prompt_suffix="/no_think", system_suffix="SYS")
    seen = {}

    class Resp:
        status_code = 200

        @staticmethod
        def raise_for_status():
            pass

        @staticmethod
        def json():
            return {"choices": [{"finish_reason": "stop", "message": {"content":
                    "RATIONALE: x\n```python\nTRAIN_SECONDS = 45.0\ndef main():\n    pass\n```"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1}}

    def post(url, json=None, timeout=None):
        seen.update(json)
        return Resp()

    monkeypatch.setattr(requests, "post", post)
    p.complete("system text", "user text")

    assert seen["messages"][0]["content"].endswith("SYS")
    assert seen["messages"][1]["content"].endswith("/no_think")


def test_thinking_tokens_are_counted():
    from harness.proposer import _thinking_tokens
    assert _thinking_tokens("<think>" + "a" * 400 + "</think>ok") == 100
    assert _thinking_tokens("no reasoning here") == 0
