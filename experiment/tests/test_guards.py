import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness import guards                                        # noqa: E402

BASELINE = (ROOT / "workload" / "train.py").read_text()


def test_baseline_passes():
    r = guards.check(BASELINE, 240.0)
    assert r.ok, r.violations


def test_budget_tampering_is_caught():
    bad = BASELINE.replace("TRAIN_SECONDS = 240.0", "TRAIN_SECONDS = 900.0")
    r = guards.check(bad, 240.0)
    assert not r.ok
    assert any("TRAIN_SECONDS" in v for v in r.violations)


def test_missing_budget_is_caught():
    bad = BASELINE.replace("TRAIN_SECONDS = 240.0", "SECONDS = 240.0")
    assert not guards.check(bad, 240.0).ok


def test_test_set_access_is_caught():
    bad = BASELINE.replace("load_splits(args.data_dir)",
                           "load_splits(args.data_dir, include_test=True)")
    r = guards.check(bad, 240.0)
    assert not r.ok
    assert any("test set" in v for v in r.violations)


def test_banned_import_is_caught():
    bad = "import subprocess\n" + BASELINE
    assert not guards.check(bad, 240.0).ok


def test_syntax_error_is_caught():
    r = guards.check("def broken(:\n  pass", 240.0)
    assert not r.ok
    assert "parse" in r.violations[0]


def test_missing_build_model_is_caught():
    bad = BASELINE.replace("def build_model()", "def make_net()")
    r = guards.check(bad, 240.0)
    assert not r.ok
    assert any("build_model" in v for v in r.violations)
