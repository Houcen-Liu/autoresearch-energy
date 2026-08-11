"""Drift must not be reported as noise.

Replays the actual pilot measurement -- steps 8500..5780, power 115..72 W at a
constant 88 % utilisation -- and requires the script to call it drift.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import noise_floor                                                # noqa: E402

# The real pilot series (45 s budget, no cooldown).
PILOT = [
    {"val_acc": 0.7520, "epochs_completed": 24, "steps": 8500, "mean_power_w": 114.7},
    {"val_acc": 0.7668, "epochs_completed": 21, "steps": 7652, "mean_power_w": 90.7},
    {"val_acc": 0.7614, "epochs_completed": 20, "steps": 7159, "mean_power_w": 80.8},
    {"val_acc": 0.7434, "epochs_completed": 18, "steps": 6486, "mean_power_w": 76.0},
    {"val_acc": 0.7664, "epochs_completed": 16, "steps": 5780, "mean_power_w": 72.0},
]


def _install(monkeypatch, series, out):
    it = iter(series)

    def fake_train(workdir, run_dir, i, cfg, device):
        r = dict(next(it))
        r.update({"errored": False, "exit": 0, "train_seconds": 45.0,
                  "peak_vram_mb": 438})
        return r

    monkeypatch.setattr(noise_floor, "run_training", fake_train)
    monkeypatch.setattr(noise_floor, "_gpu_stats",
                        lambda *a, **k: {"mean_util_pct": 88.0})
    monkeypatch.setattr(sys, "argv",
                        ["noise_floor.py", "--profile",
                         str(ROOT / "profiles" / "pilot.yaml"),
                         "--repeats", str(len(series)), "--cooldown", "0",
                         "--out", str(out)])


def test_pilot_series_is_flagged_as_drift(tmp_path, monkeypatch, capsys):
    out = tmp_path / "nf.json"
    # _gpu_stats is stubbed, so power comes from the series itself
    monkeypatch.setattr(noise_floor, "_gpu_stats",
                        lambda *a, **k: {"mean_util_pct": 88.0})
    _install(monkeypatch, PILOT, out)
    noise_floor.main()

    r = json.loads(out.read_text())
    assert r["drift_suspected"] is True
    assert r["step_drift_pct_total"] < -25          # ~-32 % throughput collapse
    assert "SYSTEMATIC DRIFT DETECTED" in capsys.readouterr().out


def test_detrending_lowers_the_suggested_eps(tmp_path, monkeypatch):
    """A downward trend inflates raw SD; EPS must be built on the residual."""
    out = tmp_path / "nf.json"
    trending = [{"val_acc": 0.80 - 0.01 * i, "epochs_completed": 20,
                 "steps": 8000 - 400 * i, "mean_power_w": 110 - 8 * i}
                for i in range(6)]
    _install(monkeypatch, trending, out)
    noise_floor.main()

    r = json.loads(out.read_text())
    assert r["val_acc_sd_detrended"] < r["val_acc_sd"]
    assert r["suggested_eps"] == pytest.approx(2 * r["val_acc_sd_detrended"], abs=1e-4)
    assert r["drift_suspected"] is True


def test_stable_series_reports_no_drift(tmp_path, monkeypatch):
    out = tmp_path / "nf.json"
    stable = [{"val_acc": v, "epochs_completed": 20, "steps": s, "mean_power_w": 100}
              for v, s in [(0.762, 8000), (0.759, 8020), (0.764, 7990),
                           (0.760, 8010), (0.763, 8005)]]
    _install(monkeypatch, stable, out)
    noise_floor.main()

    r = json.loads(out.read_text())
    assert r["drift_suspected"] is False
    assert r["suggested_eps"] < 0.01


# --------------------------------------------------------------- saturation
def _install_budget(monkeypatch, series, out, budgets):
    import budget_sensitivity
    it = iter(series)

    def fake_train(workdir, run_dir, i, cfg, device):
        r = dict(next(it))
        r.update({"errored": False, "exit": 0,
                  "train_seconds": cfg["workload"]["train_seconds"],
                  "epochs_completed": r["steps"] // 350, "peak_vram_mb": 438})
        return r

    monkeypatch.setattr(budget_sensitivity, "run_training", fake_train)
    monkeypatch.setattr(sys, "argv",
                        ["budget_sensitivity.py", "--profile",
                         str(ROOT / "profiles" / "pilot.yaml"),
                         "--budgets", budgets, "--cooldown", "0",
                         "--out", str(out)])
    return budget_sensitivity


def test_saturated_workload_is_flagged(tmp_path, monkeypatch, capsys):
    """A curve that flattens must stop the experiment, not pass quietly."""
    out = tmp_path / "bs.json"
    flat = [{"val_acc": v, "steps": s} for v, s in
            [(0.70, 1000), (0.745, 2000), (0.758, 4000),
             (0.760, 8000), (0.761, 16000)]]
    bs = _install_budget(monkeypatch, flat, out, "5,10,20,45,90")
    bs.main()
    r = json.loads(out.read_text())
    assert r["saturated"] is True
    assert r["tail_gain_pp"] < 1.0
    assert "WORKLOAD SATURATED" in capsys.readouterr().out


def test_responsive_workload_passes(tmp_path, monkeypatch, capsys):
    out = tmp_path / "bs.json"
    climbing = [{"val_acc": v, "steps": s} for v, s in
                [(0.55, 1000), (0.62, 2000), (0.69, 4000),
                 (0.75, 8000), (0.80, 16000)]]
    bs = _install_budget(monkeypatch, climbing, out, "5,10,20,45,90")
    bs.main()
    r = json.loads(out.read_text())
    assert r["saturated"] is False
    assert r["corr_steps_vs_acc"] > 0.8
    assert "still responding to compute" in capsys.readouterr().out


def test_non_monotonic_wobble_is_not_called_drift(tmp_path, monkeypatch, capsys):
    """The real 240 s pilot series: dips once, recovers. Not thermal drift."""
    out = tmp_path / "nf.json"
    series = [{"val_acc": v, "epochs_completed": e, "steps": s, "mean_power_w": p}
              for v, e, s, p in [
                  (0.7736, 93, 32837, 82.3), (0.7576, 75, 26400, 69.3),
                  (0.7766, 86, 30355, 73.5), (0.7704, 92, 32510, 76.6),
                  (0.7574, 98, 34771, 80.6)]]
    _install(monkeypatch, series, out)
    noise_floor.main()
    r = json.loads(out.read_text())
    assert r["drift_suspected"] is False, "linear slope alone gave a false alarm"
    assert abs(r["step_drift_spearman"]) < 0.8
    assert "SYSTEMATIC DRIFT DETECTED" not in capsys.readouterr().out


def test_monotonic_decline_is_still_caught(tmp_path, monkeypatch, capsys):
    """The real 45 s pilot series: perfectly monotonic. That IS drift."""
    out = tmp_path / "nf.json"
    _install(monkeypatch, PILOT, out)
    noise_floor.main()
    r = json.loads(out.read_text())
    assert r["drift_suspected"] is True
    assert r["step_drift_spearman"] == -1.0


# ------------------------------------------------------------------ headroom
def test_headroom_prefers_the_better_snr(tmp_path, monkeypatch, capsys):
    """A budget with more headroom but proportionally more noise must lose."""
    import headroom_check
    out = tmp_path / "hr.json"

    # 45 s: +6 pp headroom, 1 pp noise (SNR 6). 240 s: +10 pp, 5 pp noise (SNR 2).
    plan = {
        (45.0, "baseline"): [0.760, 0.770],
        (45.0, "reference"): [0.820, 0.830],
        (240.0, "baseline"): [0.740, 0.790],
        (240.0, "reference"): [0.840, 0.890],
    }
    state = {"budget": None, "recipe": None, "i": 0}

    def fake_copy(src, dst):
        state["recipe"] = "reference" if "reference" in str(src) else "baseline"
        state["i"] = 0

    def fake_train(workdir, run_dir, i, cfg, device):
        b = float(cfg["workload"]["train_seconds"])
        v = plan[(b, state["recipe"])][state["i"] % 2]
        state["i"] += 1
        return {"errored": False, "exit": 0, "val_acc": v, "epochs_completed": 20,
                "steps": int(b * 150), "train_seconds": b, "peak_vram_mb": 500}

    monkeypatch.setattr(headroom_check, "run_training", fake_train)
    monkeypatch.setattr(headroom_check.shutil, "copy", fake_copy)
    monkeypatch.setattr(sys, "argv",
                        ["headroom_check.py", "--profile",
                         str(ROOT / "profiles" / "pilot.yaml"),
                         "--budgets", "45,240", "--repeats", "2",
                         "--cooldown", "0", "--out", str(out)])
    headroom_check.main()

    r = json.loads(out.read_text())
    assert r["recommended_train_seconds"] == 45.0
    by = {b["train_seconds"]: b for b in r["budgets"]}
    assert by[240.0]["headroom_pp"] > by[45.0]["headroom_pp"]   # bigger gap...
    assert by[240.0]["snr"] < by[45.0]["snr"]                    # ...worse SNR


def test_cheapest_adequate_budget_wins_over_highest_snr(tmp_path, monkeypatch):
    """Extra signal the agent cannot use is not worth 5x the joules."""
    import headroom_check
    out = tmp_path / "hr.json"
    # both clear SNR 3; 240 s has the higher SNR but costs 5.3x per experiment
    plan = {
        (45.0, "baseline"): [0.760, 0.764],
        (45.0, "reference"): [0.830, 0.834],
        (240.0, "baseline"): [0.766, 0.770],
        (240.0, "reference"): [0.900, 0.904],
    }
    state = {"recipe": None, "i": 0}
    monkeypatch.setattr(headroom_check.shutil, "copy",
                        lambda src, dst: state.update(
                            recipe=("reference" if "reference" in str(src)
                                    else "baseline"), i=0))

    def fake_train(workdir, run_dir, i, cfg, device):
        b = float(cfg["workload"]["train_seconds"])
        v = plan[(b, state["recipe"])][state["i"] % 2]
        state["i"] += 1
        return {"errored": False, "exit": 0, "val_acc": v, "epochs_completed": 20,
                "steps": int(b * 150), "train_seconds": b, "peak_vram_mb": 500}

    monkeypatch.setattr(headroom_check, "run_training", fake_train)
    monkeypatch.setattr(sys, "argv",
                        ["headroom_check.py", "--profile",
                         str(ROOT / "profiles" / "pilot.yaml"),
                         "--budgets", "45,240", "--repeats", "2",
                         "--cooldown", "0", "--min-snr", "3",
                         "--out", str(out)])
    headroom_check.main()

    r = json.loads(out.read_text())
    assert r["max_snr_train_seconds"] == 240.0
    assert r["recommended_train_seconds"] == 45.0
    assert "cheapest" in r["recommendation_rule"]


def test_throughput_bound_reference_is_rejected(tmp_path, monkeypatch, capsys):
    """The real pilot failure: reference did 4 % of the baseline's steps."""
    import headroom_check
    out = tmp_path / "hr.json"
    state = {"recipe": None, "i": 0}
    monkeypatch.setattr(headroom_check.shutil, "copy",
                        lambda src, dst: state.update(
                            recipe=("reference" if "reference" in str(src)
                                    else "baseline"), i=0))

    def fake_train(workdir, run_dir, i, cfg, device):
        state["i"] += 1
        if state["recipe"] == "reference":
            # 240 s pilot numbers: 1352 steps, 7 epochs, worse than baseline
            return {"errored": False, "exit": 0, "val_acc": 0.72 + 0.01 * (state["i"] % 2),
                    "epochs_completed": 7, "steps": 1352, "train_seconds": 240,
                    "peak_vram_mb": 900}
        return {"errored": False, "exit": 0, "val_acc": 0.76 + 0.002 * (state["i"] % 2),
                "epochs_completed": 96, "steps": 33735, "train_seconds": 240,
                "peak_vram_mb": 438}

    monkeypatch.setattr(headroom_check, "run_training", fake_train)
    monkeypatch.setattr(sys, "argv",
                        ["headroom_check.py", "--profile",
                         str(ROOT / "profiles" / "pilot.yaml"),
                         "--budgets", "240", "--repeats", "2", "--cooldown", "0",
                         "--out", str(out)])
    rc = headroom_check.main()

    assert rc == 1, "a broken instrument must not return a recommendation"
    r = json.loads(out.read_text())
    b = r["budgets"][0]
    assert b["instrument_ok"] is False
    assert b["step_ratio"] < 0.25
    assert r["recommended_train_seconds"] is None
    assert "NO VALID MEASUREMENT" in capsys.readouterr().out
