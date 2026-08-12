"""experiment-runner configuration: one run == one complete agent session.

Run with:
    sudo python experiment-runner/ experiment/RunnerConfig.py

Phase 1 (GPU factorial): 2 proposers x 2 patience x 2 loop budgets x 3 reps = 24
sessions, shuffled, >=120 s cooldown between sessions.
"""
from EventManager.Models.RunnerEvents import RunnerEvents
from EventManager.EventSubscriptionController import EventSubscriptionController
from ConfigValidator.Config.Models.RunTableModel import RunTableModel
from ConfigValidator.Config.Models.FactorModel import FactorModel
from ConfigValidator.Config.Models.RunnerContext import RunnerContext
from ConfigValidator.Config.Models.OperationType import OperationType
from ProgressManager.Output.OutputProcedure import OutputProcedure as output
from Plugins.Profilers.EnergiBridge import EnergiBridge

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

EXP_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_ROOT))

from measurement.energy_align import align            # noqa: E402
from measurement.nvml_sampler import NvmlSampler      # noqa: E402
from serving.manager import ensure as ensure_served   # noqa: E402

PROFILE = os.environ.get("AR_PROFILE", str(EXP_ROOT / "profiles" / "server.yaml"))
PATIENCE_LEVELS = {"greedy": 1, "patience3": 3}


class RunnerConfig:
    ROOT_DIR = Path(__file__).resolve().parent

    name: str = "autoresearch_energy_phase1"
    results_output_path: Path = ROOT_DIR.parent / "experiments"
    operation_type: OperationType = OperationType.AUTO
    time_between_runs_in_ms: int = 120_000          # >=120 s thermal cooldown

    def __init__(self):
        EventSubscriptionController.subscribe_to_multiple_events([
            (RunnerEvents.BEFORE_EXPERIMENT, self.before_experiment),
            (RunnerEvents.BEFORE_RUN,        self.before_run),
            (RunnerEvents.START_RUN,         self.start_run),
            (RunnerEvents.START_MEASUREMENT, self.start_measurement),
            (RunnerEvents.INTERACT,          self.interact),
            (RunnerEvents.STOP_MEASUREMENT,  self.stop_measurement),
            (RunnerEvents.STOP_RUN,          self.stop_run),
            (RunnerEvents.POPULATE_RUN_DATA, self.populate_run_data),
            (RunnerEvents.AFTER_EXPERIMENT,  self.after_experiment),
        ])
        self.run_table_model = None
        self.cfg = yaml.safe_load(Path(PROFILE).read_text())
        self.profiler = None
        self.sampler = None
        self.session_proc = None
        output.console_log(f"Loaded profile {PROFILE} (attribution="
                           f"{self.cfg.get('attribution')})")

    # ------------------------------------------------------------- run table
    def create_run_table_model(self) -> RunTableModel:
        proposer = FactorModel("proposer", ["dense", "moe"])
        patience = FactorModel("patience", ["greedy", "patience3"])
        budget = FactorModel("loop_budget", [10, 20])
        self.run_table_model = RunTableModel(
            factors=[proposer, patience, budget],
            repetitions=3,
            shuffle=True,                    # randomised, interleaved cell order
            data_columns=[
                "E_total_J", "E_prop_J", "E_train_J", "E_cpu_J",
                "E_wasted_J", "E_per_kept_J", "gap_fraction",
                "test_acc", "best_val_acc", "baseline_val_acc",
                "iterations", "kept", "reverted", "rejected", "errored",
                "wallclock_s", "prompt_tokens", "completion_tokens",
                "proposer_latency_s_mean", "no_progress", "alignment_ok",
            ],
        )
        return self.run_table_model

    # --------------------------------------------------------------- hooks
    def before_experiment(self) -> None:
        rc = subprocess.run([sys.executable, str(EXP_ROOT / "scripts" / "preflight.py"),
                             "--profile", PROFILE])
        if rc.returncode != 0:
            raise RuntimeError("preflight failed; refusing to start the experiment")
        manifest = EXP_ROOT.parent / "experiments" / "env_manifest.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(self._env_manifest(), indent=2))
        output.console_log("Preflight OK; environment manifest written")

    def before_run(self) -> None:
        """Cooldown is handled by time_between_runs_in_ms; verify we are back to idle."""
        time.sleep(5)

    def start_run(self, context: RunnerContext) -> None:
        cell = context.execute_run
        run_dir = context.run_dir
        run_dir.mkdir(parents=True, exist_ok=True)

        # Both arms cannot be resident on one 20 GB card, and the run table is
        # shuffled, so the served model has to follow the cell. Swapping costs
        # ~2 min of load; blocking the table by proposer instead would align the
        # factor with run order and hand thermal drift to the comparison (D11).
        if not os.environ.get("AR_STUB"):
            ensure_served(str(cell["proposer"]), self.cfg)

        seed = context.run_nr
        cmd = [sys.executable, "-m", "harness.agent_loop",
               "--profile", PROFILE,
               "--proposer", str(cell["proposer"]),
               "--patience", str(PATIENCE_LEVELS[str(cell["patience"])]),
               "--loop-budget", str(cell["loop_budget"]),
               "--run-dir", str(run_dir),
               "--seed", str(seed)]
        if os.environ.get("AR_STUB"):
            cmd.append("--stub")

        self._cmd = cmd
        self._run_dir = run_dir
        self._cell = cell
        output.console_log(f"run {context.run_nr}: {cell}")

    def start_measurement(self, context: RunnerContext) -> None:
        gpus = self.cfg["gpus"]
        devices = sorted({int(gpus["train"]), int(gpus["proposer"])}) \
            if str(gpus["train"]).isdigit() else [0]

        if self.cfg["energy"].get("nvml", True):
            self.sampler = NvmlSampler(self._run_dir / "nvml.csv",
                                       hz=self.cfg["energy"]["nvml_hz"],
                                       devices=devices)
            self.sampler.start()

        env = dict(os.environ, PYTHONPATH=str(EXP_ROOT))
        if self.cfg["energy"].get("energibridge", True):
            self.profiler = EnergiBridge(
                target_program=" ".join(self._cmd),
                out_file=self._run_dir / "energibridge.csv",
                sample_frequency=self.cfg["energy"]["energibridge_interval_ms"],
            )
            # The vendored plugin hardcodes requires_admin=True, which prepends
            # `sudo` and blocks on a password prompt forever on a machine where
            # you have no sudo. Whether elevation is actually needed depends on
            # the CPU: this host exposes no /sys/class/powercap/intel-rapl, yet
            # `energibridge --summary -- sleep 2` returns joules unprivileged.
            # Test it on YOUR host before assuming either way.
            self.profiler.requires_admin = bool(
                self.cfg["energy"].get("energibridge_sudo", True))
            self.profiler.start()
            self.session_proc = None
        else:
            self.session_proc = subprocess.Popen(
                self._cmd, cwd=str(EXP_ROOT), env=env,
                stdout=(self._run_dir / "harness.log").open("w"),
                stderr=subprocess.STDOUT)

    def interact(self, context: RunnerContext) -> None:
        """Block until the session finishes or the hard timeout fires."""
        expected = int(self._cell["loop_budget"]) * \
            (self.cfg["workload"]["train_seconds"] + 120) + 900   # + model swap
        deadline = time.time() + 3 * expected
        while time.time() < deadline:
            if self.session_proc is not None and self.session_proc.poll() is not None:
                return
            if (self._run_dir / "summary.json").exists():
                return
            time.sleep(10)
        output.console_log("[WARN] session hit the hard timeout; killing")
        if self.session_proc:
            self.session_proc.kill()
        (self._run_dir / "TIMEOUT").write_text("hard timeout")

    def stop_measurement(self, context: RunnerContext) -> None:
        if self.profiler:
            self.profiler.stop(wait=True)
        if self.sampler:
            totals = self.sampler.stop()
            (self._run_dir / "nvml_totals.json").write_text(
                json.dumps(totals, indent=2, default=str))

    def stop_run(self, context: RunnerContext) -> None:
        for f in self._run_dir.glob("model_*.pt"):     # keep only the winner
            if not f.name.endswith(self._best_suffix()):
                f.unlink(missing_ok=True)

    def populate_run_data(self, context: RunnerContext) -> Optional[Dict[str, Any]]:
        rd = self._run_dir
        if not (rd / "summary.json").exists():
            return {"no_progress": True, "alignment_ok": False}

        summary = json.loads((rd / "summary.json").read_text())
        gpus = self.cfg["gpus"]
        row: Dict[str, Any] = dict(summary)

        if self.cfg.get("attribution") == "per_device" and (rd / "nvml.csv").exists():
            e = align(rd, int(gpus["train"]), int(gpus["proposer"]))
            row.update({k: e[k] for k in
                        ("E_prop_J", "E_train_J", "E_wasted_J", "E_per_kept_J",
                         "gap_fraction", "wallclock_s", "alignment_ok")})
            row["E_total_J"] = e["E_gpu_total_J"]

        e_cpu = self._cpu_energy(rd)
        row["E_cpu_J"] = e_cpu
        if e_cpu and row.get("E_total_J"):
            row["E_total_J"] = row["E_total_J"] + e_cpu

        iters = rd / "iterations.csv"
        if iters.exists():
            import pandas as pd
            df = pd.read_csv(iters)
            row["prompt_tokens"] = int(df.prompt_tokens.sum())
            row["completion_tokens"] = int(df.completion_tokens.sum())
            row["proposer_latency_s_mean"] = float(df.proposer_latency_s.mean(skipna=True))
        return row

    def after_experiment(self) -> None:
        subprocess.run([sys.executable, str(EXP_ROOT / "analysis" / "aggregate.py"),
                        "--experiments-dir", str(self.results_output_path / self.name)])
        output.console_log("Experiment complete; tidy.csv written")

    # --------------------------------------------------------------- helpers
    def _best_suffix(self) -> str:
        try:
            log = json.loads((self._run_dir / "summary.json").read_text())
            return f"{log.get('best_iter', 0):03d}.pt"
        except Exception:                                          # noqa: BLE001
            return "___never___"

    def _cpu_energy(self, run_dir: Path) -> Optional[float]:
        f = run_dir / "energibridge.csv"
        if not f.exists():
            return None
        try:
            import pandas as pd
            df = pd.read_csv(f)
            cols = [c for c in df.columns if "PACKAGE_ENERGY" in c or "DRAM_ENERGY" in c]
            return float(sum(df[c].iloc[-1] - df[c].iloc[0] for c in cols)) or None
        except Exception:                                          # noqa: BLE001
            return None

    def _env_manifest(self) -> dict:
        def sh(*a):
            try:
                return subprocess.run(a, capture_output=True, text=True).stdout.strip()
            except Exception:                                      # noqa: BLE001
                return "n/a"
        return {
            "profile": PROFILE,
            "profile_content": self.cfg,
            "models": yaml.safe_load((EXP_ROOT / "serving" / "models.yaml").read_text()),
            "git_sha": sh("git", "rev-parse", "HEAD"),
            "python": sys.version,
            "pip_freeze": sh(sys.executable, "-m", "pip", "freeze").splitlines(),
            "nvidia_smi": sh("nvidia-smi"),
            "timestamp": time.time(),
        }

    experiment_path: Path = None
