"""experiment-runner configuration: one run == one complete agent session.

Run with:
    sudo python experiment-runner/ experiment/RunnerConfig.py

STAGE 2b (sampling temperature): 4 levels x 3 reps = 12 sessions, MoE only.

Temperature is the only intra-model axis still open. Quantization cannot be
swept upward -- the MoE at int8 is ~30 GB and does not fit a 20 GB card -- and
sweeping downward (int3/int2) confounds the comparison with quality collapse.

WHY THIS MATTERS. Phase 1 had to be restarted because temperature 0, chosen for
reproducibility, made the search degenerate: proposals within a session had
0.982 mean pairwise similarity, one consecutive pair byte-identical, so ten
iterations delivered one idea at ten times its energy and the loop_budget factor
measured nothing (D22). That finding currently rests on a comparison between one
degenerate session and the post-fix runs. This sweep turns it into a measured
relationship, with n = 3 at each level including temperature 0 itself.

Three dependent variables are of interest here, and the first is not reported by
any comparable study:

  * proposal_similarity_mean -- search diversity, computed per session from the
    proposal files. Does 0.4 already escape degeneracy, or is 0.7 needed?
  * kept / max_val_acc_observed -- does diversity buy outcomes, or just waste?
  * E_per_kept_J -- what diversity costs.

NAMING. Stage 1 varies architecture with intra-model settings fixed (Phase 1 on
GPU, Phase 2 CPU-only); Stage 2 varies the intra-model settings themselves.
Stage 2b is reasoning; this file is Stage 2b.

The MoE is the only arm here: Phase 1 established it as both cheaper and more
accurate, and the question is now how to configure it, not which to choose.
Sessions are ~11 min at budget 10, so 12 runs cost ~2.5 h.

Stage 1 pinned reasoning off for both arms as a fixed variable (D9) and verified
by measurement that it stayed off: 0 thinking tokens across all 360 proposals.
Stage 2b promotes it to a factor against that clean baseline.

Patience and loop budget are FIXED here at the Phase-1 values that were cheapest
per kept mutation (greedy, budget 10): patience showed no effect at all, and
doubling the budget cost +107.6 % energy per kept mutation, so neither is worth
spending Stage-2b runs on.

Probe-measured cost (gate_evidence/probe_thinking_*.json): reasoning inflates
output 2.20x (dense) and 2.76x (MoE) with throughput unchanged, i.e. it costs
more tokens rather than slower tokens. Estimated ~5.5 h for the 12 runs.
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

# Fixed variables for Stage 2b, taken from the cheapest Phase-1 configuration.
STAGE2B_ARM = "moe"      # Phase 1 established the MoE as the better arm
STAGE2B_PATIENCE = 1
STAGE2B_LOOP_BUDGET = 10


class RunnerConfig:
    ROOT_DIR = Path(__file__).resolve().parent

    name: str = "autoresearch_energy_stage2b_temperature"
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
        temperature = FactorModel("temperature", [0.0, 0.4, 0.7, 1.0])
        self.run_table_model = RunTableModel(
            factors=[temperature],
            repetitions=3,
            shuffle=True,                    # randomised, interleaved cell order
            data_columns=[
                "E_total_J", "E_prop_J", "E_train_J", "E_cpu_J",
                "E_wasted_J", "E_per_kept_J", "gap_fraction",
                "test_acc", "best_val_acc", "baseline_val_acc",
                "iterations", "kept", "reverted", "rejected", "errored",
                "wallclock_s", "prompt_tokens", "completion_tokens",
                "proposer_latency_s_mean", "no_progress", "alignment_ok",
                # Stage-2b specific. thinking_tokens_total is the manipulation
                # check: a serving stack that accepts enable_thinking and
                # ignores it must not be mistaken for one that honours it.
                # Expect ~0 for thinking=off and thousands for thinking=on.
                "thinking_tokens_total", "max_val_acc_observed",
                # Stage-2b specific: search diversity is the point of the sweep.
                "proposal_similarity_mean", "proposal_similarity_max",
                "proposals_compared",
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
            ensure_served(STAGE2B_ARM, self.cfg)

        seed = context.run_nr
        cmd = [sys.executable, "-m", "harness.agent_loop",
               "--profile", PROFILE,
               "--proposer", STAGE2B_ARM,
               # fixed in Stage 2b -- see the module docstring
               "--patience", str(STAGE2B_PATIENCE),
               "--loop-budget", str(STAGE2B_LOOP_BUDGET),
               "--temperature", str(cell["temperature"]),
               "--run-dir", str(run_dir),
               "--seed", str(seed)]
        if os.environ.get("AR_STUB"):
            cmd.append("--stub")

        self._cmd = cmd
        self._run_dir = run_dir
        self._cell = cell
        output.console_log(f"run {context.run_nr}: {cell}")

    def _session_alive(self) -> bool:
        """Is a harness process running? Checked by pattern, because
        energibridge owns the child and we never see its pid."""
        return subprocess.run(["pgrep", "-f", "harness.agent_loop"],
                              capture_output=True).returncode == 0

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
            # energibridge spawns the session itself, so the session's cwd,
            # PYTHONPATH and output redirection have to be handed to it here.
            # Without this the harness never starts and the run silently
            # records energy for an idle machine.
            self._harness_log = (self._run_dir / "harness.log").open("w")
            self.profiler.stdout_path = str(self._run_dir / "harness.log")
            self.profiler.popen_kwargs = {
                "cwd": str(EXP_ROOT),
                "env": env,
                "stdout": self._harness_log,
                "stderr": subprocess.STDOUT,
            }
            self.profiler.start()

            # The session must actually be running. energibridge starting
            # successfully proves nothing about the program it wrapped: if the
            # harness dies on import, energibridge happily samples an idle
            # machine for the whole session and writes a perfectly valid CSV of
            # nothing. Fail in seconds instead.
            time.sleep(8)
            if not self._session_alive():
                self._harness_log.flush()
                tail = (self._run_dir / "harness.log").read_text()[-1500:]
                raise RuntimeError(
                    "the agent session did not start under energibridge.\n"
                    "harness.log tail:\n" + (tail or "(empty)"))
            self.session_proc = None
        else:
            self.session_proc = subprocess.Popen(
                self._cmd, cwd=str(EXP_ROOT), env=env,
                stdout=(self._run_dir / "harness.log").open("w"),
                stderr=subprocess.STDOUT)

    def interact(self, context: RunnerContext) -> None:
        """Block until the session finishes or the hard timeout fires."""
        # loop_budget is fixed in Stage 2b, not a run-table factor. Reasoning
        # roughly triples proposal time (2.20x dense, 2.76x MoE by probe), so
        # the per-iteration allowance is generous enough to cover the thinking
        # arm without letting a genuinely hung session run forever.
        per_iter = self.cfg["workload"]["train_seconds"] + 240
        expected = STAGE2B_LOOP_BUDGET * per_iter + 900   # + model swap
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

        # MANIPULATION CHECK. Verify the level actually took, and -- more
        # usefully -- surface the diversity it produced, since a temperature
        # that is accepted but ineffective looks exactly like a null result.
        want = float(self._cell.get("temperature", -1))
        got = row.get("temperature")
        sim = row.get("proposal_similarity_mean")
        if got is not None and abs(float(got) - want) > 1e-6:
            output.console_log(f"*** temperature mismatch: asked {want}, "
                               f"session recorded {got}")
            row["valid"] = False
            row["invalid_reason"] = f"temperature {got} != requested {want}"
        else:
            output.console_log(
                f"    temperature={want}, proposal similarity="
                f"{sim if sim is not None else 'n/a'} "
                f"({'DEGENERATE' if (sim or 0) > 0.9 else 'diverse'})")

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
