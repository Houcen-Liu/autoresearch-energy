"""Per-device GPU energy sampler.

This is the PRIMARY energy-attribution channel of the experiment. EnergiBridge
supplies CPU package + DRAM energy and a session-level cross-check; NVML supplies
the per-device split that the whole design rests on.

Two quantities are recorded per device at ~10 Hz:

  energy_mj  nvmlDeviceGetTotalEnergyConsumption -- a monotonic hardware counter
             in millijoules since driver reload (Volta+). Deltas of this counter
             are EXACT integrals, immune to sampling aliasing. All reported
             E_prop / E_train values come from here.
  power_mw   instantaneous board power, kept only for the per-iteration time
             series and for sanity plots.
"""
from __future__ import annotations

import argparse
import csv
import threading
import time
from pathlib import Path

try:
    import pynvml
except ImportError:                                              # pragma: no cover
    pynvml = None

FIELDS = ["t_wall", "t_mono", "dev", "power_mw", "energy_mj",
          "util_gpu", "util_mem", "temp_c", "mem_used_mb"]


class NvmlSampler:
    def __init__(self, out_path: str | Path, hz: float = 10.0,
                 devices: list[int] | None = None):
        if pynvml is None:
            raise RuntimeError("pynvml not installed (pip install nvidia-ml-py)")
        self.out_path = Path(out_path)
        self.interval = 1.0 / hz
        pynvml.nvmlInit()
        n = pynvml.nvmlDeviceGetCount()
        self.devices = devices if devices is not None else list(range(n))
        self.handles = {d: pynvml.nvmlDeviceGetHandleByIndex(d) for d in self.devices}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.start_energy: dict[int, int] = {}
        self.end_energy: dict[int, int] = {}
        self._supports_counter = {}
        for d, h in self.handles.items():
            try:
                pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
                self._supports_counter[d] = True
            except pynvml.NVMLError:
                self._supports_counter[d] = False

    # ------------------------------------------------------------------ life
    def start(self) -> None:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.out_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._fh)
        self._writer.writerow(FIELDS)
        self.start_energy = {d: self._energy(d) for d in self.devices}
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        self.end_energy = {d: self._energy(d) for d in self.devices}
        self._fh.close()
        totals = {d: (self.end_energy[d] - self.start_energy[d]) / 1000.0
                  for d in self.devices if self._supports_counter[d]}
        pynvml.nvmlShutdown()
        return {"energy_j_per_device": totals,
                "counter_supported": dict(self._supports_counter)}

    # ------------------------------------------------------------------ loop
    def _energy(self, d: int) -> int:
        if not self._supports_counter.get(d):
            return 0
        return pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handles[d])

    def _loop(self) -> None:
        next_t = time.monotonic()
        while not self._stop.is_set():
            for d, h in self.handles.items():
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(h)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                    self._writer.writerow([
                        f"{time.time():.4f}", f"{time.monotonic():.4f}", d,
                        pynvml.nvmlDeviceGetPowerUsage(h), self._energy(d),
                        util.gpu, util.memory,
                        pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU),
                        int(mem.used / 1e6),
                    ])
                except pynvml.NVMLError:
                    continue
            self._fh.flush()
            next_t += self.interval
            time.sleep(max(0.0, next_t - time.monotonic()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="standalone NVML sampling")
    ap.add_argument("--out", default="nvml.csv")
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--hz", type=float, default=10)
    a = ap.parse_args()
    s = NvmlSampler(a.out, hz=a.hz)
    s.start()
    time.sleep(a.seconds)
    print(s.stop())
