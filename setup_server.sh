#!/usr/bin/env bash
# One-shot environment setup on the S2 workstation.
#
#   git clone <repo> ~/autoresearch-energy && cd ~/autoresearch-energy
#   ./setup_server.sh
#
# Verifies rather than assumes: every step that can silently half-succeed is
# checked, because on the pilot each one of those cost a wasted session.
set -euo pipefail

cd "$(dirname "$0")"
echo "=== autoresearch-energy: server setup ==="

command -v git >/dev/null || { echo "[FAIL] git not found"; exit 1; }
command -v python3 >/dev/null || { echo "[FAIL] python3 not found"; exit 1; }

PYV=$(python3 -c 'import sys;print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
echo "[ ok ] python $PYV"
python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' \
  || { echo "[FAIL] python 3.10+ required"; exit 1; }

if [ ! -d .venv ]; then
  echo "  creating .venv ..."
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install --quiet --upgrade pip
echo "  installing requirements (torch is large, be patient) ..."
pip install --quiet -r requirements.txt

python - <<'PY'
import torch
print(f"[ ok ] torch {torch.__version__}, cuda={torch.cuda.is_available()}, "
      f"devices={torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"       GPU {i}: {p.name}, {p.total_memory/1e9:.1f} GB")
PY

python - <<'PY'
try:
    import pynvml
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
    pynvml.nvmlShutdown()
    print("[ ok ] NVML energy counter readable (per-device attribution possible)")
except Exception as e:
    print(f"[FAIL] NVML energy counter unavailable: {e}")
    print("       Per-GPU energy attribution is the core of this design.")
    raise SystemExit(1)
PY

command -v energibridge >/dev/null \
  && echo "[ ok ] energibridge on PATH" \
  || echo "[warn] energibridge NOT on PATH — install it before Phase 1 (CPU/DRAM energy)"

echo "  running the test suite ..."
( cd experiment && python -m pytest tests/ -q ) \
  || { echo "[FAIL] tests failed — the code did not survive the transfer"; exit 1; }

echo "  preparing CIFAR-10 ..."
( cd experiment && python workload/prepare_cifar.py --data-dir ./data )

echo
echo "=== setup complete ==="
echo "Next: read SERVER_RUNBOOK.md, then"
echo "  cd experiment && python scripts/preflight.py --profile profiles/server.yaml"
echo
echo "Reminder: profiles/server.yaml assumes GPU 0 = training, GPU 1 = proposer."
echo "Check nvidia-smi and change the profile if the layout differs."
