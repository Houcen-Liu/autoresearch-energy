#!/usr/bin/env bash
# Phase 1 proposer serving. One model per invocation, pinned to the PROPOSER GPU.
#
#   ./serve_vllm.sh dense 8000
#   ./serve_vllm.sh moe   8001
#
# Both arms MUST use identical flags apart from the model itself -- see
# EXPERIMENT_PLAN.md D3 (fairness rule).
set -euo pipefail

ARM="${1:?usage: serve_vllm.sh <dense|moe> <port>}"
PORT="${2:?usage: serve_vllm.sh <dense|moe> <port>}"
PROPOSER_GPU="${PROPOSER_GPU:-1}"

# Fill from serving/models.yaml once revisions are pinned (week 1, day 1).
case "$ARM" in
  dense) MODEL="${DENSE_MODEL:?set DENSE_MODEL to the pinned int4 repo}" ;;
  moe)   MODEL="${MOE_MODEL:?set MOE_MODEL to the pinned int4 repo}" ;;
  *) echo "unknown arm: $ARM" >&2; exit 2 ;;
esac
REVISION="${REVISION:?set REVISION to the pinned commit sha}"

export CUDA_VISIBLE_DEVICES="$PROPOSER_GPU"

# Tunables. Override from the environment rather than editing this file, so the
# same script serves both arms with identical settings (the D3 fairness rule).
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.94}"
# `--disable-log-requests` was removed in newer vLLM; request logging is on by
# default and harmless. Add version-specific flags here instead of inline.
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"

# CUDA toolkit for FlashInfer's JIT.
#
# vLLM's preferred attention backend (FlashInfer) compiles kernels at startup and
# needs nvcc plus the CUDA headers. A machine with only the driver fails with
#   RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda'
# even though `pip list` shows nvidia-cuda-nvcc installed: the pip toolkit lives
# under site-packages/nvidia and is neither on PATH nor pointed to by CUDA_HOME.
#
# vLLM 0.27 no longer honours VLLM_ATTENTION_BACKEND (it warns "Unknown vLLM
# environment variable") and exposes no --attention-backend flag, so selecting a
# JIT-free backend is not an option -- the toolkit has to be found instead.
if [ -z "${CUDA_HOME:-}" ] && ! command -v nvcc >/dev/null 2>&1; then
  CUDA_HOME="$(python - <<'PY'
import glob, os, site, sys
roots = site.getsitepackages() + [site.getusersitepackages()]
for r in roots:
    for nvcc in glob.glob(os.path.join(r, "nvidia", "**", "bin", "nvcc"), recursive=True):
        home = os.path.dirname(os.path.dirname(nvcc))
        # Needs headers too, not just the compiler.
        if glob.glob(os.path.join(home, "include", "cuda_runtime.h")):
            print(home); sys.exit(0)
        print(home); sys.exit(0)
PY
)"
  if [ -n "$CUDA_HOME" ]; then
    export CUDA_HOME
    export PATH="$CUDA_HOME/bin:$PATH"
    echo "[serve] CUDA_HOME -> $CUDA_HOME (pip toolkit)"
  else
    echo "[serve] WARNING: no nvcc found. FlashInfer will fail to JIT its kernels."
    echo "[serve]          pip install nvidia-cuda-nvcc  (or flashinfer-jit-cache)"
  fi
fi

# FlashInfer links its JIT-built kernels against libcudart and the driver stub.
# The pip CUDA wheels install libraries in cu13/lib (not lib64, which the build
# file hardcodes), ship only the version-suffixed libcudart.so.13, and ship no
# stubs/ directory at all -- the driver library lives in the system path. Three
# symlinks, created once, are the difference between "cannot find -lcudart" and
# a working server. Idempotent, so this is safe on every start.
python - <<'PY'
import glob, os, site
for root in site.getsitepackages():
    cu = os.path.join(root, "nvidia", "cu13")
    lib = os.path.join(cu, "lib")
    if not os.path.isdir(lib):
        continue
    lib64 = os.path.join(cu, "lib64")
    if not os.path.exists(lib64):
        os.symlink("lib", lib64)
    for stem in ("libcudart",):
        plain = os.path.join(lib, stem + ".so")
        versioned = sorted(glob.glob(os.path.join(lib, stem + ".so.*")))
        if versioned and not os.path.exists(plain):
            os.symlink(os.path.basename(versioned[-1]), plain)
    stubs = os.path.join(lib, "stubs")
    os.makedirs(stubs, exist_ok=True)
    target = os.path.join(stubs, "libcuda.so")
    if not os.path.exists(target):
        for cand in ("/usr/lib/x86_64-linux-gnu/libcuda.so.1",
                     "/usr/lib64/libcuda.so.1"):
            if os.path.exists(cand):
                os.symlink(cand, target)
                break
    break
PY

# Persist the JIT cache so the compile is paid once, not on every model swap.
# Without this each arm swap costs minutes of kernel compilation, which would
# land inside the measured window.
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-$HOME/.cache/flashinfer}"
mkdir -p "$FLASHINFER_WORKSPACE_BASE"

exec vllm serve "$MODEL" \
  --revision "$REVISION" \
  --served-model-name "$ARM" \
  --port "$PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --kv-cache-dtype fp8 \
  --enforce-eager \
  --max-num-seqs 1 \
  $VLLM_EXTRA_ARGS
# --max-num-seqs 1: the harness is strictly sequential. Continuous batching adds
# nondeterminism for no throughput benefit here, and costs KV cache we cannot
# spare on a 20 GB card.
