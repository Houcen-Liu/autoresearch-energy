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

# Attention backend. vLLM prefers FlashInfer, which JIT-compiles CUDA kernels at
# startup and therefore needs nvcc -- the CUDA *toolkit*, not just the driver.
# A driver-only machine fails with "Could not find nvcc and default
# cuda_home='/usr/local/cuda' doesn't exist". TRITON_ATTN compiles through
# Triton's bundled ptxas and needs no toolkit.
#
# This is a fixed variable: whichever backend is used, BOTH arms must use it, or
# the arm contrast picks up a kernel-implementation contrast as well (D3).
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-TRITON_ATTN}"

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
