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

exec vllm serve "$MODEL" \
  --revision "$REVISION" \
  --served-model-name "$ARM" \
  --port "$PORT" \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.94 \
  --kv-cache-dtype fp8 \
  --enforce-eager \
  --disable-log-requests \
  --max-num-seqs 1
# --max-num-seqs 1: the harness is strictly sequential. Continuous batching adds
# nondeterminism for no throughput benefit here, and costs KV cache we cannot
# spare on a 20 GB card.
