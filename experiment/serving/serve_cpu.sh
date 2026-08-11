#!/usr/bin/env bash
# Phase 2: CPU-only proposer serving via llama.cpp, training unchanged on GPU 0.
#
#   ./serve_cpu.sh dense 8000
#
# NOTE (construct change, must be reported): with the proposer on CPU, E_prop is a
# RAPL package measurement that also contains OS work and the training process's
# host-side work. Subtract the idle+training-host baseline from idle_baseline.py
# and report both raw and corrected values.
set -euo pipefail

ARM="${1:?usage: serve_cpu.sh <dense|moe> <port>}"
PORT="${2:?usage: serve_cpu.sh <dense|moe> <port>}"
GGUF="${GGUF_PATH:?set GGUF_PATH to the pinned Q4_K_M gguf}"
THREADS="${THREADS:-$(nproc)}"

exec llama-server \
  --model "$GGUF" \
  --alias "$ARM" \
  --port "$PORT" \
  --ctx-size 16384 \
  --threads "$THREADS" \
  --n-gpu-layers 0 \
  --temp 0 \
  --parallel 1
