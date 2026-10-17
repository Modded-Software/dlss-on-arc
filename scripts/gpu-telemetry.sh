#!/bin/bash
# Probe the GPU telemetry this host exposes, or sample power over an interval, and log it.
#   bash scripts/gpu-telemetry.sh                  # probe: every readable field, both cards
#   bash scripts/gpu-telemetry.sh sample [SECS] [INTERVAL]
# Output: notes/runs/gpu-telemetry.log
# What the host cannot answer (xpu-smi is B-series only, intel_gpu_top aborts here) is part
# of the probe's output, so the reason lives beside the reading.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
OUT=notes/runs/gpu-telemetry.log
export UV_PROJECT_ENVIRONMENT=$PWD/.venv XMX_DEVICE=1
mkdir -p notes/runs
: > "$OUT"
uv run python src/gpu/gpu_telemetry.py "$@" 2>&1 | tee -a "$OUT"
# the tool's own failure must not be masked by tee's success, so the pipeline's first
# status is what this script exits with
status=${PIPESTATUS[0]}
if [ "$status" -ne 0 ]; then
  exit "$status"
fi