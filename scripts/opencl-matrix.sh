#!/usr/bin/env bash
# Every src/gpu test on OpenCL with a 600 s limit each: scripts/gpu-test.sh (see there).
cd "$(git rev-parse --show-toplevel)" || exit 2
exec bash scripts/gpu-test.sh --backend opencl --timeout 600 "$@"
