#!/bin/bash
# The OpenCL gate tests: scripts/gpu-test.sh with --backend opencl (see there).
#   bash scripts/opencl-test.sh [--build] [--timeout SECS] [test_name ...]
cd "$(git rev-parse --show-toplevel)" || exit 2
exec bash scripts/gpu-test.sh --backend opencl "$@"
