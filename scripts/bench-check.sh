#!/bin/bash
# Run every bench tool the build, the tests and the scripts call, once and briefly, on the
# GPU, so a move or an import change is caught by running the tool, not by reading it.
#
#   bash scripts/bench-check.sh [--backend vulkan|opencl] [--timeout S]
#
# Each tool's output goes to notes/runs/bench-check-<backend>/<name>.log. Prints PASS/FAIL
# per tool and exits 1 if any failed. Run from the checkout root (or set NR_ROOT); XMX_DEVICE
# is passed through.
set -u
backend=vulkan
limit=900
while [ $# -gt 0 ]; do
  case "$1" in
    --backend) backend="$2"; shift 2 ;;
    --timeout) limit="$2"; shift 2 ;;
    *) echo "usage: $0 [--backend vulkan|opencl] [--timeout S]"; exit 2 ;;
  esac
done
R=${NR_ROOT:-$PWD}
[ -f "$R/pyproject.toml" ] || { echo "run from refs/dlss-on-arc or set NR_ROOT"; exit 2; }
cd "$R" || exit 2
out="notes/runs/bench-check-$backend"
mkdir -p "$out"
export NR_GPU_BACKEND="$backend" UV_PROJECT_ENVIRONMENT="$R/.venv"
pass=0
fail=0
run() {
  local name="$1"; shift
  if timeout "$limit" "$@" > "$out/$name.log" 2>&1; then
    echo "PASS $name"; pass=$((pass + 1))
  else
    echo "FAIL $name (see $out/$name.log)"; fail=$((fail + 1))
  fi
}
# what make test, make bench and CMake's ctest run
run frame_profile_tables uv run python src/bench/frame/frame_profile.py --tables
run half_probe uv run python src/bench/numerics/half_probe.py
run split_cost uv run python src/bench/frame/split_cost.py
# what scripts/ab-frame.sh, dump-isa.sh and iterate.sh run, cut short
run frame_profile uv run python src/bench/frame/frame_profile.py --size 256 256 --runs 1
run gemm_sustained uv run python src/bench/gemm/gemm_sustained.py 640x512x512 --seconds 1
run attention_curve uv run python src/bench/kernels/attention_curve.py --batches 240 960 --heads 16 --rounds 1 --stamps
# a cross-import inside a group, and one from src/layer (test_alloc imports live_rates)
run rank_ops uv run python src/bench/frame/rank_ops.py --size 256 256
run test_alloc uv run python src/layer/test_alloc.py
if [ "$backend" = vulkan ]; then
  run xmx_throughput uv run python src/bench/gemm/xmx_throughput.py 640x512x512
fi
echo "TOTAL pass=$pass fail=$fail"
[ "$fail" -eq 0 ]