#!/bin/bash
# Run the GPU gate tests on one backend, each under a hard timeout, and print one line per
# test plus its first failure line. Never waits on anything unbounded.
#   bash scripts/gpu-test.sh --backend opencl|vulkan [--build] [--timeout SECS] [test_name ...]
# No names: every src/gpu/test_*.py except the two outside `make test` (see below).
# --build rebuilds first: work/libopenclmx.so for opencl, the whole dg2 build for vulkan.
# Runs on XMX_DEVICE=1 unless XMX_DEVICE is set (GPU 0 shares the box with everything else).
# Per-test logs: notes/runs/<backend>-test/<test>.log; summary: notes/runs/<backend>-test.log
# Exit status: 0 when every test passed, 1 otherwise, 2 on a usage error.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 2
LIMIT=180
BUILD=0
BACKEND=""
TESTS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --backend) shift; BACKEND="${1:-}" ;;
    --build) BUILD=1 ;;
    --timeout) shift; LIMIT="${1:-}" ;;
    *) TESTS+=("$1") ;;
  esac
  shift
done
case "$BACKEND" in
  opencl | vulkan) ;;
  *) echo "usage: $0 --backend opencl|vulkan [--build] [--timeout SECS] [test_name ...]"; exit 2 ;;
esac
DEVICE="${XMX_DEVICE:-1}"
DIR="notes/runs/$BACKEND-test"
SUM="notes/runs/$BACKEND-test.log"
mkdir -p "$DIR"
: > "$SUM"
# one line to the summary and the console, without piping into tee (the CI hook rejects a
# pipe whose exit status is not checked)
say() { printf '%s\n' "$*" >> "$SUM"; printf '%s\n' "$*"; }
CHANGED="$(git status --porcelain --untracked-files=no)"
DIRTY=""
[ -n "$CHANGED" ] && DIRTY="-dirty"
say "backend=$BACKEND XMX_DEVICE=$DEVICE timeout=${LIMIT}s commit=$(git rev-parse --short HEAD)$DIRTY"
if [ "$BUILD" = 1 ]; then
  if [ "$BACKEND" = opencl ]; then
    # the system linker: the workspace's own ld cannot load gcc's LTO plugin
    PATH=/usr/bin:/bin:$PATH timeout 300 make CC=/usr/bin/gcc work/libopenclmx.so > "$DIR/build.log" 2>&1
  else
    timeout 1200 bash scripts/dg2-build.sh > "$DIR/build.log" 2>&1
  fi
  BUILT=$?
  if [ "$BUILT" != 0 ] || { [ "$BACKEND" = vulkan ] && ! grep -q MAKE_DONE "$DIR/build.log"; }; then
    say "BUILD FAILED (exit $BUILT)"
    tail -n 40 "$DIR/build.log" >> "$SUM"
    tail -n 40 "$DIR/build.log"
    exit 1
  fi
  say "BUILD OK"
else
  # an earlier --build's log would otherwise sit beside this run's and read as its own
  rm -f "$DIR/build.log"
fi
if [ ${#TESTS[@]} -eq 0 ]; then
  for f in src/gpu/test_*.py; do
    t="$(basename "$f" .py)"
    case "$t" in
      # Deliberately outside `make test` (notes/reviewing.md): both load work/weights_ht.bin,
      # carved from the user's own nvngx_dlssnr.dll and gitignored, so neither can run on a
      # machine that has no such DLL - and test_layer cannot pass at all, its premise being an
      # artefact of the superseded dense-FP16 decode. Skipped here by default so the count
      # reflects the runnable suite; run either by naming it explicitly.
      test_attention_gpu | test_layer)
        say "SKIP $t: needs work/weights_ht.bin from the user's own DLL (outside make test)"
        continue ;;
    esac
    TESTS+=("$t")
  done
fi
for t in "${TESTS[@]}"; do
  # a mistyped name is a usage error, not a failing test
  [ -f "src/gpu/$t.py" ] || { say "NO SUCH TEST $t (src/gpu/$t.py)"; exit 2; }
done
PASS=0
FAIL=0
for t in "${TESTS[@]}"; do
  LOG="$DIR/$t.log"
  NR_GPU_BACKEND="$BACKEND" XMX_DEVICE="$DEVICE" UV_PROJECT_ENVIRONMENT="$PWD/.venv" \
    timeout -k 5 "$LIMIT" uv run python "src/gpu/$t.py" > "$LOG" 2>&1
  STATUS=$?
  if [ "$STATUS" = 0 ]; then
    PASS=$((PASS + 1))
    say "PASS $t"
  else
    FAIL=$((FAIL + 1))
    if [ "$STATUS" = 124 ] || [ "$STATUS" = 137 ]; then WHY="TIMEOUT after ${LIMIT}s"
    else WHY="$(grep -m 1 -E "^[A-Za-z]*Error|not implemented" "$LOG")"; fi
    say "FAIL $t: $WHY"
  fi
done
say "TOTAL pass=$PASS fail=$FAIL"
[ "$FAIL" = 0 ]
