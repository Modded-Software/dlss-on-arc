#!/bin/bash
# Interleaved A/B of the working tree against HEAD, measured on the frame:
#   bash scripts/ab-frame.sh --label NAME [--backend opencl|vulkan] [--rounds N]
#        [--size "H W"]... [--runs R] [--seconds S] [--calls C] [FILE ...]
# With no FILE named, every tracked file changed against HEAD is compared. Each round builds HEAD's copy of
# them ("head"), profiles every size, then the working copy ("new"), and profiles again, so a
# clock or thermal drift lands on both sides. The working copy is saved first and put back on
# ANY exit - normal, error or interrupt - then rebuilt and checked byte-identical, so an
# aborted run never leaves HEAD's (or anything else's) source in the tree.
# Defaults: opencl, 3 rounds, 768 1280 and 576 1024, --runs 8 --seconds 10 --calls 40,
# XMX_DEVICE=1. Output: notes/runs/ab-NAME/ (profile logs, build logs, summary.txt).
# Exit status: 0 when every build and profile succeeded and the tree was restored.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 2
LABEL=""
BACKEND=opencl
ROUNDS=3
RUNS=8
SECONDS_EACH=10
CALLS=40
SIZES=()
FILES=()
while [ $# -gt 0 ]; do
  case "$1" in
    --label) shift; LABEL="${1:-}" ;;
    --backend) shift; BACKEND="${1:-}" ;;
    --rounds) shift; ROUNDS="${1:-}" ;;
    --size) shift; SIZES+=("${1:-}") ;;
    --runs) shift; RUNS="${1:-}" ;;
    --seconds) shift; SECONDS_EACH="${1:-}" ;;
    --calls) shift; CALLS="${1:-}" ;;
    *) FILES+=("$1") ;;
  esac
  shift
done
usage() { echo "usage: $0 --label NAME [--backend opencl|vulkan] [--rounds N] [--size \"H W\"]... [--runs R] [--seconds S] [--calls C] [FILE ...]"; exit 2; }
[ -n "$LABEL" ] || usage
case "$BACKEND" in opencl | vulkan) ;; *) usage ;; esac
[ ${#SIZES[@]} -gt 0 ] || SIZES=("768 1280" "576 1024")
if [ ${#FILES[@]} -eq 0 ]; then
  while IFS= read -r f; do FILES+=("$f"); done < <(git diff --name-only HEAD)
fi
[ ${#FILES[@]} -gt 0 ] || { echo "nothing differs from HEAD: no A/B to run"; exit 2; }
OUT="notes/runs/ab-$LABEL"
mkdir -p "$OUT/new" "$OUT/head"
SUM="$OUT/summary.txt"
: > "$SUM"
say() { printf '%s\n' "$*" >> "$SUM"; printf '%s\n' "$*"; }
for f in "${FILES[@]}"; do
  git cat-file -e "HEAD:$f" || { echo "$f has no HEAD version; A/B needs one"; exit 2; }
  mkdir -p "$OUT/new/$(dirname "$f")" "$OUT/head/$(dirname "$f")"
  cp "$f" "$OUT/new/$f"
  git show "HEAD:$f" > "$OUT/head/$f"
done
export UV_PROJECT_ENVIRONMENT="$PWD/.venv" XMX_DEVICE="${XMX_DEVICE:-1}" NR_GPU_BACKEND="$BACKEND"
say "ab-frame $LABEL: backend=$BACKEND XMX_DEVICE=$XMX_DEVICE rounds=$ROUNDS runs=$RUNS seconds=$SECONDS_EACH HEAD=$(git rev-parse --short HEAD)"
say "files: ${FILES[*]}"

use() { for f in "${FILES[@]}"; do cp "$OUT/$1/$f" "$f"; done; }
OK=1
restore() {
  trap - EXIT INT TERM HUP
  use new
  local bad=0
  for f in "${FILES[@]}"; do cmp -s "$f" "$OUT/new/$f" || { say "RESTORE MISMATCH $f"; bad=1; }; done
  timeout 1200 bash scripts/dg2-build.sh > "$OUT/build-restore.log" 2>&1
  grep -q MAKE_DONE "$OUT/build-restore.log" || { say "RESTORE BUILD FAILED (see $OUT/build-restore.log)"; bad=1; }
  if [ "$bad" = 0 ]; then say "restored: working copy back, byte-identical, rebuilt"; else OK=0; fi
}
trap 'restore; [ "$OK" = 1 ]; exit $?' EXIT
trap 'say "INTERRUPTED"; OK=0; exit 130' INT TERM HUP

# "99.64" -> 9964, so means are integer arithmetic (two decimals in frame_profile's output)
centi() { local v="$1"; local w="${v%%.*}" d="${v#*.}"; d="${d}00"; echo $((10#$w * 100 + 10#${d:0:2})); }
declare -A TOTAL COUNT
for round in $(seq 1 "$ROUNDS"); do
  for v in head new; do
    use "$v"
    if ! timeout 1200 bash scripts/dg2-build.sh > "$OUT/build-$v-$round.log" 2>&1 \
       || ! grep -q MAKE_DONE "$OUT/build-$v-$round.log"; then
      say "BUILD FAILED $v round $round (see $OUT/build-$v-$round.log)"
      OK=0
      exit 1
    fi
    for sz in "${SIZES[@]}"; do
      tag="${sz// /x}"
      LOG="$OUT/prof-$v-$round-$tag.log"
      # shellcheck disable=SC2086  # the size is two words on purpose
      if ! timeout 900 uv run python src/bench/frame/frame_profile.py --size $sz --runs "$RUNS" \
           --seconds "$SECONDS_EACH" --calls "$CALLS" > "$LOG" 2>&1; then
        say "PROFILE FAILED $v round $round $tag (see $LOG)"
        OK=0
        continue
      fi
      dev="$(grep -m 1 "device total" "$LOG")"
      ms="$(set -- $dev; echo "$4")"
      say "$round $v $tag device $ms"
      TOTAL["$v $tag"]=$(( ${TOTAL["$v $tag"]:-0} + $(centi "$ms") ))
      COUNT["$v $tag"]=$(( ${COUNT["$v $tag"]:-0} + 1 ))
    done
  done
done
for sz in "${SIZES[@]}"; do
  tag="${sz// /x}"
  for v in head new; do
    n=${COUNT["$v $tag"]:-0}
    [ "$n" -gt 0 ] || continue
    m=$(( TOTAL["$v $tag"] / n ))
    say "mean $tag $v $((m / 100)).$(printf '%02d' $((m % 100))) ms over $n"
  done
done
