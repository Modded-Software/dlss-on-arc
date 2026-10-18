#!/bin/bash
# The error counts a diary entry records, over a run's logs:
#   bash scripts/run-errors.sh LOG_OR_DIR [...]
# A directory contributes every *.log under it. Prints one count per pattern (FAILED, error,
# WARN, rejected, unhandled, unsupported, [PB] desync, [CRASH], openclmx:), then every
# distinct matching line with its count, so each hit can be accounted for rather than
# guessed at. Exit status: 0 when no FAILED, [CRASH] or desync line was found, 1 otherwise,
# 2 on a usage error.
set -u
[ $# -gt 0 ] || { echo "usage: $0 LOG_OR_DIR [...]"; exit 2; }
FILES=()
for a in "$@"; do
  if [ -d "$a" ]; then
    while IFS= read -r f; do FILES+=("$f"); done < <(find "$a" -name '*.log' -type f | sort)
  elif [ -f "$a" ]; then
    FILES+=("$a")
  else
    echo "no such log or directory: $a"
    exit 2
  fi
done
[ ${#FILES[@]} -gt 0 ] || { echo "no logs under: $*"; exit 2; }
echo "${#FILES[@]} log(s)"
PATTERNS=("FAILED" "error" "WARN" "rejected" "unhandled" "unsupported" "\[PB\] desync" "\[CRASH\]" "openclmx:")
FATAL=0
for p in "${PATTERNS[@]}"; do
  n=$(cat "${FILES[@]}" | grep -c -E "$p")
  printf '%-14s %s\n' "$p" "$n"
  case "$p" in FAILED | "\[CRASH\]" | "\[PB\] desync") [ "$n" = 0 ] || FATAL=1 ;; esac
done
echo "--- distinct matching lines (count, line)"
cat "${FILES[@]}" | grep -E "$(IFS='|'; echo "${PATTERNS[*]}")" | sort | uniq -c | sort -rn
[ "$FATAL" = 0 ]
