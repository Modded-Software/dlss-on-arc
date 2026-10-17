#!/bin/bash
# Build everything for the Linux/Vulkan (Arc DG2) path.
#
# The pinned build inputs - the X11 and libpng header trees under work/deps, and the
# Vulkan-Headers and MLX-DLSS checkouts - are staged by scripts/fetch_deps.py from
# res/native-pins.json, which refuses any file whose SHA-256, or checkout whose commit,
# is not the pinned one. Everything this script used to set (PATH, CPATH, CC, the PNG
# flags, LAYER32) is the Makefile's own default, so a bare `make` and this script build
# alike. The host packages themselves come from config/system-deps.yaml (make init).
# Make variables still pass through the environment:
#   STAGED_SG=8 FFN_SG=16 WINDOW_SG=16 STAGED_WIDE_SUBS_M=8 STAGED_WIDE_WARPS=8
# Prints MAKE_DONE on success.
set -e
R=${NR_ROOT:-$PWD}   # the repo root; $0 is not a path under every launcher
[ -f "$R/Makefile" ] || { echo "run from refs/dlss-on-arc or set NR_ROOT"; exit 2; }
cd "$R"
uv run python scripts/fetch_deps.py
rm -f work/nr_frame.o
make -j16 all work/libopenclmx.so
echo MAKE_DONE