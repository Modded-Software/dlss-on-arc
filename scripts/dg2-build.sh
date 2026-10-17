#!/bin/bash
# Build everything for the Linux/Vulkan (Arc DG2) path.
#
# The repo-local dependency trees in work/deps (glslang, X11 and libpng headers) are what
# this image cannot supply itself, so the Makefile now prefers them when they are present
# and falls back to the distribution's own. Only the X11 headers still need fetching here;
# everything this script used to set (PATH, CPATH, CC, the PNG flags, LAYER32) is the
# Makefile's own default, so a bare `make` and this script build alike.
# Make variables still pass through the environment:
#   STAGED_SG=8 FFN_SG=16 WINDOW_SG=16 STAGED_WIDE_SUBS_M=8 STAGED_WIDE_WARPS=8
# Prints MAKE_DONE on success.
set -e
R=${NR_ROOT:-$PWD}   # the repo root; $0 is not a path under every launcher
[ -f "$R/Makefile" ] || { echo "run from refs/dlss-on-arc or set NR_ROOT"; exit 2; }
DEPS=$R/work/deps
if [ ! -f "$DEPS/x11/usr/include/X11/Xlib.h" ]; then
  mkdir -p "$DEPS" && cd "$DEPS"
  apt-get download libx11-dev x11proto-dev
  for d in libx11-dev_*.deb x11proto-dev_*.deb; do dpkg -x "$d" x11; done
fi
cd "$R"
rm -f work/nr_frame.o
make -j16 all work/libopenclmx.so
echo MAKE_DONE