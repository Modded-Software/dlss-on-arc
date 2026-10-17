#!/bin/bash
# Remove the stale line-count exception for src/ref/nr_frame.py.
#
# nr_frame.py is 448 lines after the split into nr_deps/nr_geometry/nr_features/
# nr_grade/nr_compose (all well under the default 512), so the 1093-line override in
# config/file_length_limits.yaml is dead. The config file is root-owned and only the
# root-gated editor may write it, so run this as root:
#
#   sudo bash scripts/remove-nr-frame-exemption.sh
#
# It refuses to run as anyone but root, and verifies the result afterwards.
set -euo pipefail

if [ "$(id -u)" != 0 ]; then
    echo "needs root: sudo bash $0" >&2
    exit 1
fi

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cfg="$root/config/file_length_limits.yaml"

if ! grep -q 'path: src/ref/nr_frame.py' "$cfg"; then
    echo "no src/ref/nr_frame.py override in $cfg; nothing to remove"
    exit 0
fi

workspace-yaml-edit remove "$cfg" overrides path=src/ref/nr_frame.py
workspace-yaml-edit validate "$cfg"
echo "removed; nr_frame.py is now governed by the default max_lines:"
workspace-yaml-edit get "$cfg" max_lines