#!/usr/bin/env python3
"""The pinned build inputs must be complete, and nothing may clone around them.

res/native-pins.json is the only place this repository names a downloaded file
or a git commit for the build. A pin with a missing hash, a checkout without a
commit, or build instructions that tell a reader to `git clone` past the pin
would each make a fresh build unreproducible, and none of them would fail the
build itself - so they fail here.
"""

import json
import os
import subprocess
from pathlib import Path

ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
PINS = ROOT / "res" / "native-pins.json"
HEX = set("0123456789abcdef")
INSTRUCTIONS = ("README.md", "CMakeLists.txt", "scripts/get_weights.py", "src/ref/nr_deps.py")
CLONES = ("vulkan-headers", "MLX-DLSS")


def pins():
    return json.loads(PINS.read_text(encoding="utf-8"))


def is_hex(value, length):
    return isinstance(value, str) and len(value) == length and set(value) <= HEX


def test_schema():
    assert pins()["version"] == 1, "the pins file must name schema version 1"


def test_debs_are_pinned():
    for name, entry in pins()["debs"].items():
        assert entry["version"], f"{name}: no pinned version"
        assert is_hex(entry["sha256"], 64), f"{name}: sha256 is not 64 lowercase hex characters"
        assert entry["dest"], f"{name}: no destination tree"


def test_checkouts_are_pinned():
    for name, entry in pins()["checkouts"].items():
        assert entry["url"].startswith("https://"), f"{name}: no pinned URL"
        assert is_hex(entry["commit"], 40), f"{name}: commit is not 40 lowercase hex characters"
        assert entry["dest"], f"{name}: no destination"


def test_staged_trees_match_the_pins():
    data = pins()
    for name, entry in data["debs"].items():
        marker = ROOT / "work" / "deps" / entry["dest"] / f".pin.{name}"
        if marker.is_file():
            assert f"{name} {entry['sha256']}" in marker.read_text(encoding="utf-8"), (
                f"{name}: the staged tree is not the pinned one"
            )
    for name, entry in data["checkouts"].items():
        target = ROOT / "work" / entry["dest"]
        if (target / ".git").is_dir():
            got = subprocess.run(
                ["git", "-C", str(target), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
            )
            head = got.stdout.strip()
            assert got.returncode == 0 and head == entry["commit"], (
                f"{name}: the checkout is at {head or got.stderr.strip()}, not {entry['commit']}"
            )


def test_instructions_point_at_the_pins():
    for name in INSTRUCTIONS:
        path = ROOT / name
        if not path.is_file():
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "git clone" in line and any(repo in line for repo in CLONES):
                raise AssertionError(
                    f"{name}:{number} clones a checkout instead of naming it in res/native-pins.json"
                )


def main():
    test_schema()
    test_debs_are_pinned()
    test_checkouts_are_pinned()
    test_staged_trees_match_the_pins()
    test_instructions_point_at_the_pins()
    data = pins()
    print(
        "native pins: %d .deb tree(s) and %d checkout(s) pinned in res/native-pins.json"
        % (len(data["debs"]), len(data["checkouts"]))
    )


if __name__ == "__main__":
    main()
