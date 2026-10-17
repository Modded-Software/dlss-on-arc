#!/usr/bin/env python3
"""Stage the pinned build inputs into work/ and work/deps.

res/native-pins.json names every input this repository downloads and the bytes
or commit it must be: the three Ubuntu .deb trees the Linux/Arc build compiles
against (the X11 and libpng headers), and the two git checkouts (Vulkan-Headers
and MLX-DLSS). Nothing is taken on its name - a .deb is fetched at its pinned
version through apt and refused unless its SHA-256 is the pinned one, and a
checkout is refused unless it is at the pinned commit. What the host itself
provides (a C compiler, glslangValidator, the Vulkan loader) is declared in
config/system-deps.yaml and checked by `make init-check`; this script stages only
what is downloaded.

    uv run python scripts/fetch_deps.py            # stage anything missing or stale
    uv run python scripts/fetch_deps.py --check     # report only; never touches the network
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PINS = ROOT / "res" / "native-pins.json"
DEPS = ROOT / "work" / "deps"
WORK = ROOT / "work"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pins(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("version") != 1:
        raise ValueError(f"{path} names schema version {data.get('version')}, not 1")
    return data


def marker_path(target, name):
    """The marker naming one staged input, inside the tree it was extracted into.

    A destination can hold more than one input (both X11 packages extract into
    work/deps/x11), so the marker is named for the input, not the tree."""
    return target / f".pin.{name}"


def is_staged(target, name, identity):
    """Whether target already holds name at identity, by the marker fetch wrote there."""
    marker = marker_path(target, name)
    return marker.is_file() and marker.read_text(encoding="utf-8").strip() == f"{name} {identity}"


def git_head(target):
    got = subprocess.run(
        ["git", "-C", str(target), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    if got.returncode != 0:
        print(f"git rev-parse failed in {target}: {got.stderr.strip()}", file=sys.stderr)
        return ""
    return got.stdout.strip()


def apt_download(name, version, room):
    """Fetch name at the exact version with apt, so the host's own mirror is used."""
    got = subprocess.run(
        ["apt-get", "download", f"{name}={version}"],
        cwd=room,
        capture_output=True,
        text=True,
    )
    if got.returncode != 0:
        raise OSError(f"apt-get could not download {name}={version}: {got.stderr.strip()}")
    files = list(Path(room).glob(f"{name}_*.deb"))
    if len(files) != 1:
        raise OSError(f"apt-get download {name}={version} produced {len(files)} files")
    return files[0]


def fetch_dest(dest, members, deps=DEPS, check=False):
    """Stage one destination tree from its pinned .debs; return name -> 'current'/'staged'/'missing'.

    The tree is built from every package that names it in one pass, because more
    than one does (both X11 packages extract into work/deps/x11): extracting them
    one at a time would wipe the tree between packages."""
    target = deps / dest
    states = {}
    for name, entry in members:
        current = is_staged(target, name, entry["sha256"]) and target.is_dir()
        states[name] = "current" if current else "missing"
    if all(state == "current" for state in states.values()):
        return states
    if check:
        return states
    if shutil.which("dpkg-deb") is None:
        raise OSError("dpkg-deb is required to extract the pinned headers (Debian package dpkg)")
    if shutil.which("apt-get") is None:
        raise OSError("apt-get is required to fetch the pinned .deb trees")
    deps.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".native-", dir=deps) as room:
        archives = {}
        for name, entry in members:
            archive = apt_download(name, entry["version"], room)
            got = sha256(archive)
            if got != entry["sha256"]:
                raise ValueError(
                    f"{name}: the download's SHA-256 {got} is not the pinned {entry['sha256']}"
                )
            archives[name] = archive
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)
        for name, entry in members:
            subprocess.run(["dpkg-deb", "-x", str(archives[name]), str(target)], check=True)
            marker_path(target, name).write_text(f"{name} {entry['sha256']}\n", encoding="utf-8")
    return dict.fromkeys(states, "staged")


def fetch_checkout(name, entry, work=WORK, check=False):
    """Stage one pinned git checkout under work/; return 'current' or 'staged'."""
    target = work / entry["dest"]
    commit = entry["commit"]
    if is_staged(target, name, commit) and git_head(target) == commit:
        return "current"
    if check:
        return "missing"
    if target.exists() and not (target / ".git").is_dir():
        raise OSError(f"{target} exists and is not a checkout; move it aside and re-run")
    work.mkdir(parents=True, exist_ok=True)
    if not (target / ".git").is_dir():
        command = ["git", "clone", "--quiet"]
        if entry.get("tag"):
            command += ["--branch", entry["tag"], "--depth", "1"]
        command += [entry["url"], str(target)]
        if subprocess.run(command).returncode != 0:
            raise OSError(f"git clone of {entry['url']} failed; check the network")
    if git_head(target) != commit:
        if (
            subprocess.run(
                ["git", "-C", str(target), "fetch", "--quiet", "--depth", "1", "origin", commit]
            ).returncode
            != 0
        ):
            raise OSError(f"could not fetch {commit} for {name}")
        subprocess.run(
            [
                "git",
                "-C",
                str(target),
                "-c",
                "advice.detachedHead=false",
                "checkout",
                "--quiet",
                commit,
            ],
            check=True,
        )
    head = git_head(target)
    if head != commit:
        raise OSError(f"{name} is at {head}, not the pinned {commit}")
    marker_path(target, name).write_text(f"{name} {commit}\n", encoding="utf-8")
    return "staged"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--check", action="store_true", help="report only; never touch the network")
    parser.add_argument("--pins", type=Path, default=PINS)
    args = parser.parse_args(argv)
    try:
        pins = load_pins(args.pins)
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    grouped = {}
    for name, entry in pins["debs"].items():
        grouped.setdefault(entry["dest"], []).append((name, entry))
    results = []
    for dest in sorted(grouped):
        try:
            states = fetch_dest(dest, grouped[dest], check=args.check)
        except (OSError, ValueError) as error:
            print(f"ERROR: {error}", file=sys.stderr)
            return 3
        for name, state in states.items():
            results.append((state, name))
    for name, entry in pins["checkouts"].items():
        try:
            results.append((fetch_checkout(name, entry, check=args.check), name))
        except (OSError, ValueError) as error:
            print(f"ERROR: {error}", file=sys.stderr)
            return 3
    for state, name in results:
        print(f"  {state:8} {name}")
    if args.check and any(state != "current" for state, _ in results):
        print("\nthe pinned inputs are not all staged; run without --check", file=sys.stderr)
        return 1
    print("\nnative inputs are at their pinned versions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
