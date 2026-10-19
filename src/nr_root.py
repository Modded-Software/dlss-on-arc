"""The checkout's root directory, found one way for every tool in the tree.

A tool used to count its own distance from the root (a fixed number of parent directories
above its own file), so moving the file one directory down pointed it one directory too high
and nothing said so. This finds the root by what is there, not by where the caller sits:

1. `NR_ROOT` in the environment - the variable the daemon, the layer and the launchers
   already use to name the checkout. It must be a directory.
2. Else the nearest directory, from this file upwards, that holds `pyproject.toml` - the
   checkout's own marker, the same rule WORKSPACE-CI's `ci.sh` uses for its root.
3. Else `RootNotFound`: it refuses rather than guessing a tree to run against.

Nothing but the standard library is imported, so a single-file tool can share this.
"""

import os
import pathlib

MARKER = "pyproject.toml"


class RootNotFound(RuntimeError):
    """Neither `NR_ROOT` nor a `pyproject.toml` above this module names the checkout."""


def repo_root(start=None):
    """The checkout root: `NR_ROOT` when set, else the first ancestor of `start` (this
    module's own file when not given) holding `pyproject.toml`."""
    named = os.environ.get("NR_ROOT", "").strip()
    if named:
        root = pathlib.Path(named).expanduser().resolve()
        if not root.is_dir():
            raise RootNotFound(f"NR_ROOT={named!r} is not a directory")
        return root
    here = pathlib.Path(start if start is not None else __file__).resolve()
    for candidate in (here, *here.parents):
        if (candidate / MARKER).is_file():
            return candidate
    raise RootNotFound(f"no {MARKER} in any directory above {here}; set NR_ROOT to the checkout")


ROOT = repo_root()

# The directory holding this module - `src/` in a checkout - so a caller that needs the
# project's Python sources, and not the checkout's data or build trees, asks for them
# rather than counting its own distance from the root.
SOURCE = pathlib.Path(__file__).resolve().parent


if __name__ == "__main__":
    print(ROOT)
