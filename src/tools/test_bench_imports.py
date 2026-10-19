#!/usr/bin/env python3
"""Every bench tool's imports of this project's own modules resolve, to this checkout.

The bench tools import each other and the runtime by bare name (`import live_rates`,
`from opencl_gemm8_slm import ...`, `import nr_frame_resident`), found through the source
roots the editable install puts on `sys.path` (`dev-mode-dirs` in pyproject.toml). mypy is
told to ignore a missing import, and most of these tools need a GPU to run at all, so a tool
moved into another directory - or a group left off `dev-mode-dirs` - fails only when someone
next runs it. This parses every bench file and resolves each import of a project module.

It also holds the bench tools to `nr_root.ROOT` for the checkout root: a tool that counts
its own depth (a fixed number of parent directories, `os.path.dirname` chains) points at
the wrong tree the moment it moves.
"""

import ast
import pathlib
import sys
import unittest

from nr_root import ROOT

BENCH = ROOT / "src" / "bench"
SKIP = {"__pycache__", ".git", ".venv", "work", "build", "notes", "node_modules"}


def project_modules():
    """Top-level module names this checkout defines under src/ and scripts/: a .py file's
    stem, or a package directory's name."""
    names = {}
    for base in (ROOT / "src", ROOT / "scripts"):
        for path in base.rglob("*.py"):
            if SKIP.intersection(path.relative_to(ROOT).parts):
                continue
            if path.name == "__init__.py":
                names.setdefault(path.parent.name, path.parent)
            else:
                names.setdefault(path.stem, path)
    return names


def bench_files():
    return sorted(p for p in BENCH.rglob("*.py") if "__pycache__" not in p.parts)


def imported_names(path):
    """The top-level names an absolute import in `path` binds, with each line number."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0], node.lineno
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module.split(".")[0], node.lineno


class BenchImportTests(unittest.TestCase):
    def test_bench_tools_are_grouped(self):
        assert bench_files(), f"no bench tools under {BENCH}"
        loose = [p.name for p in BENCH.glob("*.py")]
        assert not loose, f"bench tools outside a group directory: {loose}"

    def test_project_imports_resolve_to_this_checkout(self):
        local = project_modules()
        roots = [pathlib.Path(entry or ".").resolve() for entry in sys.path]
        unresolved = []
        for path in bench_files():
            for name, line in imported_names(path):
                if name not in local:
                    continue  # the standard library or a third-party package
                expected = local[name]
                # the directory the import is found under: a .py file's directory, or the
                # directory holding a package directory
                holder = expected.parent
                # a plain `import name` finds the file under the first root that holds it,
                # so only this checkout's directory may be on sys.path with that name
                rivals = [
                    root
                    for root in roots
                    if root != holder
                    and ((root / f"{name}.py").is_file() or (root / name / "__init__.py").is_file())
                ]
                if holder not in roots or rivals:
                    where = path.relative_to(ROOT)
                    unresolved.append(
                        f"{where}:{line}: {name} -> root {holder} on sys.path "
                        f"{holder in roots}, rival copies {rivals}"
                    )
        assert not unresolved, "\n".join(unresolved)

    def test_bench_tools_take_the_root_from_nr_root(self):
        counted = []
        for path in bench_files():
            for number, text in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if "parents[" in text or "dirname(os.path.dirname" in text:
                    counted.append(f"{path.relative_to(ROOT)}:{number}: {text.strip()}")
        assert not counted, "use nr_root.ROOT:\n" + "\n".join(counted)


if __name__ == "__main__":
    unittest.main(verbosity=2)
