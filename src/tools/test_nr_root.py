#!/usr/bin/env python3
"""`nr_root` finds the checkout by what is there: NR_ROOT, else the nearest pyproject.toml.

A tool that counted its own depth below the root pointed at the wrong tree the moment it was
moved, and nothing failed until it read a file that was not there. These pin the order of the
rules and that an unresolvable root is an error, not a guess.
"""

import contextlib
import os
import pathlib
import tempfile
import unittest

import pytest

import nr_root


@contextlib.contextmanager
def without_nr_root():
    """The environment with NR_ROOT taken out, and put back afterwards."""
    saved = os.environ.pop("NR_ROOT", None)
    try:
        yield
    finally:
        if saved is not None:
            os.environ["NR_ROOT"] = saved


@contextlib.contextmanager
def with_nr_root(value):
    """NR_ROOT set to `value` for the block, and whatever was there restored after."""
    with without_nr_root():
        os.environ["NR_ROOT"] = value
        yield


@contextlib.contextmanager
def with_marker(value):
    """`nr_root.MARKER` set to `value` for the block, and restored after."""
    saved = nr_root.MARKER
    nr_root.MARKER = value
    try:
        yield
    finally:
        nr_root.MARKER = saved


class RootTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.tree = pathlib.Path(self.scratch.name).resolve()
        (self.tree / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
        self.deep = self.tree / "src" / "bench" / "frame"
        self.deep.mkdir(parents=True)
        self.tool = self.deep / "tool.py"
        self.tool.write_text("", encoding="utf-8")

    def tearDown(self):
        self.scratch.cleanup()

    def test_this_checkout_is_found_from_the_module_itself(self):
        with without_nr_root():
            root = nr_root.repo_root()
        assert (root / "pyproject.toml").is_file()
        assert (root / "src" / "nr_root.py").is_file()

    def test_the_nearest_marker_above_a_nested_file_wins(self):
        with without_nr_root():
            assert nr_root.repo_root(self.tool) == self.tree
            assert nr_root.repo_root(self.deep) == self.tree

    def test_depth_does_not_matter(self):
        deeper = self.deep / "one" / "two"
        deeper.mkdir(parents=True)
        with without_nr_root():
            assert nr_root.repo_root(deeper) == self.tree

    def test_nr_root_overrides_the_marker(self):
        other = self.tree / "elsewhere"
        other.mkdir()
        with with_nr_root(str(other)):
            assert nr_root.repo_root(self.tool) == other

    def test_nr_root_that_is_not_a_directory_is_refused(self):
        with (
            with_nr_root(str(self.tree / "missing")),
            pytest.raises(nr_root.RootNotFound, match="not a directory"),
        ):
            nr_root.repo_root(self.tool)

    def test_source_is_the_directory_holding_this_module(self):
        assert pathlib.Path(nr_root.__file__).resolve().parent == nr_root.SOURCE
        assert (nr_root.SOURCE / "layer" / "nr_daemon.py").is_file()

    def test_the_shebang_scripts_find_the_same_root(self):
        # scripts/ cannot import nr_root: setup.sh runs them with a bare interpreter, and a
        # release copy has neither pyproject.toml above it nor an editable install. Each
        # repeats the walk, so this holds the copies to the one answer.
        import build_release
        import fetch_dxvk
        import get_weights

        checkout = nr_root.SOURCE.parent
        assert (checkout / "pyproject.toml").is_file()
        assert checkout == build_release.ROOT
        assert checkout == fetch_dxvk.ROOT
        assert checkout == get_weights.ROOT

    def test_no_marker_anywhere_is_refused(self):
        # a marker name no directory above the scratch tree can hold
        unique = f"nr-root-marker-{os.getpid()}-{id(self)}.absent"
        with (
            without_nr_root(),
            with_marker(unique),
            pytest.raises(nr_root.RootNotFound, match="set NR_ROOT"),
        ):
            nr_root.repo_root(self.tool)


if __name__ == "__main__":
    unittest.main(verbosity=2)
