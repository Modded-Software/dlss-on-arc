#!/usr/bin/env python3
"""
nr_deps - the vendored MLX-DLSS NumPy half, and this tree's optional native pass.

The network itself is `nr_model`.  The surrounding contract (the 16-channel feature
assembly and the head-to-RGB composition) lives in MLX-DLSS's pure-numpy `features.py` /
`composition.py`, loaded by path from the clone in `work/` - they need no torch.
Apache-2.0, see work/mlx-dlss/LICENSE.

This module resolves both, once, and re-exports the symbols the rest of the package
builds on, so the loading order stays in one place.  `nr_frame` re-exports these names
for the callers that reach the frame through it.
"""

from __future__ import annotations

import os
import pathlib
import sys
import types

import nr_model

HERE = pathlib.Path(__file__).resolve().parent
ROOT = pathlib.Path(os.environ.get("NR_ROOT") or os.path.dirname(os.path.dirname(HERE)))

WEIGHTS = nr_model.default_weights()
_MLX = ROOT / "work" / "mlx-dlss" / "python" / "mlxdlss"

# The pure-numpy halves of MLX-DLSS. `temporal` and `motion_quality` import cv2 and
# PIL lazily - only for optical flow and for a processing scale other than 1 - so at
# scale 1 with engine motion they need neither.
MLX_NUMPY_MODULES = ("features", "composition", "motion_quality", "temporal")


def load_mlx_numpy_modules(names=MLX_NUMPY_MODULES):
    """Import MLX-DLSS's numpy modules without its torch-laden `__init__`."""
    if not _MLX.is_dir():
        raise SystemExit(
            f"missing {_MLX}\n  stage the pinned checkout with: uv run python scripts/fetch_deps.py"
        )
    if "mlxnp" not in sys.modules:
        package = types.ModuleType("mlxnp")
        package.__path__ = [str(_MLX)]
        sys.modules["mlxnp"] = package
    # The names are fixed, so import each module explicitly under the namespace package
    # that points at the MLX-DLSS tree; this skips its `__init__`, which imports torch.
    # The imports are what load the modules and the mapping below hands them out, so they
    # are used: an autofix that drops them as unused leaves `mlxnp` empty and every
    # importer of this file failing (KeyError 'mlxnp.features').
    import mlxnp.composition
    import mlxnp.features
    import mlxnp.motion_quality
    import mlxnp.temporal

    loaded = {
        "features": mlxnp.features,
        "composition": mlxnp.composition,
        "motion_quality": mlxnp.motion_quality,
        "temporal": mlxnp.temporal,
    }
    return [loaded[name] for name in names]


features_mod, composition_mod = load_mlx_numpy_modules(("features", "composition"))
NetworkGeometry = features_mod.NetworkGeometry

# The three noise channels depend on the extent and the frame index and on nothing else,
# and both callers copy the result into a slice rather than writing through it. In a live
# mode the frame index does not move - the daemon never sets one - so the same array was
# being rebuilt from four transcendentals per pixel on every frame, at 29-42 % of the
# whole feature assembly (notes/phase48). Memoised here rather than in `work/mlx-dlss`,
# which is a vendored dependency; the result is marked read-only so a future in-place
# user fails loudly instead of corrupting every later frame.
_raw_noise = features_mod.deterministic_noise
_noise_cache: dict = {}


def _cached_noise(height, width, frame_index=0):
    key = (int(height), int(width), int(frame_index))
    value = _noise_cache.get(key)
    if value is None:
        if len(_noise_cache) >= 8:  # extents change rarely; a swapchain resize
            _noise_cache.clear()  # or a scale change should not grow this
        value = _raw_noise(height, width, frame_index)
        value.flags.writeable = False
        _noise_cache[key] = value
    return value


features_mod.deterministic_noise = _cached_noise
deterministic_noise = _cached_noise
make_features = features_mod.make_features
PROFILES = features_mod.PROFILES
compose_head = composition_mod.compose_head
compose_detail = composition_mod.compose_detail
AutomaticMask = features_mod.AutomaticMask
half = features_mod.half
scaled_color = features_mod.scaled_color

try:
    import nr_image  # the same passes in C, when they are built
except ImportError:  # pragma: no cover - the pure-NumPy path is the point
    nr_image = None
