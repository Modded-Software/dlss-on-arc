#!/usr/bin/env python3
"""
nr_frame - one RGB frame in, one RGB frame out, through the recovered graph.

The network itself is `nr_model`.  The surrounding contract (the 16-channel feature
assembly with its deterministic noise, and the head-to-RGB composition) lives in
MLX-DLSS's pure-numpy `features.py` / `composition.py`, which are loaded by path from the
clone in `work/` - they need no torch.  Apache-2.0, see work/mlx-dlss/LICENSE.

    python3 src/ref/nr_frame.py IN.png OUT.png [--scale 1] [--profile standard]

This module is the frame's front door: it re-exports the pieces from the `nr_deps`,
`nr_geometry`, `nr_features`, `nr_grade` and `nr_compose` modules beside it, adds the
resident GPU backend and the two frame entry points (`run_head`, `run_frame`), and is
the command line.  Callers that reach the frame through `nr_frame` keep working; a new
caller can import the module it needs directly.
"""

from __future__ import annotations

import argparse
import pathlib
import time

import image_io
import nr_model
import numpy as np
from nr_compose import compose, compose_encode
from nr_deps import (
    PROFILES,
    WEIGHTS,
    AutomaticMask,
    NetworkGeometry,
    _raw_noise,
    compose_detail,
    compose_head,
    composition_mod,
    deterministic_noise,
    features_mod,
    half,
    load_mlx_numpy_modules,
    make_features,
    nr_image,
    scaled_color,
)
from nr_features import (
    BLEND_SCALE,
    HOLD_RAMP,
    apply_history,
    build_features,
    gate_formula,
    gate_table,
    history_weight,
    hold_floor,
    release_factor,
    release_slope,
)
from nr_geometry import (
    GRAPH_MINIMUM_EXTENT,
    VENDOR_MINIMUM_EXTENT,
    network_geometry,
    render_extent,
)
from nr_grade import (
    STYLE_GRADES,
    VENDOR_AUTOMATIC_MASK,
    VENDOR_DEFAULTS,
    controls,
    grade_for,
    style_grade,
    truncate_half,
)

__all__ = [
    "BLEND_SCALE",
    "GRAPH_MINIMUM_EXTENT",
    "HOLD_RAMP",
    "PROFILES",
    "STYLE_GRADES",
    "VENDOR_AUTOMATIC_MASK",
    "VENDOR_DEFAULTS",
    "VENDOR_MINIMUM_EXTENT",
    "WEIGHTS",
    "AutomaticMask",
    "NetworkGeometry",
    "ResidentBackend",
    "_raw_noise",
    "apply_history",
    "build_features",
    "compose",
    "compose_detail",
    "compose_encode",
    "compose_head",
    "composition_mod",
    "controls",
    "deterministic_noise",
    "features_mod",
    "gate_formula",
    "gate_table",
    "grade_for",
    "half",
    "history_weight",
    "hold_floor",
    "load_mlx_numpy_modules",
    "main",
    "make_features",
    "network_geometry",
    "nr_image",
    "release_factor",
    "release_slope",
    "render_extent",
    "run_frame",
    "run_head",
    "scaled_color",
    "style_grade",
    "truncate_half",
]


class ResidentBackend:
    """Runs the graph on the GPU, keeping activations in device buffers.

    Retains the most recently used extents, one by default. Evicted frames are
    closed, releasing their GPU allocations and captured commands. A caller must
    not keep using a frame after asking this backend for a different extent.
    """

    def __init__(self, weights_path=None, *, max_cached_frames=1):
        if not isinstance(max_cached_frames, int) or max_cached_frames < 1:
            raise ValueError("max_cached_frames must be a positive integer")
        import nr_frame_resident
        from xmxrt.recorder import Recorder

        self._module = nr_frame_resident
        self.runtime = Recorder()
        # halves stay halves: the device takes them as they are, and nothing is widened
        self.weights, _ = nr_model.load_logical(weights_path or WEIGHTS, keep_half=True)
        # one upload for as long as the backend is open: the extent changes under a knob
        # and the weights do not depend on it. Built on the first frame, not here, so that
        # constructing a backend still allocates nothing on the device.
        self.device_weights = None
        nr_model.FUSE_BRANCHED = True
        self._frames = {}
        self.split = (0.0, 0.0, 0.0)
        self.max_cached_frames = max_cached_frames

    def frame(self, height, width):
        key = (height, width)
        if key in self._frames:
            frame = self._frames.pop(key)
        else:
            while len(self._frames) >= self.max_cached_frames:
                self._frames.pop(next(iter(self._frames))).close()
            if self.device_weights is None:
                self.device_weights = self._module.DeviceWeights(self.runtime, self.weights)
            frame = self._module.ResidentFrame(self.runtime, self.device_weights, height, width)
        self._frames[key] = frame
        return frame

    def close(self):
        for frame in self._frames.values():
            frame.close()
        self._frames.clear()
        # closing means everything, including the weights; the next frame uploads them
        # again, which is what this cost before they were shared
        if self.device_weights is not None:
            self.device_weights.close()
            self.device_weights = None

    def input_view(self, height, width):
        """The mapped input of the frame at this extent, to build the features in: then
        `run_features` has nothing to copy."""
        return self.frame(height, width).input_view()

    def run_features(self, features):
        height, width = features.shape[:2]
        frame = self.frame(height, width)
        head = frame.run(features)
        self.split = frame.split  # host write, graph, host read
        return head


def run_head(
    model,
    color,
    *,
    profile="standard",
    frame_index=0,
    style_index=None,
    local_tone=None,
    local_structure=None,
    automatic_mask=None,
    control_mask=None,
    verbose=False,
):
    """The network alone: (H, W, 3) colour -> the cropped (H, W, 4) head.

    Everything that reaches the network is decided here - the conditioning scalars
    land in feature channels 10-14, so changing any of them costs a forward pass.
    The head can then be composed at any `intensity` for free.
    """
    height, width = color.shape[:2]
    geometry = network_geometry(width, height)
    values = controls(profile, style_index, local_tone, local_structure)
    features = make_features(
        color,
        frame_index=frame_index,
        geometry=geometry,
        automatic_mask=automatic_mask,
        control_mask=control_mask,
        **values,
    )
    if verbose:
        print(
            f"  network extent {geometry.network_width}x{geometry.network_height} "
            f"for output {width}x{height}",
            flush=True,
        )

    started = time.perf_counter()
    marks = [started]

    def progress(name):
        marks.append(time.perf_counter())
        if verbose:
            print(
                f"    {name:<12} {marks[-1] - marks[-2]:6.2f}s (total {marks[-1] - started:7.2f}s)",
                flush=True,
            )

    if isinstance(model, ResidentBackend):
        head = model.run_features(features)
    else:
        head = model.forward(features[None], progress=progress if verbose else None)[0]
    return geometry.crop(head), time.perf_counter() - started


def run_frame(
    model,
    color,
    *,
    intensity=1.0,
    detail_strength=1.0,
    colour_strength=1.0,
    detail_radius=4.0,
    control_mask=None,
    **head_options,
):
    """color: (H, W, 3) float32 in [0,1] -> (H, W, 3) float32, with the style's grade."""
    head, elapsed = run_head(model, color, control_mask=control_mask, **head_options)
    grade = grade_for(
        **controls(
            head_options.get("profile", "standard"),
            head_options.get("style_index"),
            head_options.get("local_tone"),
            head_options.get("local_structure"),
        )
    )
    output = compose(
        head,
        color,
        intensity=intensity,
        detail_strength=detail_strength,
        colour_strength=colour_strength,
        detail_radius=detail_radius,
        control_mask=control_mask,
        grade=grade,
    )
    return output, head, elapsed


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--weights", default=str(WEIGHTS))
    parser.add_argument("--size", help="resize input first, as HxW")
    parser.add_argument(
        "--profile",
        default="standard",
        choices=sorted(PROFILES),
        help="preset for style/tone/structure; overridden by the flags below",
    )
    parser.add_argument(
        "--style-index",
        type=int,
        help="vendor style index, normalised by 1/128 (0, 1, 2 are the presets)",
    )
    parser.add_argument(
        "--local-tone",
        type=float,
        help="local tone strength, 0..2 (the vendor's range; default 1.0)",
    )
    parser.add_argument(
        "--local-structure",
        type=float,
        help="local structure strength, 0..2 (vendor default 1.5)",
    )
    parser.add_argument(
        "--skin-structure",
        type=float,
        help="skin structure strength; enables automatic masking, -1 to follow local structure",
    )
    parser.add_argument(
        "--auto-mask",
        type=float,
        help="automatic-mask structure strength; -1 to follow local structure",
    )
    parser.add_argument(
        "--control-mask",
        help="per-pixel RGB mask: red intensity, green tone, blue structure",
    )
    parser.add_argument(
        "--intensity",
        type=float,
        default=1.0,
        help="blend of the model's picture against the source, 0..2; "
        "above 1 extrapolates past the model's own picture",
    )
    parser.add_argument(
        "--intensity-ladder",
        help="comma-separated intensities; renders once and writes one file each",
    )
    parser.add_argument("--detail-strength", type=float, default=1.0)
    parser.add_argument("--colour-strength", type=float, default=1.0)
    parser.add_argument("--detail-radius", type=float, default=4.0)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument(
        "--gpu", action="store_true", help="run the weight GEMMs on the Xe2 XMX units"
    )
    parser.add_argument(
        "--resident",
        action="store_true",
        help="run the whole graph on the GPU, activations never "
        "returning to the host (the fastest path)",
    )
    parser.add_argument(
        "--accel",
        action="store_true",
        help="use torch's SIMD half and E4M3 conversions (bit-identical)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.accel:
        import nr_accel

        nr_accel.install()
        print("accel   torch rounding, verified bit-identical", flush=True)

    if args.resident:
        started = time.perf_counter()
        model = ResidentBackend(args.weights)
        print(
            f"resident backend ready in {time.perf_counter() - started:.1f}s",
            flush=True,
        )

    backend = None
    if args.gpu and not args.resident:
        import nr_xmx

        backend = nr_xmx
        print(f"backend {backend.install()}", flush=True)

    size = None
    if args.size:
        size = tuple(int(part) for part in args.size.lower().split("x"))
    color = image_io.load(args.input, size=size)
    print(f"input {color.shape[1]}x{color.shape[0]}", flush=True)

    if not args.resident:
        started = time.perf_counter()
        model = nr_model.NeuralRenderingModel.from_safetensors(args.weights)
        print(
            f"weights loaded in {time.perf_counter() - started:.1f}s "
            f"({len(model.weights)} tensors)",
            flush=True,
        )

    mask = image_io.load(args.control_mask) if args.control_mask else None
    automatic = None
    if args.skin_structure is not None or args.auto_mask is not None:
        automatic = AutomaticMask(
            skin_structure_strength=(
                args.skin_structure if args.skin_structure is not None else -1.0
            ),
            automatic_mask_structure_strength=(
                args.auto_mask if args.auto_mask is not None else -1.0
            ),
        )

    head, elapsed = run_head(
        model,
        color,
        profile=args.profile,
        frame_index=args.frame_index,
        style_index=args.style_index,
        local_tone=args.local_tone,
        local_structure=args.local_structure,
        automatic_mask=automatic,
        control_mask=mask,
        verbose=args.verbose,
    )
    print(f"network {elapsed:.1f}s")
    if backend is not None:
        print(backend.report())
    print(f"head    min {head.min():+.4f} max {head.max():+.4f} sd {head.std():.4f}")
    grade = grade_for(
        **controls(args.profile, args.style_index, args.local_tone, args.local_structure)
    )
    if grade is not None:
        print(
            f"style grade: exposure x{grade[0]:.4f}, contrast {grade[1]:+.3f}, "
            f"saturation x{grade[2]:.3f}"
        )

    def emit(path, intensity):
        output = compose(
            head,
            color,
            intensity=intensity,
            detail_strength=args.detail_strength,
            colour_strength=args.colour_strength,
            detail_radius=args.detail_radius,
            control_mask=mask,
            grade=grade,
        )
        residual = output - color
        print(
            f"intensity {intensity:5.2f}  change mean|d| {np.abs(residual).mean():.5f} "
            f"max|d| {np.abs(residual).max():.5f}"
        )
        image_io.save(output, path)
        print(f"wrote {path}", flush=True)

    if args.intensity_ladder:
        stem = pathlib.Path(args.output)
        for value in (float(part) for part in args.intensity_ladder.split(",")):
            emit(str(stem.with_name(f"{stem.stem}_i{value:g}{stem.suffix}")), value)
    else:
        emit(args.output, args.intensity)


if __name__ == "__main__":
    main()
