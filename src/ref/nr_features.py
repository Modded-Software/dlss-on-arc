#!/usr/bin/env python3
"""
nr_features - the 16-channel feature assembly and the temporal gate.

`build_features` mirrors the frame onto the network extent and rounds its colour into the
feature channels; `apply_history` writes a previous output into channels 7-9.  The gate
helpers are the model's history weight and the two corrections this tree adds to it (a
per-pixel floor from unchanged game pixels and a release where the game changed them).
"""

from __future__ import annotations

import numpy as np
from nr_deps import deterministic_noise, half, make_features, nr_image, scaled_color

# half(blend_scale) of the recovered package, `notes/phase12-temporal.md`.
BLEND_SCALE = 0.73974609375


def build_features(
    colour,
    *,
    geometry,
    history=None,
    frame_index=0,
    normalized_style=0.0,
    local_tone_strength=1.0,
    local_structure_strength=1.0,
    out=None,
):
    """`make_features` with the history folded in, natively where that is available.

    The two steps are one pass in C: the mirror onto the network extent, the three FP16
    roundings of `scaled_color` and the history that replaces channels 7-9 all happen
    while each pixel is in a register. In NumPy they are a full-frame gather each, and
    feature assembly was the largest single host pass in the frame (`notes/phase57`).

    Only the plain recipe goes native - no control mask and no automatic mask - which is
    the one a game uses; anything else falls back and is bit-identical either way.

    `out`, when given, is where the features are written - the graph's own mapped input
    (`ResidentBackend.input_view`), float32 or half - and is returned.
    """
    values = {
        "normalized_style": normalized_style,
        "local_tone_strength": local_tone_strength,
        "local_structure_strength": local_structure_strength,
    }
    native = None
    if nr_image is not None:
        controls = np.array(
            [
                half(normalized_style),
                half(local_tone_strength),
                half(local_structure_strength),
                -1.0,
                -1.0,
            ],
            np.float32,
        )
        native = nr_image.features(
            colour,
            geometry.source_rows(),
            geometry.source_columns(),
            deterministic_noise(geometry.network_height, geometry.network_width, frame_index),
            controls,
            history=history,
            out=out,
        )
    if native is not None:
        return native
    features = make_features(colour, geometry=geometry, frame_index=frame_index, **values)
    if history is not None:
        apply_history(features, history, geometry)
    if out is not None:
        # NumPy's half rounding is the GPU's to_half, to the bit (test_input_fp16.py)
        np.copyto(out, features, casting="unsafe")
        return out
    return features


def apply_history(features, history, geometry=None):
    """Put a previous output into feature channels 7-9, reprojected by the identity.

    MLX-DLSS's `make_temporal_features` samples the history along engine motion
    vectors. A Vulkan layer at `vkQueuePresentKHR` has no motion vectors - it sees a
    finished frame and nothing that produced it - so the reprojection is the identity,
    which `sample_history` computes bit-exactly at pixel centres (checked: max |delta|
    is 0 over a random image, so the five-tap collapses to its middle tap).

    That is the correct history wherever the scene stood still, and the *wrong* history
    wherever it moved. Rejecting the wrong one is the learned gate's job, and it does
    discriminate: 0.705 with correct history against 0.032 with wrong motion
    (`notes/phase12-temporal.md`). The first-frame layout already writes the current
    colour into these channels, so this is a drop-in replacement at the same scaling.
    """
    history = np.asarray(history, dtype=np.float32)
    if history.ndim != 3 or history.shape[2] != 3:
        raise ValueError("history must be (height, width, 3)")
    if geometry is not None and not geometry.is_identity:
        history = history[geometry.source_rows()[:, None], geometry.source_columns()[None, :], :]
    if history.shape[:2] != features.shape[:2]:
        raise ValueError("history must match the feature extent")
    features[..., 7:10] = scaled_color(history)
    return features


def history_weight(head, *, blend_scale=BLEND_SCALE):
    """The per-pixel history weight the model asked for, from head channel 4.

    The logit is rounded to half before anything else, so the gate has 65536 possible
    inputs. NumPy's own expression - `gate_formula`, below - is evaluated once on every one
    of them, and each frame indexes that table: the same expression on the same values, so
    bit-identical, where it used to run an exp, a reciprocal and a clip on every pixel of
    the output. 3.1 ms of a 1280x720 frame. `test_nr_model.py` checks the table against the
    formula on all 65536 inputs.
    """
    bits = np.asarray(head, dtype=np.float32)[..., 3:4].astype(np.float16).view(np.uint16)
    return gate_table(float(blend_scale))[bits]


def gate_formula(logit, blend_scale=BLEND_SCALE):
    """The gate as the model defines it, on a logit already rounded to half."""
    return np.clip(1.0 / (1.0 + np.exp(-logit)) * half(blend_scale), 0, 1)


HOLD_RAMP = np.float32(4.0)


def release_slope(levels):
    """The folded constant of the release: the gate's share falls from all of it at no
    change to none by `levels` of 255. 0 turns it off."""
    return float(np.float32(-255.0 / levels)) if levels > 0 else 0.0


def release_factor(current, previous, slope):
    """How much of the model's gate a pixel keeps, from what the game did to it: the
    largest step of its three channels, `clip(1 + moved * slope, 0, 1)` in the order the
    native composition computes it."""
    moved = np.abs(np.subtract(current[..., 0], previous[..., 0], dtype=np.float32))
    scratch = np.empty_like(moved)
    for channel in (1, 2):
        np.subtract(current[..., channel], previous[..., channel], out=scratch)
        np.maximum(moved, np.abs(scratch, out=scratch), out=moved)
    np.multiply(moved, np.float32(slope), out=moved)
    np.add(moved, np.float32(1.0), out=moved)
    return np.clip(moved, 0, 1, out=moved)[..., None]


def hold_floor(current, previous, strength):
    """Per-pixel lower bound on the history weight, from what the *game* did: full where
    the game handed back the same pixel, gone by `HOLD_RAMP` levels of 255 (notes/phase54).

    Channel by channel and in place - the obvious `max(abs(a - b), axis=2)` builds two
    temporaries and runs seven full-frame passes at the output resolution - and folded
    into one multiply-add-clip, which is the order the native composition transcribes.
    """
    floor = np.abs(np.subtract(current[..., 0], previous[..., 0], dtype=np.float32))
    scratch = np.empty_like(floor)
    for channel in (1, 2):
        np.subtract(current[..., channel], previous[..., channel], out=scratch)
        np.maximum(floor, np.abs(scratch, out=scratch), out=floor)
    # clip(1 - moved * 255 / ramp, 0, 1) * strength, folded into one multiply-add-clip
    np.multiply(floor, np.float32(-255.0 * strength / HOLD_RAMP), out=floor)
    np.add(floor, np.float32(strength), out=floor)
    return np.clip(floor, 0, strength, out=floor)[..., None]


_GATE_TABLES = {}


def gate_table(blend_scale):
    """`gate_formula` on every half value, indexed by the value's sixteen bits."""
    table = _GATE_TABLES.get(blend_scale)
    if table is None:
        every = np.arange(1 << 16, dtype=np.uint32).astype(np.uint16).view(np.float16)
        with np.errstate(over="ignore", invalid="ignore"):
            table = gate_formula(every.astype(np.float32), blend_scale).astype(np.float32)
        _GATE_TABLES[blend_scale] = table
    return table
