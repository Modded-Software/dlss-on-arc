#!/usr/bin/env python3
"""
nr_grade - the three conditioning scalars and the vendor's post-network colour grade.

`controls` turns a profile and its overrides into the style/tone/structure scalars the
network conditions on.  `grade_for` and `style_grade` are the colour grade the vendor
applies after the network for each style, read out of the DLL
(`notes/phase70-post-process.md`).
"""

from __future__ import annotations

import numpy as np
from nr_deps import PROFILES

# What the vendor's own panel starts at, which is not what MLX-DLSS's profiles use:
# structure 1.5 rather than 1.0, skin structure 2.0, and the automatic mask on.
# Recovered from the shipped control surface, not from the DLL -
# `notes/phase30-control-atlas.md`.
VENDOR_DEFAULTS = {
    "normalized_style": 0.0,
    "local_tone_strength": 1.0,
    "local_structure_strength": 1.5,
}
VENDOR_AUTOMATIC_MASK = (2.0, -1.0)  # skin structure, automatic-mask structure


def controls(profile="standard", style_index=None, local_tone=None, local_structure=None):
    """The three conditioning scalars, a profile plus any explicit overrides.

    `style_index` is the vendor's integer style, normalised by 1/128; the profiles
    are style 0 / 1 / 2 with tone and structure at 1, and `neutral` is style 0 with
    both at 0. `vendor` is what the shipped panel starts at.
    """
    values = dict(VENDOR_DEFAULTS if profile == "vendor" else PROFILES[profile])
    if style_index is not None:
        values["normalized_style"] = style_index / 128.0
    if local_tone is not None:
        values["local_tone_strength"] = local_tone
    if local_structure is not None:
        values["local_structure_strength"] = local_structure
    return values


# The colour grade the vendor applies after the network for each style: an exposure in
# EV, a smoothstep contrast and a saturation offset, each the preset times the local tone
# strength clamped to [0, 1]. Read out of the DLL itself - the per-style table at
# 0x1800b0de4 (style 1's mask 0x34 is exposure, contrast and saturation, style 2's 0x20
# saturation alone), its lerp from neutral at 0x18001d5f0, and the operator in
# `cg2r_post_process_kernel` (`notes/phase70-post-process.md`). Style 0 has no entry and
# no grade. The hosts that name the styles call 1 `natural` and 2 `cinematic`.
STYLE_GRADES = {1: (-0.1, -0.25, -0.1), 2: (0.0, 0.0, -0.15)}


def grade_for(normalized_style=0.0, local_tone_strength=1.0, **_):
    """The grade a set of conditioning scalars asks for, as the three float32 factors
    `style_grade` takes - the exposure as a multiplier, the contrast, the saturation as a
    multiplier - or None where there is none. The vendor scales the preset in float32 and
    skips its pass when every value is neutral, which is a tone of 0."""
    preset = STYLE_GRADES.get(round(float(normalized_style) * 128))
    tone = np.float32(min(max(float(local_tone_strength), 0.0), 1.0))
    if preset is None or tone == 0:
        return None
    exposure, contrast, saturation = (np.float32(value) * tone for value in preset)
    return (
        np.float32(2.0 ** float(exposure)),
        np.float32(contrast),
        saturation + np.float32(1.0),
    )


_THIRD, _SIXTH, _TWO_THIRDS = np.float32(1 / 3), np.float32(1 / 6), np.float32(2 / 3)


def _hue_channel(p, q, t):
    t = np.where(t < 0, t + np.float32(1), t)
    t = np.where(t > 1, t - np.float32(1), t)
    rising = p + t * ((q - p) * np.float32(6))
    descending = p + (_TWO_THIRDS - t) * (q - p) * np.float32(6)
    return np.where(
        t < _SIXTH,
        rising,
        np.where(t < np.float32(0.5), q, np.where(t < _TWO_THIRDS, descending, p)),
    )


def style_grade(rgb, grade):
    """`cg2r_post_process_kernel`'s grade on RGB in [0, 1], for the parameters a style
    sets. Per channel an exposure and a smoothstep contrast; then in HSL a saturation
    multiplier. The kernel's other stages - levels, temperature and tint, a five-zone tone
    curve, a gamma and a saturation power - are identities at the values no style changes
    and are left out; the vendor evaluates two of them as `ex2(lg2(x))`, which the
    hardware's approximations make a 1e-7 wobble rather than an identity. The operations
    and their order are `nr_image.c`'s `style_grade`, so the two are the same bytes."""
    exposure, contrast, saturation = grade
    c = np.clip(np.asarray(rgb, dtype=np.float32), 0, 1)
    c = np.clip(c * exposure, 0, 1)
    c = np.clip(c + contrast * (c * c * (np.float32(3) - (c + c)) - c), 0, 1)
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    high = np.maximum(np.maximum(r, g), b)
    low = np.minimum(np.minimum(r, g), b)
    light = (high + low) * np.float32(0.5)
    delta = high - low
    grey = ~(high > low)
    with np.errstate(divide="ignore", invalid="ignore"):
        chroma = np.where(
            light > np.float32(0.5),
            delta / ((np.float32(2) - high) - low),
            delta / (high + low),
        )
        hue = np.where(
            high == r,
            ((g - b) / delta + np.where(g < b, np.float32(6), np.float32(0))) / np.float32(6),
            np.where(
                high == g,
                ((b - r) / delta + np.float32(2)) / np.float32(6),
                ((r - g) / delta + np.float32(4)) / np.float32(6),
            ),
        )
    chroma = np.clip(np.where(grey, np.float32(0), chroma) * saturation, 0, 1)
    hue = np.where(grey, np.float32(0), hue)
    q = np.where(
        light < np.float32(0.5),
        light * (chroma + np.float32(1)),
        (light + chroma) - light * chroma,
    )
    p = (light + light) - q
    out = np.stack(
        (
            _hue_channel(p, q, hue + _THIRD),
            _hue_channel(p, q, hue),
            _hue_channel(p, q, hue - _THIRD),
        ),
        axis=-1,
    )
    out = np.where((chroma <= 0)[..., None], light[..., None], out)
    return np.clip(out, 0, 1).astype(np.float32)


def truncate_half(value):
    """float32 toward zero onto the half grid: what the vendor's RGBA16F history keeps of
    the prediction, as OpenDLSS-NR captured it - truncated, not rounded."""
    value = np.asarray(value, dtype=np.float32)
    bits = value.view(np.uint32)
    kept = (bits & np.uint32(0xFFFFE000)).view(np.float32)
    small = np.floor(np.abs(value) * np.float32(2**24)) / np.float32(2**24)
    return np.where(
        (bits & np.uint32(0x7FFFFFFF)) >= np.uint32(0x38800000),
        kept,
        np.copysign(small, value),
    ).astype(np.float32)
