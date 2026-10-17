#!/usr/bin/env python3
"""
nr_compose - the head over the frame: `compose` and its single native pass `compose_encode`.

Post-network and cheap, so a caller can sweep intensity, detail and colour without
re-running the network.  With a history image this is MLX-DLSS's `compose_temporal`
instead of its `compose_head`: the model's own gate decides, per pixel, how much of the
previous output survives into this one, and this tree's floor and release correct it.
"""

from __future__ import annotations

import numpy as np
from nr_deps import compose_detail, compose_head, half, nr_image
from nr_features import (
    BLEND_SCALE,
    HOLD_RAMP,
    gate_table,
    history_weight,
    hold_floor,
    release_factor,
    release_slope,
)
from nr_grade import style_grade, truncate_half


def _vendor_prediction(predicted, grade, neural):
    """The prediction after the history's blend, as the vendor goes on with it: kept in
    `neural` truncated to half, and that half graded - or untouched without either."""
    if grade is None and neural is None:
        return predicted
    kept = truncate_half(predicted)
    if neural is not None:
        np.copyto(neural, kept)
    return predicted if grade is None else style_grade(kept, grade)


def compose(
    head,
    color,
    *,
    intensity=1.0,
    detail_strength=1.0,
    colour_strength=1.0,
    detail_radius=4.0,
    control_mask=None,
    history=None,
    history_confidence=1.0,
    history_floor=None,
    history_previous=None,
    history_hold=0.0,
    history_release=0.0,
    blend_scale=BLEND_SCALE,
    grade=None,
    neural=None,
):
    """The head over the frame. Post-network and cheap: sweep it without re-running.

    With a `history` image this is MLX-DLSS's `compose_temporal` instead of its
    `compose_head`: the model's own gate decides, per pixel, how much of the previous
    output survives into this one. `history_confidence` scales that gate globally -
    0 is the still-frame path exactly, 1 is the model's own answer.

    The vendor reads the history back out of feature channels 7-9, which pins it to the
    network's extent; a live frame composes at the *output* extent, so it arrives here
    as an image instead. That skips one `scaled_color` round trip through fp16, worth
    at most 0.06 of a 0-255 level, and keeps the game's own pixels off the resampler.

    `history_floor` raises the gate per pixel rather than lowering it, which the vendor
    has no need of and we do. The gate is not local: on a frame where most things move it
    reads 0.12 even over pixels that did not move, against 0.705 on a scene where the
    history was correct everywhere (`notes/phase54-flicker-fix.md`, `phase12`). A caller
    that can prove the history is correct for a pixel - because the game handed back the
    same bytes - can say so here. The floor is still bounded by `blend_scale`, so no
    pixel is held harder than the model itself ever holds one.

    `history_release` is the other half of the same signal: where the game's pixel changed
    by `history_release` levels of 255 or more, none of the gate survives, and by less, a
    share that falls linearly with the change. Without motion vectors the history at a
    pixel something moved across is what was there before, and a gate that reads 0.6 over
    a whole Tekken frame kept a trail of it behind everything that moved. It lowers only
    the gate - never the floor, which is zero wherever the game changed a pixel anyway.

    `intensity` blends the model's picture against the source, per pixel when a
    ControlMask supplies its red channel. `detail_strength` and `colour_strength`
    then re-weight the high and low frequencies of whatever change remains.

    Above 1 the blend extrapolates past the model's own picture. MLX-DLSS's
    `compose_head` clamps it, so anything over 1 was a no-op here; the
    vendor's panel goes to 2 and ships screenshots at 1.66
    (`notes/phase30-control-atlas.md`). At or below 1 this is bit-identical to the
    clamped path - the extrapolation only replaces the final blend, and the residual,
    its half rounding and the [0,1] clamp on the result are unchanged.

    `grade`, from `grade_for`, is the vendor's colour grade for a style, and `neural`, an
    array of the colour's shape, receives what the vendor keeps as the next frame's
    history: the prediction after the history's blend and before the grade, the intensity
    or anything else, truncated to half. The grade reads that same half value, as the
    vendor's post-process reads it back out of its RGBA16F scratch
    (`notes/phase70-post-process.md`). With neither, nothing here changes by a bit.
    """
    composed = None
    if history is not None or intensity > 1.0:
        head = np.asarray(head, dtype=np.float32)
        color = np.asarray(color, dtype=np.float32)
        alpha = None
        if history is not None:
            if head.shape[2] < 4:
                raise ValueError("the temporal gate is head channel 4; pass all four")
            history = np.asarray(history, dtype=np.float32)
            if history.shape != color.shape:
                raise ValueError("history must match the colour image shape")
            previous = (
                history_previous
                if history_previous is not None and (history_hold > 0 or history_release > 0)
                else None
            )
            slope_release = release_slope(history_release)
            if nr_image is not None and history_floor is None:
                # Everything temporal in the one native pass: the gate from its table (the
                # exp that kept it in NumPy is inside the table), the confidence, and the
                # floor from the game's previous frame in the same folded multiply-add-clip
                # as `hold_floor`. 7 ms of NumPy at a 1280x720 output.
                confidence = (
                    float(np.clip(np.float32(history_confidence), 0, 1))
                    if history_confidence != 1.0
                    else 1.0
                )
                composed = nr_image.compose_temporal(
                    head,
                    color,
                    history,
                    previous,
                    None,
                    control_mask,
                    intensity=intensity,
                    blend_scale=blend_scale,
                    hold=float(history_hold) if previous is not None else 0.0,
                    slope=(
                        float(np.float32(-255.0 * history_hold / HOLD_RAMP))
                        if previous is not None
                        else 0.0
                    ),
                    table=gate_table(float(blend_scale)),
                    confidence=confidence,
                    release=slope_release if previous is not None else 0.0,
                    grade=grade,
                    neural=neural,
                )
            if previous is not None and history_floor is None and composed is None:
                history_floor = hold_floor(color, previous, history_hold)
            alpha = history_weight(head, blend_scale=blend_scale) if composed is None else None
            if composed is None and history_confidence != 1.0:
                alpha = alpha * np.clip(np.float32(history_confidence), 0, 1)
            if composed is None and previous is not None and slope_release != 0.0:
                alpha = alpha * release_factor(color, previous, slope_release)
            if composed is None and history_floor is not None:
                floor = np.clip(np.asarray(history_floor, dtype=np.float32), 0, 1)
                np.maximum(alpha, floor * np.float32(blend_scale), out=alpha)
            if composed is None and nr_image is not None:
                composed = nr_image.compose_temporal(
                    head,
                    color,
                    history,
                    None,
                    alpha,
                    control_mask,
                    intensity=intensity,
                    blend_scale=blend_scale,
                    hold=0.0,
                    slope=0.0,
                    grade=grade,
                    neural=neural,
                )
        elif nr_image is not None and control_mask is None:
            composed = nr_image.compose(head, color, intensity, grade=grade, neural=neural)
        if composed is None:
            residual = half(head[..., :3]) * np.float32(0.25)
            predicted = np.clip(color + residual, 0, 1)
            if history is not None:
                predicted += alpha * (history - predicted)
            predicted = _vendor_prediction(predicted, grade, neural)
            blend = np.float32(intensity)
            if control_mask is not None:
                blend = np.asarray(control_mask, dtype=np.float32)[..., :1] * blend
            composed = np.clip(color + blend * (predicted - color), 0, 1).astype(np.float32)
    else:
        # The still frame natively too: nr_compose clamps the blend to [0, 1] below 1 as
        # compose_head does, and is byte-identical to it (test_native_image.py). It was
        # only reached above 1, so photo mode and every cut paid 3.9 ms of NumPy at
        # 512x288 for the same bytes.
        if nr_image is not None and control_mask is None:
            composed = nr_image.compose(head, color, intensity, grade=grade, neural=neural)
        if composed is None and (grade is not None or neural is not None):
            # compose_head, with the prediction the vendor keeps and grades in the middle
            head = np.asarray(head, dtype=np.float32)
            color = np.asarray(color, dtype=np.float32)
            predicted = np.clip(color + half(head[..., :3]) * np.float32(0.25), 0, 1)
            predicted = _vendor_prediction(predicted, grade, neural)
            blend = np.float32(intensity)
            if control_mask is not None:
                blend = np.asarray(control_mask, dtype=np.float32)[..., :1] * blend
            blend = np.clip(blend, 0, 1)
            composed = np.clip(color + blend * (predicted - color), 0, 1).astype(np.float32)
        if composed is None:
            composed = compose_head(head, color, control_mask=control_mask, intensity=intensity)
    return compose_detail(
        color,
        composed,
        detail_strength=detail_strength,
        colour_strength=colour_strength,
        radius=detail_radius,
    )


def compose_encode(
    head,
    color,
    encoded,
    *,
    top=0,
    left=0,
    bgra=True,
    intensity=1.0,
    control_mask=None,
    history=None,
    history_confidence=1.0,
    history_previous=None,
    history_hold=0.0,
    history_release=0.0,
    blend_scale=BLEND_SCALE,
    samples=None,
    grade=None,
    neural=None,
):
    """`compose` of the head brought up to the colour's size, encoded into `encoded` at
    (`top`, `left`), in one native pass; `None` where that pass does not apply, and then
    nothing has been written.

    The same composition as `resample` then `compose` then `nr_daemon.encode`, byte for
    byte (`test_native_image.py`), for the cases the daemon meets live: a history with or
    without a control mask, or a still frame without one, at detail and colour strength 1
    - where `compose_detail` hands the composition back untouched. The parameters are
    derived exactly as `compose`'s native branches derive them. With `samples`, a step,
    returns `(composition, head samples)` - see `nr_image.compose_encode`. `grade` and
    `neural` are `compose`'s.
    """
    if nr_image is None or (history is None and control_mask is not None):
        return None
    if history is None:
        return nr_image.compose_encode(
            head,
            color,
            None,
            None,
            None,
            encoded,
            top=top,
            left=left,
            bgra=bgra,
            intensity=intensity,
            samples=samples,
            grade=grade,
            neural=neural,
        )
    previous = (
        history_previous
        if history_previous is not None and (history_hold > 0 or history_release > 0)
        else None
    )
    confidence = (
        float(np.clip(np.float32(history_confidence), 0, 1)) if history_confidence != 1.0 else 1.0
    )
    return nr_image.compose_encode(
        head,
        color,
        history,
        previous,
        control_mask,
        encoded,
        top=top,
        left=left,
        bgra=bgra,
        intensity=intensity,
        blend_scale=blend_scale,
        hold=float(history_hold) if previous is not None else 0.0,
        slope=(
            float(np.float32(-255.0 * history_hold / HOLD_RAMP)) if previous is not None else 0.0
        ),
        table=gate_table(float(blend_scale)),
        confidence=confidence,
        release=release_slope(history_release) if previous is not None else 0.0,
        samples=samples,
        grade=grade,
        neural=neural,
    )
