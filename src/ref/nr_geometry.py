#!/usr/bin/env python3
"""
nr_geometry - the network's typing extent for a frame, and the frame a scale asks for.

Both are the vendor's field: the picture is mirrored out to at least the graph's own
reductions, so every halving in the network is exact (OpenDLSS-NR's `geometryFromValid`,
bit-exact against captures of the original).
"""

from __future__ import annotations

from nr_deps import NetworkGeometry

# The network's frame is padded, by mirroring, to at least this on a side. 320 is what
# NVIDIA's own driver does; the graph itself runs down to 128 - a window of 8 at a sixteenth
# of the extent, MLX-DLSS's graph contract. At a small live size most of a 320x320 frame is
# mirror padding: 44 % of it for a 320x180 frame, 82 % for 179x101.
VENDOR_MINIMUM_EXTENT = 320
GRAPH_MINIMUM_EXTENT = 128


def _field_alignment(extent):
    """Two to the number of times the graph reduces an axis of `extent` pixels: six halvings,
    each rounded up to 4, that shrink it, and a seventh when level 0 is not whole 8-pixel
    windows. The field is aligned to that, so every halving is exact."""
    reductions, size = 0, extent
    for level in range(6):
        half = -(-((size + 1) // 2) // 4) * 4
        reductions += half < size
        reductions += level == 0 and half % 8 != 0
        size = half
    return 1 << reductions


def network_geometry(width, height, minimum=VENDOR_MINIMUM_EXTENT):
    """The network extent for a `width` x `height` frame, as the vendor pads it (OpenDLSS-NR's
    `geometryFromValid`, bit-exact against captures of the original at eleven sizes): each
    side aligned to the graph's own reductions (`_field_alignment`) and at least `minimum`
    (never below the graph's 128), and the width one alignment more when both sides are
    four alignments.

    That last step is not cosmetic. Such a field - 1280x768, 1024x768, 512x512 - pools down
    to a bottleneck with no padding token in it, and there the pass comes out 25-30 % weaker
    than on any field around it (notes/opendlss-reference.md). MLX-DLSS's rule, a multiple
    of 64, ran a 1280x720 frame on exactly such a field.

    Below 129 pixels a side the vendor's field need not be a multiple of 64, which this graph
    - exact halvings down to level 3 - cannot follow; there it is rounded up to 64.
    """
    floor = max(GRAPH_MINIMUM_EXTENT, int(minimum))
    align_width, align_height = _field_alignment(width), _field_alignment(height)
    field_width = max(floor, -(-width // align_width) * align_width)
    field_height = max(floor, -(-height // align_height) * align_height)
    if field_width % (4 * align_width) == 0 and field_height % (4 * align_height) == 0:
        field_width += align_width
    return NetworkGeometry(width, height, -(-field_width // 64) * 64, -(-field_height // 64) * 64)


# MLX-DLSS's own pipeline and temporal session ask `vendor_aligned` for their field, and their
# answer - a multiple of 64 - is theirs, not the vendor's; in this tree it is the vendor's.
NetworkGeometry.vendor_aligned = classmethod(
    lambda cls, width, height: network_geometry(width, height)
)

_EXTENTS: dict = {}


def render_extent(width, height, scale, minimum=VENDOR_MINIMUM_EXTENT):
    """The frame the network is handed for a `width` x `height` picture at render `scale`, as
    (width, height): of all the frames at least the scale's own, aspect kept, the one on the
    cheapest network field - and of those the largest.

    The graph runs on the field, not on the frame: a frame smaller than the field is mirrored
    out to it, and every frame on the same field costs the same. On a small window the field
    is held at `minimum` a side, and there a low scale bought no speed and handed the network
    mostly its own mirror image - 640x360 at 0.35 is 224x126 in a 320x320 field, 28 % of it
    the picture, and the pass came out a quarter weaker than at 0.5, on the same field in the
    same time. And the vendor's field is not monotonic in the frame: 800x450 at 0.35 is
    280x158, whose width aligns to 384, where 0.4's 320x180 lands on 320x320 - the lower scale
    was the slower. Now 640x360 runs as 0.5 for any scale up to it and 800x450 at 0.35 as 0.4,
    a lower scale is never the slower, and where a scale already fills its field - 1280x720
    and up - nothing moves.
    """
    key = (int(width), int(height), float(scale), int(minimum))
    if key in _EXTENTS:
        return _EXTENTS[key]
    if scale >= 1.0:
        best = (int(width), int(height))
    else:

        def cost(w, h):
            geometry = network_geometry(w, h, minimum=minimum)
            return geometry.network_width * geometry.network_height

        best = (max(64, round(width * scale)), max(64, round(height * scale)))
        cheapest = cost(*best)
        for w in range(best[0] + 1, int(width) + 1):
            candidate = (w, max(64, round(height * w / width)))
            price = cost(*candidate)
            if price <= cheapest:
                best, cheapest = candidate, price
    _EXTENTS[key] = best
    return best
