#!/usr/bin/env python3
"""Window attention's cost against its batch count, on whichever backend NR_GPU_BACKEND names.

The frame's window-attention call sites run from 240 to 15617 (window, head) batches, one
workgroup each. A call's time split into a part that grows with the batches and a part
that does not is what says whether a kernel loses on its work or on a fixed cost - a cold
first wave, a launch, a tail. Each point records CALLS back-to-back calls of the merged,
biased attention the frame runs, submits them once, and keeps the best of ROUNDS walls,
so the submit's own cost is spread over the calls.

    NR_GPU_BACKEND=vulkan XMX_DEVICE=1 uv run python src/bench/kernels/attention_curve.py
    ... --batches 240 480 960 --heads 16 --calls 200 --rounds 5
    ... --stamps --after 15360 384 128

--stamps also reads the per-pass GPU timestamps frame_profile.py attributes by (one after
each pass, a pass charged ts[i] - ts[i-1]) and prints the attention's mean beside the wall
time. --after M N K records an M x N x K GEMM into its own buffers before every attention
call, as the frame's QKV projection precedes it; with --stamps the GEMM's attributed mean
is printed too. If the GEMM-then-attention pair's wall time is the two alone summed while
the attention's stamp grows and the GEMM's shrinks, the timestamps are moving the GEMM's
tail onto the attention rather than the attention costing more.

Prints us per call and per batch for every point, then a least-squares line over the
points (us = fixed + per_batch * batches) with its residuals.
"""

import argparse
import time

import numpy as np
from xmxrt import attention, gemm, memory, profile
from xmxrt.recorder import Recorder

# (batches, heads) of the frame's 1280x768 call sites, small to large
FRAME_POINTS = [
    (240, 16),
    (288, 16),
    (320, 16),
    (384, 16),
    (480, 8),
    (528, 8),
    (560, 8),
    (616, 8),
    (960, 4),
    (1008, 4),
    (1040, 4),
    (1092, 4),
    (1920, 2),
    (2000, 2),
    (2050, 2),
    (3840, 1),
    (3969, 1),
    (15360, 1),
    (15617, 1),
]
ROW_FAMILY = profile.PROFILE_FAMILIES.index("row")
WINDOW_ATTENTION = ROW_FAMILY * 32 + 3  # window_attention.comp's profile kind


def measure(rt, batches, heads, calls, rounds, after, stamps):
    """(wall us a call, attention stamp us or None, preceding GEMM stamp us or None)"""
    buffers = []

    def alloc(count, dtype=np.float32):
        buf = rt.buffer(count, dtype)
        buffers.append(buf)
        return buf

    try:
        rng = np.random.default_rng(batches * 31 + heads)
        q, k, v = [alloc(batches * 2048, np.float16) for _ in range(3)]
        for buf in (q, k, v):
            memory.host_write(buf, rng.normal(0, 0.3, batches * 2048).astype(np.float16))
        bias = alloc(heads * 4096)
        memory.host_write(bias, rng.uniform(-7, 7, heads * 4096).astype(np.float32))
        out = alloc(batches * 2048, np.float16)
        if after:
            m, n, kk = after
            ga, gb = alloc(m * kk, np.float16), alloc(kk * n, np.float16)
            for buf, count in ((ga, m * kk), (gb, kk * n)):
                memory.host_write(buf, rng.normal(0, 0.3, count).astype(np.float16))
            gc = alloc(m * n)

        def record():
            if after:
                gemm.gemm(rt, ga, gb, gc, m, n, kk)
            attention.window_attention(rt, q, k, v, out, batches, heads, bias=bias, merged=True)

        rt.begin()  # warm: pipelines built, buffers resident
        record()
        rt.submit()
        best = float("inf")
        attn_ms, gemm_ms = [], []
        for _ in range(rounds):
            rt.begin()
            for _ in range(calls):
                record()
            if stamps:
                profile.profile_reset()
            start = time.perf_counter()
            rt.submit()
            best = min(best, time.perf_counter() - start)
            if stamps:
                for ms, kind in zip(
                    profile.profile_each(), profile.profile_each_kinds(), strict=True
                ):
                    if ms < 0:
                        continue
                    (attn_ms if kind == WINDOW_ATTENTION else gemm_ms).append(ms)
        attn = 1000 * min_mean(attn_ms, calls) if stamps else None
        pre = 1000 * min_mean(gemm_ms, calls) if stamps and after else None
        return 1e6 * best / calls, attn, pre
    finally:
        for buf in buffers:
            buf.free()


def min_mean(values, per_round):
    """the lowest per-round mean: the rounds' stamps are concatenated in order"""
    rounds = [values[i : i + per_round] for i in range(0, len(values), per_round)]
    return min(sum(r) / len(r) for r in rounds if r)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--batches", type=int, nargs="*", help="batch counts (default: the frame's)"
    )
    parser.add_argument("--heads", type=int, help="heads for --batches (must divide each)")
    parser.add_argument("--calls", type=int, default=100, help="calls recorded per submit")
    parser.add_argument("--rounds", type=int, default=5, help="submits per point; the best is kept")
    parser.add_argument(
        "--stamps", action="store_true", help="also report timestamp-attributed time"
    )
    parser.add_argument(
        "--after",
        type=int,
        nargs=3,
        metavar=("M", "N", "K"),
        help="record an M x N x K GEMM before every attention call",
    )
    args = parser.parse_args()
    if args.batches:
        if not args.heads:
            parser.error("--batches needs --heads")
        points = [(b, args.heads) for b in args.batches]
    else:
        points = FRAME_POINTS
    for b, h in points:
        if b % h:
            parser.error(f"{h} heads do not divide {b} batches")
    rt = Recorder()
    if args.stamps:
        profile.profile(True)
    header = f"  {'batches':>8} {'heads':>5} {'us/call':>9} {'ns/batch':>9}"
    if args.stamps:
        header += f" {'attn stamp':>11}" + (f" {'gemm stamp':>11}" if args.after else "")
    if args.after:
        print("  every call preceded by a %dx%dx%d GEMM; us/call is the pair's" % tuple(args.after))
    print(header)
    xs, ys = [], []
    for b, h in points:
        us, attn, pre = measure(rt, b, h, args.calls, args.rounds, args.after, args.stamps)
        xs.append(b)
        ys.append(us)
        line = f"  {b:>8} {h:>5} {us:>9.2f} {1000 * us / b:>9.2f}"
        if attn is not None:
            line += f" {attn:>11.2f}"
        if pre is not None:
            line += f" {pre:>11.2f}"
        print(line)
    if len(points) >= 2:
        per_batch, fixed = np.polyfit(np.array(xs, float), np.array(ys, float), 1)
        print(f"  fit: {fixed:.2f} us fixed + {1000 * per_batch:.2f} ns a batch")
        print(
            "  residuals (us): "
            + " ".join(f"{y - (fixed + per_batch * x):+.2f}" for x, y in zip(xs, ys, strict=True))
        )


if __name__ == "__main__":
    main()
