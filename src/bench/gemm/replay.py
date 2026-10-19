#!/usr/bin/env python3
"""Replay a frame's GEMM census on the device: per-shape rate and the frame total.

Each shape is recorded `calls` times into one submit, exactly as the frame records it
(one global barrier between dispatches), so the sum is the GEMM half of a frame time.
The census's publish flags are replayed too: without them this benches the raw-store
path, which is not what the frame's published shapes run.
"""

import json
import sys
import time

import numpy as np
from xmxrt import gemm
from xmxrt.recorder import Recorder

from nr_root import ROOT

shapes = json.loads(
    (ROOT / "work" / (sys.argv[1] if len(sys.argv) > 1 else "shapes-768x1280.json")).read_text()
)
shapes.sort(key=lambda s: -2.0 * s["m"] * s["n"] * s["k"] * s["batch"] * s["calls"])

rt = Recorder()
print("  %-30s %6s %9s %9s %9s" % ("M x N x K (batch)", "calls", "ms", "GFLOP/s", "GB/s"))
total_ms = total_flops = total_bytes = 0.0
rows = []
for s in shapes:
    m, n, k, b, t, calls = (
        s["m"],
        s["n"],
        s["k"],
        s["batch"],
        s["transpose_b"],
        s["calls"],
    )
    epilogue, narrow = s.get("epilogue", 0), s.get("narrow", False)
    width = 2 if narrow else 4
    a = rt.buffer(m * k * b, np.float16)
    w = rt.buffer(k * n * b, np.float16)
    c = rt.buffer(m * n * b, np.float16 if narrow else np.float32)
    # upload(), not view(): an OpenCL device's buffers are not host addressable
    a.upload(np.full(m * k * b, 0.01, np.float16))
    w.upload(np.full(k * n * b, 0.01, np.float16))
    # The GPU idles at a low clock between submits, so a single timed pass measures
    # the ramp as much as the kernel; a frame runs continuously. Warm, then best of 3.
    ms = 1e9
    for repeat in range(3):
        rt.begin()
        for _ in range(calls):
            gemm.gemm(
                rt, a, w, c, m, n, k, batch=b, transpose_b=t, epilogue=epilogue, narrow=narrow
            )
        started = time.perf_counter()
        rt.submit()
        elapsed = 1000 * (time.perf_counter() - started)
        if repeat:
            ms = min(ms, elapsed)
    flops = 2.0 * m * n * k * b * calls
    traffic = float(b) * calls * (m * k * 2 + k * n * 2 + m * n * width)
    publish = f"[{epilogue}{' n' if narrow else ''}]" if (epilogue or narrow) else ""
    rows.append((m, n, k, b, t, calls, ms, flops, traffic, publish))
    total_ms += ms
    total_flops += flops
    total_bytes += traffic
    a.free()
    w.free()
    c.free()

for m, n, k, b, t, calls, ms, flops, traffic, publish in rows[:24]:
    label = f"{m} x {n} x {k}" + (f" (b={b})" if b > 1 else "") + (" T" if t else "")
    if publish:
        label += " " + publish
    print(
        "  %-30s %6d %9.2f %9.1f %9.1f"
        % (label, calls, ms, flops / (ms / 1000) / 1e9, traffic / (ms / 1000) / 1e9)
    )
print("  %-30s %6s %9s %9s %9s" % ("-" * 30, "", "", "", ""))
print(
    "  %-30s %6d %9.1f %9.1f %9.1f"
    % (
        "TOTAL",
        sum(r[5] for r in rows),
        total_ms,
        total_flops / (total_ms / 1000) / 1e9,
        total_bytes / (total_ms / 1000) / 1e9,
    )
)
print(
    f"\n  {total_flops / 1e9:.1f} GFLOP and {total_bytes / 1e9:.2f} GB of traffic in {total_ms:.0f} ms of GEMM per frame"
)
