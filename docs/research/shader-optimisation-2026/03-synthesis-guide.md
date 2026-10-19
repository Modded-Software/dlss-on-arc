# 03 - Synthesis guide (ranked directions)

Scope: how to move the DLSS-NR network on 2x Arc A770 from the current ~4-10 TFLOP/s (in-frame)
toward the hardware's ~138-157 TFLOP/s FP16 XMX peak, **without changing the model**. Built from
the evidence library in `02-evidence-library/`. Every claim below is backed by a source id
(`wN-Sn`) there. Coverage caveats, the Wave-2 gap-fill and claim confidence are in
`04-gap-analysis.md`.

## The one-paragraph diagnosis

The frame is ~450 dependent dispatches, each with a full barrier and a ~45 us floor, at
FLOP/B 11-26 (right at or below the memory/compute ridge). The project's own numbers say the GEMM
loop is issue-bound (only ~20% `mad`, dominated by `mov`/address math), and an OpenCL DPAS
microbench reaches 33 TF while a register-resident probe reaches 86 TF. The published literature
agrees on the shape of the problem: at this FLOP/B the win is in **not moving bytes** and **not
paying per-op launch/barrier overhead**, not in raising DPAS utilisation alone (`w2-S3`).

## Ranked directions

Ranking = (expected ceiling on this frame) x (confidence) / (effort x risk).

### D1. Fused, register/SLM-resident per-window Swin-block kernel  [highest confidence]
- Mechanism: one kernel per 8x8 window (64 tokens) doing QKV + attention + out-proj + residual +
  gated FFN, gather/stage via SLM, 8x8x16 DPAS for QK^T and PV, epilogue in registers.
- Sources: `w4-S1` (keys shared per window; shift is a memory permutation), `w4-S3` (unfused
  window attention is killed by the scatter/gather), `w2-S4` (epilogue-visitor fusion), `w1-S1`
  (DG2 tile is 8x8x16).
- Attacks: window-attention cost; QKV epilogue (cosine-normalise + E4M3 publish); gated-FFN
  epilogue; the DRAM round trips between sub-ops.
- Ceiling: eliminates the per-window DRAM traffic for the shallow levels entirely; bounded by
  64 KB SLM and the 8x8x16 DPAS rate. For C<=256 shallow levels the whole block's activations fit.
- Effort: high. Risk: medium (SLM layout, masked shift).
- FIRST falsifiable experiment: for one shallow block, compute the SLM bytes for a fused
  per-window kernel; if < 64 KB, build it and compare total GPU time against the current
  multi-dispatch path on that block.
- Adversarial check: if the block does NOT fit in SLM, this direction collapses to "fuse only the
  epilogues" (D4). Also, on DG2 the ANV SLM coopmat-load lowering is a known problem (`w1`, Mesa
  issue 10961); the OpenCL/IGC path may be required, not Vulkan.

### D2. Persistent whole-frame kernel with a device-side dependency graph (Stream-K for ops)
- Mechanism: one long-lived workgroup set reads a task list and advances a counter array; combine
  with Stream-K-style work-claiming for the skinny GEMMs so N=16..128 no longer quantises badly
  across 32 Xe-cores.
- Sources: `w3-S1` (MPK SM-level tGraph, 1.7x), `w3-S5` (Hazy megakernel, bubbles at 50% BW),
  `w2-S1` (Stream-K, 14x vs single-tile on A100, tail quantisation).
- Attacks: the ~45 us/dispatch floor x450 AND the per-dispatch tail idle.
- Ceiling: MPK/Hazy band 1.5-1.7x on latency; Stream-K removes the scheduling loss that sits
  between 3.4-5 TF and the 33 TF microbench.
- Effort: high. Risk: high (device-side synchronisation correctness on Xe).
- FIRST falsifiable experiment: take 2-3 real frame shapes chained, compare 450 barrier-separated
  dispatches vs one persistent counter-synced kernel; measure total GPU time.
- Adversarial check: MPK/Hazy are CUDA and their gains are partly kernel-launch elimination on
  NVIDIA; on Xe the launch cost may already be smaller (the 45 us floor is measured, but how much
  of it is launch vs barrier is unknown - W7's protocol must split it). If most of the 45 us is
  the barrier, split barriers / fewer barriers may capture most of the win for less effort.

### D3. Keep E4M3 weights 1 byte on device; expand in-register via a 256-entry LUT
- Mechanism: store the model's own E4M3 bytes; index a register/SLM-resident 256-entry FP16 LUT in
  the GEMM load path; feed DPAS FP16.
- Sources: `w5-S6` (LUT-GEMM, exact for the stored format, 2.1x), `w5-S5` (MARLIN: weight-only
  pays while memory-bound and requires wide loads + offline reshuffle).
- Attacks: the 292 MB of FP16 weights streamed per frame (dominant DRAM traffic in a
  bandwidth-bound frame).
- Ceiling: ~halves weight bytes -> up to ~2x on the weight-streaming portion; **exact**, no quality
  change (E4M3->FP16 is a pure function).
- Effort: medium. Risk: low. This is the highest ratio of gain to risk because it changes no
  arithmetic.
- FIRST falsifiable experiment: measure per-frame DRAM bytes for weights under (a) the FP16 decode
  pass and (b) direct 1-byte LUT decode inside the GEMM; compare total bytes and time.
- Adversarial check: if the LUT lookup itself becomes the issue bottleneck (it is a gather), it may
  not beat a vectorised convert; measure the LUT lookup cost on Xe before committing.

### D4. Epilogue-visitor fusion of the named "major costs"
- Mechanism: keep DPAS output in registers, run cosine-normalise + E4M3 publish and the gated FFN
  as a fused epilogue tree, store once (`w2-S4`).
- Attacks: QKV epilogue, gated-FFN epilogue (both named major costs in the brief).
- Ceiling: removes one full read-write round trip of the activation per epilogue.
- Effort: medium. Risk: low.
- FIRST falsifiable experiment: count DRAM bytes for the current separate epilogue pass vs a fused
  prototype on one Swin block.

### D5. Depth-first fusion of the shallow U-Net levels (fuse-depth rule)
- Mechanism: keep the stack's weights + activations in L2/SLM and run whole levels tile-by-tile;
  choose fuse depth from DeFiNES' "weights fit in on-chip memory" rule (`w3-S4`, 4.1x-10-fold on
  accelerators).
- Attacks: DRAM round trips at the shallow, bandwidth-bound levels.
- Ceiling: bounded by A770's 16 MB L2; up to the DF-vs-LBL bound.
- Effort: medium-high. Risk: medium.
- FIRST falsifiable experiment: per U-Net level, bytes(weights+activations); pick the deepest stack
  <16 MB and measure bytes saved vs the current per-op round trip.

### D6. Multi-GPU displaced patch parallelism (DistriFusion/PipeFusion)
- Mechanism: spatial split with async previous-frame halo (DistriFusion) for shallow levels; layer
  split (PipeFusion) for deep levels, over PCIe.
- Sources: `w6-S2` (6.1x/8 A100), `w6-S3` (PCIe-friendly, shards params).
- Attacks: per-GPU compute. Ceiling ~2x (2 GPUs), and the project already has an alternate-frame
  mode at 44-52 dB - so the marginal gain is only if it keeps both GPUs busy full-time.
- Effort: medium-high. Risk: quality at the halo.
- FIRST falsifiable experiment: measure PCIe bytes/frame for spatial-split-with-halo vs layer-split
  at 1024x576.

### D7. Eventful-Transformer token gating (temporal, training-free)
- Sources: `w6-S1` (2-4x compute, 1.74x runtime, no re-training).
- Attacks: full-frame compute on static regions. Ceiling 2-4x for static content.
- Effort: medium-high. Risk: quality (approximate).
- FIRST falsifiable experiment: measure the fraction of unchanged tokens between consecutive game
  frames; if >50% (the Eventful break-even), prototype on one block.

### D8. Split-KV + LSE for the ViT-1D bottleneck (C=1024, 160-192 tokens)
- Sources: `w4-S4` (Flash-Decoding: <1% GPU utilisation with batch 1).
- Attacks: under-utilisation of 32 Xe-cores on a small attention.
- Effort: medium. Risk: low. FIRST experiment: measure active Xe-core count during the 1D
  attention pass (W7 protocol).

### D9 (monitor, do not act yet). Mesa 26.2 DGC; int8 Ozaki emulation of E4M3xFP16
- `w3-S6` DGC: not in Mesa 25.2.8; and the spec says it does not run async - uncertain win.
- `w5-S2/S3`: int8 DPAS is 2x FP16; emulating E4M3xFP16 exactly on int8->int32 DPAS is the
  highest-ceiling numeric idea, but the slice-count and reduction cost (`w5-S4`) must be measured
  first. Risk: high.
- **Closed on DG2 (Run 93) - do not act.** The int8-stands-in-for-E4M3 half is settled and loses:
  an int8 staged GEMM (configuration 4, tile `8x8x32` at this part's own N=8) was built, is
  bit-exact against exact integer arithmetic on six shapes, and measured **0.66-0.98x** of f16 on
  the eight ViT-bottleneck shapes (1392.7 us f16 against 1807.8 us int8). Both DG2 tiles are N=8,
  so only K per instruction doubles while the operand-fragment traffic that the kernel is bound on
  (`W1` tile row; `phase77`) is unchanged. "int8 DPAS is 2x FP16" is an Xe2/NVIDIA density figure
  and does not transfer. Only the DGC half of D9 remains a monitor item.

## Sequencing recommendation

1. First: **D3** (exact, low risk, directly attacks the dominant DRAM traffic) and **D4** (exact,
   low risk, attacks named costs). Both are exact so they cannot regress quality.
2. Then: **D1** and **D8**, which need SLM/one-kernel work but stay exact.
3. Then: **D2** and **D5**, the scheduler/fusion overhaul (highest ceiling, highest effort).
4. **D6/D7** are quality-gated (approximate) and should follow only after the project's quality
   metric is wired into the loop.

## Cross-cutting prerequisite: measurement (W7)
None of the above can be attributed without the W7 protocol: `unitrace`/`cl_gpu_metrics` +
`instcount` + Advisor GPU roofline with XMX (`w7-S1/S2/S3`). Specifically, the frame must be
classified per op as DPAS-issue / load-issue / SLM / L3 / DRAM / latency, and the 45 us floor must
be split into launch vs barrier. Several rankings (D2 vs split-barriers) hinge on that split.

## Adversarial review of this synthesis
- Most high multipliers in the literature (Stream-K 14x, Welder 21x, Souffle 7.8x) are against
  *unfused* baselines; against an already-fused engine the realistic band is 1.5-2x (`w3` variance
  table). We do not claim more.
- Every "exact" direction (D3, D4, D5, D8) changes data movement only, so it cannot change outputs
  - the strongest guarantee the brief allows.
- The biggest single unknown is DG2's ANV coopmat path itself: the project measured that
  `coopMatLoad` from SLM lowers to 16 scalar 16-bit loads + 16 movs, and llama.cpp independently
  found coopmat on A770 gave ~no gain and even a regression on Ubuntu (`w4-S5`). This means the
  DPAS route may have to go through OpenCL/IGC, not Vulkan. That is a fork in every DPAS direction.

## Limitations
- The seven Wave-1 subagents stalled on a non-responding model endpoint, so this review was
  assembled by the orchestrator from directly-retrieved sources. Coverage is therefore narrower
  than the protocol's 10-15 sources/workstream target (see protocol section 9 for actual counts).
- The single-file evidence format and the orchestrator's own retrieval both bias toward sources
  that were easy to fetch. The Wave-1 named leads (DeltaCNN, Skip-Convolutions, MCFuser, Gemini
  COT) were resolved in Wave 2 (`04-gap-analysis.md` §3); none changes this ranking.
- No GPU experiments were run for this review (protocol restricts it); every "ceiling" is a
  literature-derived estimate and every "experiment" is proposed, not executed.
- Hardware readings (tiles, subgroup, extensions) are for Mesa 25.2.8 on this exact box and must be
  re-checked if the driver is bumped.