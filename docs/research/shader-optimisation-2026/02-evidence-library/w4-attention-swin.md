# W4 - Attention and Swin-block kernels

Author: orchestrator (Wave-1 W4 subagent stalled on the provider endpoint; compiled from
verified primary sources, 2026-10-17).

## Scope and search log

Queries: FlashAttention 1/2/3; neighborhood/window attention kernels; Flash-Decoding; Swin
shifted-window and its latency; Intel Vulkan coopmat flash-attention kernels in llama.cpp.
Retrieved and confirmed: FlashAttention-2 (arXiv 2307.08691), Faster Neighborhood Attention
(arXiv 2403.04690, NeurIPS 2024), Flash-Decoding (PyTorch blog + Stanford CRFM), Swin Transformer
(arXiv 2103.14030), llama.cpp PR #14001 (Xe2 coopmat enable), PR #24406 (Intel Xe flash attention),
PR #19320 (Intel coopmat matmul).
Counts: found ~20 / screened 12 / included 6 / excluded 6 (long-sequence-only, or no small-head-dim
relevance).

## Sources

### S1. Swin Transformer (shifted windows)
- arXiv: 2103.14030 (Liu, Lin, Cao, Hu, Wei et al., Microsoft Research Asia), 2021. Retrieved: yes.
- Tier: T1 (peer-reviewed, code).
- Key claims:
  - Window-based MSA has complexity linear in image size, unlike global MSA.
  - Shifted-window efficiency: "all query patches within a window share the same key set ...
    which simplifies memory access in hardware"; the shift has "much lower latency than the
    sliding window method ... yet is similar in modeling power."
  - Cyclic shift + masking keeps the window count the same and is "also efficient" (vs padding
    which would give 2.25x more windows).
  - Example uses M=4 windows on an 8x8 map (our model uses 8x8 windows = 64 tokens).
- Transferability: yes. This is the architecture of our network. Two concrete openings the paper
  itself flags: (a) keys are shared per window -> QKV/attention can be a single per-window kernel;
  (b) the cyclic shift is a memory permutation that can be folded into the gather index instead of
  a physical copy (avoid the roll pass).

### S2. FlashAttention-2
- arXiv: 2307.08691. Retrieved: yes.
- Tier: T1 (peer-reviewed, code).
- Key claims:
  - "50-73% of the theoretical maximum FLOPs/s on A100"; up to 10-fold vs PyTorch.
  - Parallelise over sequence length to raise occupancy when batch*heads is small.
  - "Typically choose blocks of size {64,128}x{64,128}, depending on the head dimension."
- Transferability to DG2/A770: partial. Our window is 64 tokens with head dim 32, so the whole
  QK^T for a window is 64x64 with 32-deep K: it fits entirely in registers/SLM, which is *smaller*
  than FlashAttention's {64,128} blocks. So FA2's contribution here is the work-partitioning
  principle, not the tiling numbers; the actual kernel is a fused per-window attention.

### S3. Faster Neighborhood Attention (NATTEN)
- arXiv: 2403.04690 (Hassani, Hwu, Shi), NeurIPS 2024; code https://github.com/SHI-Labs/NATTEN/ . Retrieved: yes.
- Tier: T1 (NeurIPS 2024, code).
- Key claims:
  - "neighborhood attention can be represented as a batched GEMM problem, similar to standard
    attention" with a fused gather/scatter.
  - Fused NA improves naive kernels by 1759% (1-D) / 958% (2-D) FP16; up to 104% inference gain.
  - Crucially: "all unfused neighborhoods ... are greatly limited by the scatter/gather operation"
    and "aside from being heavily bound by memory bandwidth."
- Transferability to DG2/A770: partial-to-yes. Mechanism transfers: our 8x8-window attention is a
  2-D neighborhood attention; the fused-gather insight says do NOT materialise the gathered window
  to DRAM, gather in SLM/registers. This is the same conclusion as S1(b). Numbers are A100.
- Exact vs approximate: exact.

### S4. Flash-Decoding
- PyTorch blog https://pytorch.org/blog/flash-decoding/ (2023-10-13); Stanford CRFM
  https://crfm.stanford.edu/2023/10/12/flashdecoding.html . Retrieved: yes.
- Tier: T3 (blog) - directional, strong pedagogical content.
- Key claims:
  - Adds a parallelisation dimension over keys/values; "up to 8x speedups in decoding" and
    attention up to 50x faster than FlashAttention for long sequences.
  - With batch size 1, FlashAttention uses "<1% of the GPU"; split-KV fixes that.
- Transferability: yes for the principle. Our ViT-1D bottleneck has C=1024 with only 160-192
  tokens and 32 heads - i.e. a small number of query blocks. The split-over-KV + log-sum-exp
  reduce trick is exactly how to keep all 32 Xe-cores busy on such a small attention. Even though
  our sequence is short, the *head count x query blocks* may still undersubscribe 32 Xe-cores.

### S5. llama.cpp PR #14001 - coopmat enablement for Intel
- https://github.com/ggml-org/llama.cpp/pull/14001 . Retrieved: yes.
- Tier: T2/T3 (vendor-adjacent project PR with measurements).
- Key claims:
  - "Enabling VK_KHR_cooperative_matrix on Intel Xe2 GPUs ... significant performance improvement,
    while we also see performance regressions with older GPUs like **Arc A770**. This PR will
    enable ... only for Xe2 GPUs until performance regression is resolved for older GPUs."
  - A770 measurements: Windows pp512 974.92 -> 977.81 (~100%, no change); Ubuntu Mesa 24.2.8
    pp512 333.88 -> 328.25 (~98%, slight regression). Also Ubuntu A770 is **~3x slower than
    Windows** for pp512 (328 vs 975).
- Transferability to DG2/A770: yes - this is *our* GPU and it is a strong, independent,
  measurement-based warning: naively enabling coopmat on DG2 did not help. It corroborates the
  project's own finding that the ANV SLM coopmat-load lowering is the problem (W1).
- Exact vs approximate: exact (perf only).

### S6. llama.cpp PR #24406 / #19320 - Intel Xe flash attention and coopmat matmul
- https://github.com/ggml-org/llama.cpp/pull/24406 ; https://github.com/ggml-org/llama.cpp/pull/19320 . Retrieved: yes.
- Tier: T2/T3 (project PRs with Intel-authored measurements and design discussion).
- Key claims:
  - Blocking fact stated explicitly by a maintainer: "We currently require **16x16x16 coopmat
    shape support** for Flash Attention, which IIRC Intel GPUs don't offer. They have **8x8x16**?"
    Intel authors confirm they "initially tried to add CM1 with existing FA kernel flow" but could
    not get consistent wins without major changes to dispatch/memory access/SLM layout.
  - Intel GPU FA needs a *different* kernel: "GRF/SLM message/Thread/WG arch" forces changes to
    "dispatch/memory access pattern, SLM layout, variable location (SLM to GRF), output
    rescaling/GEMM PV flow."
  - They needed a **two-phase (split) design** because "For single-phase FA, due to GRF/SLM size
    limitation, either GRF spill or extra reduce ... both have extra overheads."
  - On A770, `matrix cores: none` is reported by ggml even when coopmat is compiled, and coopmat
    matmul PR (#19320) was explicitly not targeted at A770/Xe1-dGPU; maintainers closed it pending
    a driver.
- Transferability to DG2/A770: yes - directly. Two hard constraints for our design: (1) DG2's
  8x8x16 vs FA's required 16x16x16 means FA2/FA3 kernels cannot be lifted as-is; the flash-attention
  structure must be rebuilt around 8x8x16 DPAS and two-phase reduction; (2) GRF/SLM pressure
  (128 GRF, 64 KB SLM) forces split-phase rather than single-phase for larger head dims.
- Exact vs approximate: exact.

## Variance table

| Source | Claim | Arch | Caveat for us |
|---|---|---|---|
| S2 FA2 | 50-73% of peak | A100 | needs 16x16x16 coopmat; DG2 has 8x8x16 |
| S3 NATTEN | fused gather wins 958% 2-D | A100 | CUDA/CUTLASS; mechanism transfers |
| S5 llama.cpp | coopmat on A770 = no gain | A770 | matches our ANV lowering problem |
| S6 llama.cpp | FA needs 16x16x16, two-phase | Intel | DG2 must use 8x8x16 + split |

Variance: S2's block-size advice ({64,128}) does not apply because our window (64 tokens, dim 32)
is already small; S6 shows the Intel-specific kernel must differ fundamentally from S2's.

## Practical implications for this frame

1. Direction: one fused per-window Swin-block kernel (QKV + attention + out-proj + residual + FFN).
   Mechanism: 64 tokens x C(<=256) fits in SLM (64 KB) for the shallow levels; gather the shifted
   window via index remap (S1b), run 8x8x16 DPAS for QK^T and PV, epilogue in registers. Attacks:
   window-attention cost, the QKV epilogue, and the DRAM round trips between sub-ops. Ceiling:
   removes the per-window DRAM traffic entirely for the shallow levels; bounded by SLM capacity and
   by the 8x8x16 DPAS rate. Effort: high. Risk: medium. FIRST falsifiable experiment: for one
   shallow block, count SLM bytes needed for a fused per-window kernel; if < 64 KB, build it and
   compare against the current multi-dispatch path.
2. Direction: split-KV + log-sum-exp reduce for the ViT-1D bottleneck (S4).
   Mechanism: split the 160-192 keys across sub-groups, each computes partial attention + LSE, then
   reduce. Attacks: under-utilisation of 32 Xe-cores on a small attention. Ceiling: brings the
   small attention from <1% to near-full core use *if* it is core-bound; if DRAM-bound the ceiling
   is bandwidth. Effort: medium. Risk: low. FIRST falsifiable experiment: measure active Xe-core
   count during the 1D attention pass (W7's protocol) - if it undersubscribes, apply split-KV.
3. Direction: do NOT lift FA2/FA3 as-is (S6). Mechanism: build the attention around 8x8x16 and a
   two-phase reduction from the start. This is a *negative result that saves time*.

## Excluded leads

- FA3 / Hopper-specific (TMA, warp-specialisation) - no DG2 analogue.
- Sliding-window attention (Longformer/BigBird) - our windows are non-overlapping Swin windows,
  already cheaper.
- Global-MSA kernels - wrong complexity class for this model.