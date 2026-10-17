# W2 - GEMM kernel SotA for skinny, small-K and batched shapes

Author: orchestrator (Wave-1 W2 subagent stalled on the provider endpoint; compiled from
verified sources, 2026-10-17).

## Scope and search log

Queries: Stream-K / work-centric decomposition; tall-and-skinny GEMM tensor cores; split-K and
wave quantisation; epilogue fusion (CUTLASS EVT); batch small-size HGEMM; skinny-GEMM roofline.
Retrieved and confirmed: Stream-K (arXiv 2301.03598), TSM2X (arXiv 2002.03258), Ernst/Hager
tall&skinny (arXiv 1905.03136, IJHPCA DOI 10.1177/1094342020965661), EVT (HKUST/ASPLOS 2024),
MAGMA batch small HGEMM (UTK ICL 2019), EVT explainer (Colfax Research 2024).
Counts: found ~15 / screened 10 / included 6 / excluded 4 (square-only or NVIDIA-proprietary
with no transferable mechanism).
Wave 2 (gap-fill): +1 source (S7 MCFuser, SC 2024); total = 7.

## Sources

### S1. Stream-K: Work-centric Parallel Decomposition for Dense Matrix-Matrix Multiplication
- arXiv: 2301.03598 (Osama, Merrill, Cecka, Garland, Owens), 2023. Retrieved: yes.
- Tier: T1 (peer-reviewed; implemented in CUTLASS; 32,824-shape corpus).
- Key claims:
  - Partitions the aggregate MAC-loop iterations evenly across processing elements instead of
    tiles; "near-perfect utilization ... regardless of how efficiently the output tiling ...
    quantizes across the underlying processing elements."
  - Peak speedup up to **14x** vs CUTLASS's data-parallel kernel and up to **6.7x** vs cuBLAS
    across 32,824 GEMM geometries, on NVIDIA A100.
  - Communication overhead scales with processor width, not output tiles; "compiles to a single
    kernel."
- Transferability to DG2/A770: partial - the *mechanism* (split the K-loop into fixed-size chunks
  and let a fixed number of workgroups claim them via an atomic counter) is arch-agnostic and
  directly attacks our observed problem: skinny N=16..128 quantises badly into 8-wide DPAS tiles,
  leaving whole Xe-cores idle at the tail of every dispatch. Implementing Stream-K on a persistent
  Xe workgroup set is feasible. No Xe data in the paper.
- Exact vs approximate: exact (reassociation of additions only; same result modulo fp rounding).
- Risk: the A770 has 32 Xe-cores / 512 XVEs; a persistent Stream-K grid is small, so the atomic
  counter contention is negligible. This is the single most on-point scheduling source for our
  shapes.

### S2. TSM2X: High-Performance Tall-and-Skinny Matrix-Matrix Multiplication on GPUs
- arXiv: 2002.03258 (2021). Retrieved: yes.
- Tier: T2 (preprint with benchmarks on K40c/V100).
- Key claims:
  - Defines three regimes for tall-and-skinny GEMM: compute-bound, memory-bound, latency-bound.
  - "When k is very small (m >> k ~= n), each GPU thread would not perform enough workload to
    hide latency and hence low occupancy" -> latency-bound.
  - Published (JPDC) abstract: TSM2R "speeds up the computation by 1.1x~3x and improves the memory
    bandwidth utilization and computing power utilization by 8%~47.6% and 7%~37.3%"; TSM2L
    "speeds up ... by 1.1x~3.5x and improve[s] the memory bandwidth utilization by up to 55%".
    (Note: earlier v1 text quoted 1.6x/1.9x average; the JPDC numbers are the authoritative ones.)
  - cuBLAS used <10% of peak bandwidth at n=2, <20% at n=16 (v1 measurements, K40c).
- Transferability to DG2/A770: partial. Our shapes are exactly the "M huge, N=16..128, K=32..128"
  latency-bound case. The transferable idea is to parallelise the K loop (split-K) and use
  ILP/unrolling to hide latency, because the output tile is too small to occupy the machine
  otherwise. Numbers are NVIDIA-pre-TC; mechanism transfers.
- Exact vs approximate: exact.

### S3. Performance engineering for real and complex tall & skinny GEMM kernels on GPUs
- arXiv: 1905.03136; IJHPCA 2021 (Ernst, Hager, Thies, Wellein). Retrieved: yes.
- Tier: T1 (peer-reviewed journal, DOI 10.1177/1094342020965661).
- Key claims:
  - Skinny = 1..64 columns, tall = >1e6 rows.
  - Arithmetic intensity is 8 flop/byte at M,N=64, "close to the roofline knee of a V100"
    (knee 5-17 flop/byte) -> the operation is at the memory/compute boundary.
  - Their code generation + autotuning reaches **>= 2/3 of roofline** and "often substantially
    outperform" cuBLAS on Volta; CUBLAS/CUTLASS "far below" potential for the transposed-times
    and skinny cases.
  - Limiter is **memory latency** (pointer-chase ~435 cy), not bandwidth: they top out at
    4766 GF/s vs a 7066 GF/s roofline (2/3) for one kernel.
- Transferability to DG2/A770: yes for the model. Our FLOP/B is 11-26 (the brief), i.e. right at
  the ridge for a 560 GB/s / ~138 TF machine (ridge ~246 flop/byte if purely DRAM-bound, so at
  11-26 flop/byte our GEMMs are strongly DRAM-bound unless producer/consumer chaining removes the
  round trip). The practical conclusion: for our shapes the win is in *not moving bytes*, not in
  raising DPAS utilisation alone.
- Exact vs approximate: exact.

### S4. EVT: Accelerating Deep Learning Training with Epilogue Visitor Tree
- HKUST / ASPLOS 2024 (Chen, Kerr, Cai, Kosaian, Wu); DOI 10.1145/3620666.3651359;
  explainer: https://research.colfax-intl.com/epilogue_visitor_tree/ . Retrieved: yes.
- Tier: T1 (peer-reviewed, integrated in CUTLASS).
- Key claims:
  - A composable tree of epilogue visitors ("each leaf node ... add, multiply, load, store")
    lets arbitrarily complex epilogues fuse into the GEMM store path; 1.26-3.1x speedup.
  - Enables reductions, broadcasts, activations in the epilogue.
- Transferability to DG2/A770: partial. CUDA-specific implementation, but the *pattern* is exactly
  what we need: our QKV epilogue (cosine normalisation + E4M3 publish) and gated-FFN epilogue are
  listed as major frame costs. The transferable design is: keep the DPAS result in registers and run
  the whole epilogue as a visitor tree before storing, instead of a separate pass.
- Exact vs approximate: exact if the fused epilogue reproduces the same arithmetic; approximate if
  it substitutes (e.g. exp2 for exp).

### S5. Optimizing Batch HGEMM on Small Sizes Using Tensor Cores (MAGMA)
- UTK ICL report 2019, https://icl.utk.edu/files/publications/2019/icl-utk-1234-2019.pdf . Retrieved: yes.
- Tier: T2 (tech report, code in MAGMA).
- Key claims:
  - For very small matrices that "cannot fully occupy the tensor core units", a warp can run
    multiple GEMMs per tensor-core op: max simultaneous = min(TC_M/M, TC_N/N).
  - Outperforms cuBLAS for sizes up to 100 by 1.2x-10.5x; very small sizes up to 26x.
  - Below ~25% tensor-core utilisation (square <= ~10), a non-TC codegen path wins.
- Transferability to DG2/A770: yes, directly analogous. DPAS tile is 8x8x16; our N=16 and K=32..
  mean we can pack several independent skinny GEMMs (or window blocks) into one wide DPAS tile.
  This is the "batch the skinny Ns into one 16-wide DPAS" idea, and the source gives the packing
  rule and the crossover heuristic (switch off XMX when utilisation < ~25%).
- Exact vs approximate: exact.

### S6. FlashAttention-2 block-size guidance (cross-ref W4)
- arXiv 2307.08691. See W4 for detail. Included here only for the "blocks {64,128}x{64,128}"
  small-head-dim guidance.

### S7. MCFuser - fusion of memory-bound compute-intensive operator chains (Wave 2)
- arXiv: 2506.22169 (Zhang et al., SC 2024; IEEE doi 10.1109/SC41406.2024.00040). Retrieved: yes.
- Tier: T1 (SC 2024, peer-reviewed) / directional for us (NVIDIA-only evaluation).
- Key claims:
  - Generates fused kernels for "**memory-bound compute-intensive (MBCI) operator chains**" - i.e.
    chains that *look* compute-heavy but become memory-bound at the actual (dynamic) tensor sizes.
    This is precisely our case: small N/K per GEMM makes each one bandwidth- or latency-bound.
  - High-level tiling expressions define the search space; DAG analysis "decouple[s] memory access
    statements from loops, significantly reducing redundant memory transfers between GPU global and
    shared memory"; analytical performance model replaces the ML cost model.
  - "up to a **5.9x** speedup in kernel performance" over Ansor and "reducing tuning time by over
    **70-fold**"; average 2.7x/1.6x over Ansor, 6.6x/3.7x over PyTorch on A100/RTX3080.
- Transferability to DG2/A770: partial. Framework is NVIDIA-only in evaluation; the *idea* (fuse the
  chain, prune redundant global<->SLM traffic) maps to the D4 epilogue-fusion lever here. Its main
  value for us is the method: an analytical cost model (no ML training) over a tiling search space,
  which is exactly what W7 autotuning would need on our box.
- Exact vs approximate: exact (it is a search/scheduling method; numerics unchanged).

## Variance table

| Source | Regime claim | Hardware | Note |
|---|---|---|---|
| S1 Stream-K | scheduling / quantisation | A100 | up to 14x vs single-tile; the tail effect |
| S2 TSM2X | latency-bound when k tiny | K40c/V100 | split-K + ILP |
| S3 Ernst | memory-latency bound at ridge | V100 | 2/3 of roofline max |
| S5 MAGMA | pack many small GEMMs | V100 | min(TC_M/M, TC_N/N) |
| S7 MCFuser | fuse MBCI chains, cut global/SLM traffic | A100/3080 | 5.9x over Ansor; -70x tune time |

No direct contradiction: all agree our shapes are at/below the ridge and are limited by
movement/latency, not by DPAS throughput.

## Practical implications for this frame

1. Direction: Stream-K / persistent work-claiming over the K-loop for the skinny GEMMs.
   Mechanism: fixed persistent workgroups claim fixed-size K-chunks via a global atomic; removes
   the per-dispatch tail quantisation of N=16..128 across 32 Xe-cores. Attacks: the ~45 us dispatch
   floor AND the tail idle. Ceiling: at FLOP/B 11-26 the machine is DRAM-bound; Stream-K cannot
   exceed the roofline, but it removes the scheduling loss which is the current 3.4-5 TF vs ~33 TF
   microbench gap (a 6-9x deficit). Effort: medium-high. Risk: medium. FIRST falsifiable
   experiment: run the same skinny GEMMs at the frame shapes with (a) one dispatch per shape and
   (b) a persistent Stream-K grid; compare GFLOP/s and per-dispatch time in the runtime log.
2. Direction: batch skinny Ns into one 16-wide DPAS tile (MAGMA-style packing).
   Mechanism: process 8 independent N=16..128 GEMMs whose Ns sum to <= 16 columns as one DPAS.
   Attacks: low DPAS utilisation on N=16. Ceiling: increases useful DPAS work per instruction;
   bounded by the register-resident probe (~86 TF). Effort: medium. Risk: medium (layout).
   FIRST falsifiable experiment: microbench 8x(N=16,K=64) separately vs packed; compare TFLOP/s.
3. Direction: epilogue-visitor fusion of the cosine-normalise + E4M3 publish and the gated FFN.
   Mechanism: keep DPAS output in registers, run the epilogue tree, store once. Attacks: the
   named "major costs" (QKV epilogue, gated FFN epilogue). Ceiling: removes one full
   read-write round trip of the activation per epilogue. Effort: medium. Risk: low.
   FIRST falsifiable experiment: count DRAM bytes for the current separate epilogue pass vs a
   fused prototype on one Swin block via a bandwidth microbenchmark.
4. Direction: use MCFuser's analytical-model-over-tiling-space method (S7) to drive the fusion/Stream-K
   autotuning rather than an ML cost model.
   Mechanism: high-level tiling expressions + DAG redundancy elimination + analytic cost model;
   replaces the expensive ML cost model that makes Kernel Tuner/BaCO slow for us (W7). Attacks: the
   tuning-time (not the steady-state) part of the problem. Ceiling: same as D4; benefit is effort/time.
   Effort: medium. Risk: low (method transfer only). FIRST falsifiable experiment: replicate S7's
   analytical cost model for one fused epilogue chain on our box and compare its predicted vs
   measured bandwidth-bound time (calibration error target < 20%).

## Excluded leads

- Square-GEMM-specific tiling papers (no benefit for skinny).
- cuBLAS/CUTLASS proprietary heuristics (mechanism already captured by S1/S3).
- Blackwell/TMA-specific kernels (no TMA on DG2).