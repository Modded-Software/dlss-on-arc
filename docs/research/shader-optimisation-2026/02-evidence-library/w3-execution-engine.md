# W3 - Execution-engine overhaul

Author: orchestrator (Wave-1 W3 subagent stalled on the provider endpoint; compiled from
verified primary sources, 2026-10-17).

## Scope and search log

Queries: megakernel / persistent whole-model kernel; tile-graph and dataflow schedulers;
layer/depth-first fusion; device-generated commands; GPU workgroup progress models.
Retrieved and confirmed: Mirage MPK (arXiv 2512.22219 / OSDI'26), Welder (OSDI'23), Souffle
(ASPLOS'24, DOI 10.1145/3617232.3624858), DeFiNES (arXiv 2212.05344), Hazy "No Bubbles" megakernel
(blog 2025-05-27 + github HazyResearch/Megakernels), VK_EXT_device_generated_commands (Phoronix
2026-05-06; Khronos proposal).
Counts: found ~18 / screened 12 / included 7 / excluded 5 (no artifact, or closed CUDA-only with
no mechanism).

## Sources

### S1. Mirage Persistent Kernel (MPK) - mega-kernelizing tensor programs
- arXiv: 2512.22219 (Cheng, Zhang, Zhou, Ji, Jiang et al.), OSDI'26 PDF: https://www.usenix.org/system/files/osdi26-cheng.pdf
  Code: https://github.com/mirage-project/mirage . Retrieved: yes.
- Tier: T1 (OSDI'26 + open source).
- Key claims:
  - Transforms multi-GPU model inference into a **single megakernel**; "up to **1.7x** lower
    end-to-end inference latency" vs kernel-per-operator baselines.
  - Key idea: an **SM-level task graph** (`tGraph`) - "nodes denote tasks running on individual
    SMs and whose edges encode fine-grained dependencies"; enables cross-operator software
    pipelining and fine-grained overlap.
  - In-kernel runtime partitions SMs into **workers** (FIFO task queues) and **schedulers**
    (dispatch when prerequisites satisfied), using device-memory semaphores.
- Transferability to DG2/A770: partial. CUDA/NVSHMEM, but the entire architectural idea is
  transferable to a persistent Vulkan/OpenCL compute kernel: our frame is ~450 dependent
  dispatches each with a full barrier at a ~45 us floor. An SM->Xe-core-level task graph with
  device-side dependency counters would remove the host-side dispatch and the global barrier.
  The paper is the closest published blueprint for our exact problem.
- Exact vs approximate: exact.

### S2. Welder - tile-graph memory-access scheduling
- OSDI'23, https://www.usenix.org/system/files/osdi23-shi.pdf . Retrieved: yes.
- Tier: T1 (OSDI'23).
- Key claims:
  - `tile-graph`: a tile-level dataflow graph; each node processes one data tile and edges carry a
    chosen reuse memory layer (`SetConnect`/`Propagate`).
  - Found **89** operator-fusion patterns automatically; up to **21.4x** over PyTorch, **2.8x**
    over Ansor, and **1.7x** over FasterTransformer.
  - Explicitly targets the "underutilized compute cores + saturated memory bandwidth" mismatch -
    i.e. exactly our frame (SKINNY GEMMs at the ridge). Handles models larger than device memory
    by spanning device+host memory.
  - Maps to code via four hardware interfaces: `Allocate, LoadTiles, ComputeTile, StoreTiles`.
- Transferability: yes, strongly. The four-interface mapping is runtime-agnostic and could be
  emitted as Vulkan compute dispatches; the tile-traffic cost model would tell us which of our
  inter-block edges (residual, gated FFN) to keep in SLM/L1 vs DRAM. This gives the *cost model*
  that W2's chaining and W3's fusion both need.
- Exact vs approximate: exact (data movement only).

### S3. Souffle - global analysis and tensor expressions
- ASPLOS'24, DOI 10.1145/3617232.3624858 ; artifact https://github.com/summerspringwei/souffle-ae . Retrieved: yes.
- Tier: T1 (ASPLOS'24, open artifact).
- Key claims:
  - Top-down: treats the whole graph as one merged kernel, then *partitions* into subprograms
    using data-reuse in shared-memory/registers and generated instruction count.
  - "geometric mean speedup of up to **3.7x over TensorRT** and **7.8x over XLA**" on A100.
  - Uses a tensor-expression (TE) dependency graph + affine transformations to eliminate redundant
    memory access across operators; partitions when "the number of blocks of TE4 exceeds the max
    blocks per wave limit."
- Transferability: partial-to-yes. The "max blocks per wave" partition criterion is directly
  applicable to Xe: with 32 Xe-cores, a subprogram that needs >32 workgroups in one wave forces a
  barrier; Souffle's analysis would split/merge accordingly. Useful as the partitioning algorithm.
- Exact vs approximate: exact.

### S4. DeFiNES - depth-first / layer-fusion scheduling space
- arXiv: 2212.05344 (Mei, Goetschalckx, Symons, Verhelst; KU Leuven / imec), **HPCA 2023**;
  code https://github.com/ZigZag-Project/DeFiNES . Retrieved: yes.
- Tier: T1 (HPCA 2023 + open source, validated against taped-out DepFiN).
- Key claims:
  - Formalises layer-by-layer (LBL) vs depth-first (DF) / cross-layer fusion and gives an
    analytical energy+latency cost model across **every** memory level (not just DRAM).
  - DF beats LBL on 4 of 5 architectures, "gains of up to **4.1x**"; up to **10-fold** better
    solutions than generalized SotA; TPU-like needs an on-chip weight buffer to benefit.
  - Fuse-depth is chosen automatically "as long as the total number of weights in the stack fit in
    the highest on-chip memory level that holds weights."
- Transferability to DG2/A770: yes for the *model and the fuse-depth rule*. Our shallow U-Net
  levels are bandwidth-bound (brief), and DeFiNES says DF specifically wins exactly there. The
  "weights in the stack fit in on-chip memory" rule tells us how deep a fused stack can go given
  A770's 16 MB L2 / 64 KB SLM.
- Exact vs approximate: exact.

### S5. "No Bubbles" megakernel for Llama-1B (Hazy Research)
- Blog: https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles (2025-05-27);
  code https://github.com/HazyResearch/Megakernels . Retrieved: yes.
- Tier: T3 (blog) + open source; directional, strong artefact.
- Key claims:
  - "vLLM and SGLang are only able to use at most **50%** of available GPU bandwidth ... each
    kernel comes with a setup and teardown period and during this time no useful work gets done."
  - Their megakernel uses **78%** of memory bandwidth and is **>1.5x** faster than vLLM, **>2.5x**
    vs vLLM at some points; on B200 <680 us/forward.
  - Three techniques: (1) an on-GPU **interpreter** executing a pre-scheduled per-SM instruction
    stream; (2) **paged shared memory** (13 x 16 KiB pages on H100) to pipeline loads across
    instructions; (3) explicit **counter-based synchronisation** in global memory.
- Transferability to DG2/A770: partial. The bubble diagnosis matches our ~45 us floor exactly;
  the paged-SLM + counter design transfers conceptually to a persistent Xe compute kernel with
  SLM paging (64 KB SLM/Xe-vector-engine). CUDA-specific code doesn't.
- Exact vs approximate: exact.

### S6. VK_EXT_device_generated_commands (DGC) on Intel ANV
- Phoronix: https://www.phoronix.com/news/Intel-ANV-Vulkan-DGC ("Intel's Vulkan Linux Driver Now
  Supports Device Generated Commands", 2026-05-06); spec: https://docs.vulkan.org/refpages/latest/refpages/source/VK_EXT_device_generated_commands.html
  Retrieved: yes.
- Tier: T1 (spec) + T3 (news, dated).
- Key claims:
  - DGC "is finally merged" for Intel ANV, "ahead of next quarter's **Mesa 26.2** release" (2026).
  - Spec: "While the GPU can be faster than a CPU to generate the commands, it will not happen
    asynchronously to the device, therefore the primary use case is generating **less** total work
    (occlusion culling, classification to use specialized shaders, etc.)."
- Transferability to DG2/A770: **no today** - we run Mesa 25.2.8; DGC needs Mesa 26.2. Even then,
  the spec warns it is not asynchronous, so for ~450 *fixed* dispatches the saving is uncertain.
  Re-evaluate after a Mesa bump. This is a "monitor", not an "act".
- Exact vs approximate: exact.

## Variance table

| Source | Lever | Reported gain | Scope |
|---|---|---|---|
| S1 MPK | single megakernel, SM-level graph | 1.7x latency | multi-GPU LLM |
| S2 Welder | tile-graph fusion + reuse cost model | 1.7-21.4x | DNN inference |
| S3 Souffle | top-down partition + TE transform | 3.7-7.8x | DNN inference A100 |
| S4 DeFiNES | depth-first layer fusion model | 4.1x-10-fold | accelerators |
| S5 Hazy | manual megakernel | 1.5-2.5x | Llama-1B |
| S6 DGC | device-generated commands | n/a | Mesa 26.2 only |

Notes on variance: the large multipliers (S2/S3) are vs *unfused* baselines; against an already
fused engine the realistic band is S1/S5's **1.5-2x**. The common mechanism across S1/S2/S3/S5 is
the same: replace many barrier-separated kernels with one persistent kernel plus a device-side
dependency graph. They differ only in how the graph is built (SM-level tGraph vs tile-graph vs TE
subprograms vs hand schedule).

## Practical implications for this frame

1. Direction: persistent whole-frame compute kernel with a device-side dependency graph.
   Mechanism: launch one long-lived workgroup set (e.g. 32 Xe-core-aligned groups) that reads a
   task list from a buffer and advances a counter array; each task is one of the frame's ops.
   Attacks: the ~45 us/dispatch floor (x450) AND the barrier stalls. Ceiling: if the 45 us is
   mostly launch/teardown, the ceiling is near the sum of pure compute; S5 suggests >1.5x, S1 up to
   1.7x. Effort: high. Risk: high (correctness of device-side sync). FIRST falsifiable experiment:
   on 2-3 real frame shapes chained, compare 450 barrier-separated dispatches vs one persistent
   kernel with counter sync; measure total GPU time.
2. Direction: depth-first fusion of the shallow U-Net levels (DeFiNES fuse-depth rule).
   Mechanism: keep the stack's weights in L2/SLM and run whole levels tile-by-tile. Attacks: the
   DRAM round trips at the shallow, bandwidth-bound levels. Ceiling: up to the 4.1x DF-vs-LBL
   bound, but bounded by 16 MB L2 on A770. Effort: medium-high. Risk: medium.
   FIRST falsifiable experiment: compute, per U-Net level, (weights + activations) bytes; pick the
   deepest stack that fits in 16 MB; measure bytes saved vs the current per-op round trip.
3. Direction: adopt Welder's tile-traffic cost model as the scheduler's decision rule.
   Mechanism: model each inter-op edge's traffic at each memory layer and choose the reuse layer.
   Attacks: scheduling and cache-residency decisions. Effort: medium. Risk: low. FIRST falsifiable
   experiment: reproduce the model offline for one Swin block and check it predicts the measured
   L2 hit behaviour.
4. Direction (monitor): Mesa 26.2 DGC. Do not act until we can bump Mesa; even then, only helps if
   it reduces *work*, not just command bytes.

## Excluded leads

- CUDA Graphs / streams: analogous to what we already have (barrier per dispatch).
- `VK_NV_device_generated_commands` (NVIDIA-only).
- PDL (programmatic dependent launch): NVIDIA-only, no Intel analogue found.