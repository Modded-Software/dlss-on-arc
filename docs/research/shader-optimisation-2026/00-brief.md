# Stage 0 Brief - Shader / execution-engine optimisation directions for DLSS-NR on 2x Arc A770

Date: 2026-10-17. Method: `../../../../../../WORKSPACE-WEB-CONTENT/docs/instructions/research-methodology.md`
(PRISMA-ScR, Tricco et al. 2018), same layout as the workspace's other PRISMA-ScR reviews.

## Research question
Which optimisation directions - hardware-specific (Xe-HPG / DG2 XMX), kernel-level, and
whole-execution-engine - can move the DLSS-NR network (`notes/MODEL-SPEC.txt`: 71-block Swin/ViT
U-Net, C=32..1024, 8x8 windows, head dim 32, E4M3 large weights, ~145.8 M params) from today's
**~4-10 TFLOP/s achieved** toward the **~138-157 TFLOP/s FP16 XMX peak of each of our two A770s**,
**without changing the model fundamentally** (same weights, same graph semantics; approximation
allowed only where it is training-free and quality-measurable)?

## Starting state (measured, from `recomp/diary/dlss-nr.md`, Runs 7-73)
- Vulkan (Mesa ANV 25.2.8, `VK_KHR_cooperative_matrix`, forced SIMD32, DG2 tile 8x8x16):
  117 ms device @1280x768; ~82 ms @1024x576. GEMM ~50-65 % of the frame.
- OpenCL (NEO/IGC, plain FMA core, SG16): 107 ms device @1280x768; beats Vulkan in replay.
- Frame GEMM work: 160-405 GFLOP/frame, 276-339 GEMM dispatches, ~410-500 passes in total.
  Achieved 3.4-5.0 TFLOP/s in-frame; best microbench 33 TFLOP/s (OpenCL block-read DPAS,
  big-K); register-resident DPAS probe ~86 TFLOP/s.
- Dominant shapes are skinny: M up to 983040, N=16..128, K=32..128 (FLOP/B 11-26, at/below ridge).
- Two GPUs: alternate-frame with stale history measured at 44-52 dB vs exact; a 2-daemon layer exists.
- Closed by measurement (do not re-propose without a new mechanism): SLM-staged FMA GEMM,
  workgroup-shape, persistent grid-stride loop, 256-GRF flag, compile-time N/K specialisation,
  fp32-widened operands, block-read DPAS on the frame's shapes, row-per-lane fused FFN,
  host/device overlap on the game path (temporal chain).

## Audience and artefact
The project owner and the next implementing agent. Output: a ranked catalogue of directions,
each with mechanism, evidence (tiered), the expected ceiling on *this* frame, the cost, and the
first falsifiable experiment. Evidence persisted in files, not chat.

## Boundaries
- Hardware: Intel Arc A770 16 GB (ACM-G10, Xe-HPG), x2, PCIe; Linux 7.1, Mesa ANV 25.2.8,
  Intel compute-runtime (NEO) + IGC. Other hardware only where the technique transfers.
- Software surface: Vulkan compute (GLSL/SPIR-V), OpenCL C, SYCL/ESIMD/oneDNN only if callable
  from the C runtime. Windows path (D3D12) noted, not required.
- Model: no retraining, no architecture change. Training-free approximations (caching, reuse,
  sparsity of change, quantisation of already-quantised weights) are in scope but tagged
  "approximate" and need a quality gate.
- Time window: primary emphasis 2022-2026 (SotA); seminal older work allowed when still current.

## Non-negotiables
1. Only real, retrieved sources (arXiv id / DOI / vendor doc / repo URL). No invented citations.
2. Every direction mapped to this frame's measured shapes and costs, with an honest ceiling.
3. Hardware claims about DG2 must be Tier 1 (Intel docs, IGC/Mesa source, Khronos registry) or
   measured on this box; Xe2/PVC/NVIDIA-only facts flagged as non-transferable where so.
4. Exact vs approximate tagged on every direction.
5. Already-closed levers listed with the measurement that closed them; a re-proposal must name
   what is different.
6. "Outside the box" coverage: execution-engine overhaul (megakernel, dataflow, graph compiler,
   multi-GPU, temporal reuse), not only GEMM tuning.

## Definition of done (empty checklist - Stage 8 gate)
- [ ] 7 workstream evidence files, each meeting its source target or proven thin
- [ ] Every critical claim corroborated (>=2 sources) or Tier 1
- [ ] Every citation retrieved, dated, URL'd
- [ ] Conflicts recorded in variance tables
- [ ] Time-sensitive items (driver/extension availability) carry re-verification dates
- [ ] Synthesis: ranked directions, per-direction ceiling on this frame, first experiment
- [ ] Adversarial self-check done; limitations documented
- [ ] PRISMA-ScR 20-item checklist 100 % ticked with file locations
