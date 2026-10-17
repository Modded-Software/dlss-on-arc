# 01 - Protocol: Shader and execution-engine optimisation for DLSS-NR on Arc A770 (scoping review)

Standard: **PRISMA-ScR** (Tricco AC et al., *Ann Intern Med* 2018;169:467-473,
doi:10.7326/M18-0850), adapted to a technical literature survey per the workspace methodology
(`WORKSPACE-WEB-CONTENT/docs/instructions/research-methodology.md`). Stage 0 brief: `00-brief.md`.
Protocol written 2026-10-17 before any searching (gate).

## 1. Title
Scoping review of GPU kernel, compiler and execution-engine techniques (2016-2026, emphasis
2022-2026) applicable to running a fixed Swin/ViT U-Net (DLSS-NR) near XMX peak on Intel Xe-HPG.

## 2. Rationale
The network runs at ~3-6 % of the A770's FP16 XMX peak in-frame. 70+ diary runs exhausted
local knob-tuning; the owner asks for the full map of directions, including a rethink of the
execution engine on the assumption that the current one is naive.

## 3. Objectives - PCC
- **Population:** inference of transformer/CNN-hybrid vision networks with small head dim, small
  channel counts and huge token counts, on GPUs with matrix engines (XMX / Tensor Cores / WMMA).
- **Concept:** techniques that raise achieved throughput or cut work without retraining.
- **Context:** Intel Xe-HPG DG2 (A770 x2), Vulkan/OpenCL/SYCL on Linux, real-time per-frame budget.

Workstreams (one evidence file each):
| # | Workstream | Sub-questions |
|---|---|---|
| W1 | Xe-HPG/DG2 matrix-engine programming | DPAS lowering in IGC/ANV; coopmat limits; ESIMD/joint_matrix/SPIR-V ext; GRF/SIMD modes; SLM/L1/L3; what oneDNN/sycl-tla/XeTLA do on DG2 |
| W2 | GEMM kernel SotA for skinny / small-K / batched shapes | stream-K, split-K, tile scheduling, epilogue fusion, layout (VNNI/swizzle), double buffering, autotuning |
| W3 | Execution-engine overhaul | megakernels/persistent kernels, dataflow/tile scheduling, graph compilers (TVM/Ansor, Welder, Souffle, Mirage, XLA), dispatch-overhead removal (Vulkan DGC, command buffers), concurrency |
| W4 | Attention and Swin-block kernels | FlashAttention 1/2/3, window attention, small head-dim, relative bias / cosine attention, whole-block fusion |
| W5 | Numerics and precision (model-preserving) | FP8/E4M3 weights on int8/fp16 engines, LUT dequant, fp16 accumulation, weight-only quant, exactness |
| W6 | Temporal / spatial redundancy and multi-GPU | video delta compute, feature/activation caching, token reuse, tiled dirty-region recompute, pipeline/tensor/spatial parallelism over PCIe |
| W7 | Measurement, roofline, autotuning on Intel | Intel GPU roofline, VTune/unitrace/pti, IGC dumps, OA counters, autotuners (Kernel Tuner, BaCO, KTT) |

## 4. Eligibility
**Include:** peer-reviewed papers, arXiv preprints, vendor documentation, specs (Khronos), compiler
/driver source, maintained repos with benchmarks; the technique must be applicable to a fixed
pretrained network. **Exclude:** methods requiring retraining/fine-tuning or architectural change
(except as "context, not adoptable"); marketing without technical content; unverifiable claims;
sources not retrieved. Language: English. Dates: any, SotA emphasis 2022-2026.

## 5. Information sources and search
Channels: arXiv (primary for SotA), ACM/IEEE/USENIX proceedings pages, Intel developer docs,
Khronos registry, GitHub (intel/llvm, intel-graphics-compiler, mesa, oneDNN, sycl-tla), web search.
Query patterns per workstream, e.g. W1 `"DPAS" "Xe-HPG" GEMM`, `ESIMD DPAS DG2`, `joint_matrix
Arc A770`, `ANV cooperative matrix lowering`; W2 `"Stream-K"`, `skinny GEMM tensor core`,
`epilogue fusion CUTLASS`; W3 `megakernel inference`, `persistent kernel transformer`, `Mirage
superoptimizer`, `VK_EXT_device_generated_commands compute`; W4 `FlashAttention window attention
Swin kernel`, `small head dimension attention kernel`; W5 `FP8 weights int8 tensor core
dequantization LUT`; W6 `video inference temporal redundancy delta transformer training-free`,
`spatial partition multi-GPU CNN halo`; W7 `Intel GPU roofline Advisor`, `GPU kernel autotuning
Bayesian`. Searches executed by parallel subagents (Wave 1) plus orchestrator cross-check searches.

## 6. Selection and appraisal
Single reviewer screens title/abstract, then full text for anything that drives a recommendation.
Tiers (methodology Stage 3, adapted):
- **T1** - vendor/spec/compiler-source documentation (Intel, Khronos, IGC/Mesa source), peer-reviewed
  venue papers with artefacts, or measurements on this box.
- **T2** - arXiv preprints from established groups with code/benchmarks; university reports.
- **T3** - blogs, forum/issue threads, vendor marketing - directional only.
A critical recommendation needs T1, or >=2 independent T2.

## 7. Data charting
Fields per source: name; authors/publisher; year/version; URL / arXiv id; tier + reason;
atomic key claims (numbers verbatim); hardware the claim was measured on; transferability to DG2
(yes / partial / no + why); exact vs approximate; recency flag (re-verify date); conflicts.

## 8. Synthesis
Narrative synthesis organised by direction (not by source) in `03-synthesis-guide.md`: for each
direction - mechanism, evidence tier, transferability, ceiling on this frame (from the diary's
measured costs), effort, risk, first falsifiable experiment. Ranked table; variance tables for
conflicts; "closed levers" table; escalation triggers.

## 9. Results table (planned vs actual)
| WS | Planned sources | Actual | File |
|---|---|---|---|
| W1 | 10-15 (>=6 T1) | 7 (T1: S1,S2,S3,S4) | `02-evidence-library/w1-xe-hpg-xmx.md` |
| W2 | 10-15 | 7 (S7 MCFuser added Wave 2) | `w2-gemm-sota.md` |
| W3 | 10-15 | 6 | `w3-execution-engine.md` |
| W4 | 8-12 | 6 | `w4-attention-swin.md` |
| W5 | 8-12 | 6 | `w5-numerics.md` |
| W6 | 8-12 | 8 (S6-S8 added Wave 2) | `w6-temporal-multigpu.md` |
| W7 | 6-10 | 4 | `w7-measurement-autotuning.md` |

**Actual total: 44 sources** (target 60-91); Wave 1 = 40, Wave 2 gap-fill = +4. Shortfall cause and honesty note: the seven Wave-1
research subagents were launched via `scripts/ask-claude.sh` but the model endpoint stopped
responding (all sessions idle in `epoll_wait`, logs frozen for 9-10 minutes; direct orchestrator
web tooling still worked). The orchestrator therefore assembled every evidence file itself from
directly-retrieved sources, and had to fit within that budget. Sources named as leads but not
retrieved were **excluded, not refuted** (see each file's "Excluded leads"). Stage 6 existence
verification: all 18 arXiv ids cited across the library were re-fetched via the arXiv API and all
18 resolved to the expected titles (2026-10-17).

## 10. Deviations and limitations (declared up front, extended at the end)
Single reviewer, no dual screening. Web-only access (paywalled full texts read via arXiv versions
where they exist). Subagent-collected citations are re-verified by the orchestrator (Stage 6) before
they can carry a recommendation. No new on-box measurement is part of this review; ceilings are
projections from the diary's measured costs.
