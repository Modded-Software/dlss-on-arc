# W6 - Temporal/spatial redundancy and multi-GPU

Author: orchestrator (Wave-1 W6 subagent stalled on the provider endpoint; compiled from
verified primary sources, 2026-10-17).

## Scope and search log

Queries: training-free temporal redundancy for video networks (token reuse, feature caching);
multi-GPU spatial/pipeline parallelism for U-Net/DiT.
Retrieved and confirmed: Eventful Transformers (arXiv 2308.13494), DistriFusion (arXiv 2402.19481,
CVPR 2024), PipeFusion (arXiv 2405.14430, NeurIPS 2025), DeepCache (arXiv 2312.00858, CVPR 2024).
Counts: found ~16 / screened 11 / included 5 (incl. the project's own alternate-frame result) /
excluded 6 (training-time, or not transferable).
Wave 2 (gap-fill): +3 sources (S6 DeltaCNN, S7 Skip-Convolutions, S8 redundancy-aware VSR); total = 8.

## Sources

### S1. Eventful Transformers - temporal redundancy in vision transformers
- arXiv: 2308.13494 (Dutson, Li, Gupta, ICCV 2023). Retrieved: yes.
- Tier: T1 (ICCV 2023, code).
- Key claims:
  - Identifies and re-processes "only those tokens that have changed significantly over time";
    "can be converted from existing Transformers (often **without any re-training**)".
  - "significant computational savings (on the order of **2-4x**) with only minor reductions in
    accuracy"; GPU runtime speedup "up to **1.74x**", CPU 2.48x.
  - The query-key product updates cost `2NMD` vs `N^2 D`, so save when `M < N/2`; it exploits both
    changed rows and changed columns of K, scattering into the old attention map.
- Transferability to DG2/A770: yes (training-free, token-level). Our network is temporal (frame N+1
  consumes frame N). Caveat: "model unchanged" - the gates add a threshold but no new weights if
  converted without retraining; accuracy must be measured on our restoration metric.
- Exact vs approximate: **approximate** (only re-processing changed tokens). Must be validated
  against the quality metric (the project already measures 44-52 dB on alternate frames).

### S2. DistriFusion - displaced patch parallelism
- arXiv: 2402.19481 (MIT-HAN Lab, CVPR 2024); code https://github.com/mit-han-lab/distrifuser . Retrieved: yes.
- Tier: T1 (CVPR 2024, code).
- Key claims:
  - Splits the input into patches, one GPU each; "reusing the pre-computed feature maps from the
    previous timestep to provide context for the current step" -> "asynchronous communication,
    which can be pipelined by computation."
  - "up to a **6.1x speedup on eight NVIDIA A100s** compared to one" with no quality loss on
    SDXL.
- Transferability to DG2/A770: partial. Our setup is 2 GPUs, so the ideal 2x is the target, not
  6.1x. The key mechanism - reuse *stale* (previous-frame) features across the patch boundary so
  the halo exchange is asynchronous - is exactly right, and the project already has a 2-daemon
  alternate-frame mode. DistriFusion formalises how to keep it asynchronous and quality-preserving.
- Exact vs approximate: approximately (stale features at the halo). Measured as no quality loss.

### S3. PipeFusion - patch-level pipeline parallelism
- arXiv: 2405.14430 (Fang, Pan, Li, Sun, Wang), NeurIPS 2025. Retrieved: yes.
- Tier: T1 (NeurIPS 2025, code).
- Key claims:
  - Partitions image into patches AND model layers across GPUs; "reuses one-step stale feature
    maps to provide context for the current pipeline step."
  - "notably reduces communication costs compared to ... DistriFusion"; "suitable for DiTs
    inference to be hosted on GPUs connected via **PCIe**."
  - DistriFusion replicates parameters and allgathers activations every layer; PipeFusion *shards
    parameters* and uses async P2P for activations.
- Transferability to DG2/A770: yes for the PCIe point - our two A770s are PCIe (no NVLink), so
  per-layer allgather is expensive. Sharding parameters and passing one patch's activations per
  stage is the cheaper topology. Quantify for 1024x576 below.

### S4. DeepCache - training-free U-Net feature caching
- arXiv: 2312.00858 (Ma, Fang, Wang, CVPR 2024). Retrieved: yes.
- Tier: T1 (CVPR 2024, code).
- Key claims:
  - Training-free; reuses high-level U-Net features across denoising steps, updates low-level ones
    cheaply: "**2.3x** for Stable Diffusion v1.5 with only a 0.05 CLIP decline"; 4.1x for LDM-4-G.
  - Non-uniform 1:N caching strategy for long intervals.
- Transferability to DG2/A770: partial. This is a *diffusion* iteration structure (many steps of
  the same U-Net); our frame-to-frame temporal chain is one step. Still, the mechanism - cache the
  expensive high-level (deep, C=1024) features, recompute the cheap shallow ones - is directly
  usable because the brief says deep-level caching gave only 1.13x and the shallow levels dominate.
  DeepCache's non-uniform schedule may explain when deep caching *does* pay.
- Exact vs approximate: approximate (cached features). Quality-measured.

### S5. Project's own alternate-frame rendering (baseline, closed-ish)
- From the brief/diary: two daemons, alternate-frame rendering, 44-52 dB.
- This is a 2-GPU temporal split already in place. W6's job is to say whether DistriFusion/PipeFusion
  style patch splitting can beat it or combine with it.

## Variance table

| Source | Redundancy exploited | Regime | Gain | Exact? |
|---|---|---|---|---|
| S1 Eventful | temporal token change | video ViT | 1.74x GPU | approx |
| S2 DistriFusion | previous-step features at halo | diffusion patches | 6.1x / 8 GPU | approx |
| S3 PipeFusion | previous-step features (pipeline) | DiT patches | SotA on 8xL40 PCIe | approx |
| S4 DeepCache | U-Net high-level features | diffusion steps | 2.3-4.1x | approx |
| S5 project | alternate frames | our frame | 44-52 dB | approx |

All are approximate (quality-gated) except none are exact.

## Practical implications for this frame

1. Direction: DistriFusion-style displaced patch parallelism across the two A770s.
   Mechanism: split the 1024x576 frame into 2 patches; each GPU computes its patch using the
   previous frame's halo features (async, no per-frame PCIe sync). Attacks: halves per-GPU compute.
   Ceiling ~2x (2 GPUs) if communication is fully hidden; the project's alternate-frame mode is
   already near this, so the marginal gain is only if it lets each GPU keep a full-time queue.
   Effort: medium-high. Risk: quality at the halo. Quantify: for a spatial split of 1024x576, the
   halo activations that must cross PCIe per layer are `(C * halo_width * height)`; for shallow
   levels (C=32) this is tiny, for C=1024 it is large - so a *layer split* (S3) suits the deep
   levels and a *spatial split* (S2) suits the shallow ones. FIRST falsifiable experiment: measure
   PCIe bytes/frame for (a) spatial split with halo and (b) layer split, at 1024x576.
2. Direction: Eventful-Transformer token gating (training-free) on the temporal chain.
   Mechanism: re-process only tokens that changed vs the reference frame; recompute only changed
   rows/cols of the attention map. Attacks: full-frame compute on static regions. Ceiling: up to
   2-4x for mostly-static content with minor accuracy cost. Effort: medium-high. Risk: quality.
   FIRST falsifiable experiment: measure the fraction of unchanged tokens between consecutive
   frames in the game; if >50% (the Eventful break-even), prototype gating on one block.
3. Direction: combine deep caching (S4) with the already-known 1.13x result.
   Mechanism: cache the deep (C=1024) features, recompute shallow. Attacks: the deep-level compute
   that is currently only 1.13x improved. Ceiling: S4's 2.3x is for many diffusion steps; our
   1-step chain likely sees much less - treat as low priority.

## Wave 2 gap-fill (added 2026-10-17, orchestrator)

Wave 2 targeted the leads left unverified in Wave 1. Retrieved and confirmed:

### S6. DeltaCNN - end-to-end sparse frame-difference CNN inference
- arXiv: 2203.03996 (Parger et al., CVPR 2022); code https://github.com/facebookresearch/DeltaCNN (CUDA). Retrieved: yes.
- Tier: T1 (CVPR 2022, open code).
- Key claims:
  - "sparse frame-by-frame updates ... propagate sparse feature updates end-to-end - **without
    accumulating errors over time**"; "applicable to all convolutional neural networks **without
    retraining**"; "speedups of up to **7x**" over cuDNN "with only marginal differences in accuracy".
  - Truncates insignificant updates by auto-tuning a per-layer threshold `eps` "to allow each
    truncation layer to contribute equally to the output error" (total loss increase capped at 3%).
  - Explicitly **does not support Tensor Cores**: "performance comparisons against cuDNN should be
    done in 32 bit mode for apples to apples".
- Transferability to DG2/A770: **partial and risky.** The backward-explicit note (no tensor cores)
  is decisive for us: our whole premise is XMX. A tensor-core-free sparse path wins only where the
  dense reference is also non-XMX; on DG2 the dense reference IS XMX (DPAS), so the 7x cuDNN gap
  does not transfer. The *mechanism* (mask + cache, one dense frame then sparse deltas) is sound but
  the *implementation* is CUDA-only and tensor-core-free.
- Exact vs approximate: approximate (thresholded updates), error-gated.

### S7. Skip-Convolutions - residual-frame convolution with magnitude gating
- arXiv: 2104.11487 (Habibian et al., CVPR 2021); code https://github.com/Qualcomm-AI-research/Skip-Conv . Retrieved: yes.
- Tier: T1 (CVPR 2021, open code).
- Key claims:
  - Reformulates convolution on residual frames; per-layer binary gate ("Norm gate" - no training,
    plug into a trained net) or "Gumbel gate" (trainable).
  - "reduce their computational cost consistently by a factor of **3~4x** ... **without any accuracy
    drop**" on EfficientDet/HRNet.
  - Requires **block-wise structured sparsity** because "many hardware platforms perform the
    convolutions distributed over small patches (e.g. 8x8), so cannot use any fine-grained
    spatial sparsity smaller than these block sizes."
  - Known limitation: "unclear how our model would perform in the presence of severe camera motion."
- Transferability to DG2/A770: partial. The Norm gate is training-free and the 8x8 block-sparsity
  requirement is exactly the granularity XMX tiles want (DG2 DPAS tile is 8x8x16) - so unlike
  DeltaCNN this *can* retain XMX. But it is a **CNN** mechanism (residuals do not survive the
  LayerNorm/softmax of a ViT/Swin block), so it applies to the CNN-ish stages, not the attention.
- Exact vs approximate: approximate (gated skips); Norm-gate variant needs no retraining.

### S8. Redundancy-aware inference / kernel-split VSR (adjacent, directional)
- "Real-Time Video Super-Resolution with Spatio-Temporal Modeling and Redundancy-Aware Inference",
  MDPI Sensors 23(18):7880, 2023 (doi 10.3390/s23187880); and "Kernel Dimension Matters" ACM MM
  2023 (doi 10.1145/3581783.3611908). Retrieved: yes (abstract-level).
- Tier: T2.
- Key claims: caching and reusing intermediate spatial features across adjacent frames "improves the
  latency for inferring a frame ... to" only the new frames' extraction; real-time 720p with
  "minimal impact on accuracy". Mechanism = feature cache + reuse, same family as S4/S6.
- Transferability: directional only (super-resolution task, not restoration); confirms the
  feature-cache mechanism is broadly used but adds no new lever beyond S4/S6.

### Wave 2 variance additions

| Source | Redundancy exploited | Regime | Gain | Tensor cores kept? | Exact? |
|---|---|---|---|---|---|
| S6 DeltaCNN | sparse pixel deltas, cached maps | video CNN | 7x vs cuDNN | **No (explicit)** | approx |
| S7 Skip-Conv | residual-frame magnitude | video CNN | 3-4x, no acc. drop | Yes (8x8 blocking) | approx |
| S8 RAI/VSR | intermediate feature cache | video SR | real-time 720p | n/a (directional) | approx |

**Wave-2 verdict:** the "unverified leads" resolve to *concept-adjacent, not adoptable-as-is*. DeltaCNN
is disqualified by its explicit tensor-core-free design; Skip-Convolutions is the useful one but only
for the CNN stages and only in its training-free Norm-gate form. No change to the D1-D9 ranking: the
fused per-window kernel (D1) and Stream-K/persistent (D2) remain the top exact-ish levers.

## Excluded leads

- Gemini "compute-on-the-fly" (3D Gaussian Splatting) - retrieved searches point to 3DGS rendering
  work (Kerbl et al.; CTRL-GS arXiv 2505.18306), not a training-free CNN/ViT redundancy method.
  **Out of scope**, excluded after inspection, not refuted.
- Training-based token merging (ToMe) - requires adaptation/retraining.
- Tensor parallelism for the 2 GPUs - our per-layer GEMMs are already too small to split further
  usefully (`N=16..128`), so TP would fragment them.
- DeltaCNN as an *implementation* - CUDA-only and tensor-core-free (see S6); kept as evidence, not
  as a candidate.