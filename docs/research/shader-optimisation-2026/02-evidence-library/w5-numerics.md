# W5 - Numerics and precision without changing the model

Author: orchestrator (Wave-1 W5 subagent stalled on the provider endpoint; compiled from
verified primary sources, 2026-10-17).

## Scope and search log

Queries: FP8/E4M3 weight-only GEMM on hardware without FP8; int8 tensor-core emulation of
higher precision (Ozaki); LUT-based dequant; int8 activation quantisation.
Retrieved and confirmed: Ozaki scheme II (arXiv 2504.08009), DGEMM on IMMU (arXiv 2306.11975),
Performance Enhancement of Ozaki on IMMU (arXiv 2409.13313), MARLIN (arXiv 2408.11743),
LUT-GEMM (arXiv 2206.09557, ICLR 2024). Plus W1 S1 (no FP8/E4M3 component type on DG2 coopmat).
Counts: found ~18 / screened 12 / included 6 / excluded 6 (FP8-hardware-only, or training-time).

## Sources

### S1. DG2 has no FP8 in the cooperative-matrix path (W1 cross-ref)
- See W1 S1/S2/S3. Mesa 25.2.8 `CMAT_XEHP_CONFIGURATIONS` lists only FLOAT16, BFLOAT16, SINT8,
  UINT8 - **no E4M3/E5M2**. `vulkaninfo` confirms no float8 extension on this box.
- Consequence: E4M3 weights (1 byte) cannot be consumed by DPAS directly; they must be expanded
  to FP16/BF16 (or emulated with int8).

### S2. Ozaki scheme II - GEMM-oriented emulation using integer modular technique
- arXiv: 2504.08009 (Ozaki, Uchino, Imamura), 2025. Retrieved: yes.
- Tier: T2 (arXiv, code referenced).
- Key claims:
  - Decomposes inputs into low-precision components so the product is "computed exactly through
    numerical operations"; uses **INT8 tensor cores** with INT32 accumulate.
  - "the first advantage of using INT8 TCs is that it is theoretically **twice or 4x as fast as
    FP16 TCs**"; "INT8-input IMMU is the most suitable for the Ozaki scheme in INT32-accumulation
    IMMUs."
  - Conventional Ozaki needs "7 to 8 slices, corresponding to 28 to 35 matrix multiplications"
    for FP64; scheme II reduces that via CRT.
  - Measured FP64 emulation: 7.4-9.8 TFLOPS on RTX 4090, 56.6-80.2 TFLOPS on GH200.
- Transferability to DG2/A770: partial. The *idea* - represent a wide-dynamic-range float product
  as sums of exact int8 products accumulated in int32 - maps to DG2's `{8,8,32 int8->int32}` DPAS,
  which IS advertised. The "2x the FP16 rate" is an arithmetic-density figure (NVIDIA there, Xe2
  in W1): measured on this A770's staged GEMM it is **0.66-0.98x** of f16 (Run 93), because both
  DG2 tiles are N=8 and the kernel is shared-memory-message bound. For our case the target is not FP64 but
  "E4M3 x FP16" (or FP16 with higher fidelity): E4M3 has a 3-bit mantissa and 4-bit exponent, so
  splitting E4M3 into an int8 slice is more plausible than FP64 emulation. Numbers are NVIDIA.
- Exact vs approximate: **exact** for the represented inputs (the whole point). If we use fewer
  slices it becomes approximate but controlled.

### S3. DGEMM on Integer Matrix Multiplication Unit
- arXiv: 2306.11975. Retrieved: yes.
- Tier: T2.
- Key claims:
  - "using IMMU, especially INT8-input-INT32-accumulation IMMU, we can reduce the memory
    consumption and the number of operations while keeping the same accuracy as the FMMU."
  - "The theoretical peak throughput of IMMUs is typically higher than that of FMMUs."
  - Outperforms cuBLAS DGEMM and the FP16-TC Ozaki implementation by up to ~6x on consumer GPUs.
- Transferability: same as S2. Confirms int8-INT32 DPAS is the right primitive for emulation. The
  memory-traffic claim does not transfer: no site in this frame is bandwidth-bound (37-250 GB/s
  against ~450), and the int8 operands are half the bytes but the kernel is bound on
  shared-memory messages, not DRAM (Run 93).

### S4. Performance Enhancement of the Ozaki Scheme on IMMU
- arXiv: 2409.13313. Retrieved: yes.
- Tier: T2.
- Key claims:
  - Splitting method matters: "the splitting method contributes to obtaining more accurate
    results than those of Ootomo's ozIMMU"; the FP64 accumulation line "accounts for a large ratio
    of the total computation time."
  - `ozIMMU` needs "at least 7 or 8 slices ... even for well-conditioned matrix multiplications."
- Transferability: cautionary - the *reduction/accumulation* cost can dominate. For us, if we
  emulate at int8, the epilogue accumulation must stay cheap (int32 accumulate in DPAS is free,
  so this is actually in our favour).

### S5. MARLIN - mixed-precision weight-only kernels
- arXiv: 2408.11743 (Frantar, Castro, Chen, Hoefler, Alistarh), 2024; code https://github.com/IST-DASLab/marlin/ . Retrieved: yes.
- Tier: T1/T2 (well-cited; production kernel in vLLM).
- Key claims:
  - Weight-only quantisation gives speedups "due to reduced memory movement"; close to ideal 4x
    for 4-bit weights up to batch 16-32, ~1.5x at batch 128; up to 2.8x end-to-end.
  - Holds as long as "less than 25-50 (tensor core) multiply-accumulates per 4-bit quantized
    weight"; requires maximising load width (128-bit) and reshuffling weights offline into
    contiguous 16x64 tiles.
- Transferability to DG2/A770: yes for the principle (weight-only quantisation pays only while
  memory-bound, which our frame is), partial for the kernel (CUDA/Ampere). The offline reshuffle +
  wide loads are directly applicable to staging E4M3 weights into SLM.
- Exact vs approximate: **approximate** (it accepts a quantisation accuracy trade-off). For our
  "model unchanged" constraint this is risky unless the E4M3 weights are the model's own format
  and dequant is exact.

### S6. LUT-GEMM - quantised matmul with lookup tables
- arXiv: 2206.09557 (Park et al., NAVER), ICLR 2024; code https://github.com/naver-aics/lut-gemm . Retrieved: yes.
- Tier: T1 (ICLR 2024, code).
- Key claims:
  - "eliminates the resource-intensive dequantization process"; 2.1x faster token generation vs
    OPTQ for 3-bit OPT-175B.
  - Pre-compute "2^mu possible values" of the activation sub-vector and use a key from the
    quantised weight bits to index the LUT.
- Transferability to DG2/A770: partial. The LUT idea could expand E4M3->FP16 in-register cheaply:
  E4M3 has 256 possible byte values; a 256-entry FP16 LUT indexed by the weight byte is a
  register/SLM-resident dequant. The paper's LUT is for dot-products, not elementwise decode,
  but the elementwise-LUT variant is the obvious adaptation for "keep weights 1 byte on device".
- Exact vs approximate: **exact** if the LUT stores the true FP16 value of every E4M3 byte (which
  it can, since E4M3->FP16 is a pure function). This is the cleanest way to keep the model's own
  E4M3 bytes on device and expand in-register.

## Variance table

| Source | Precision target | Primitive | Result | Exact? |
|---|---|---|---|---|
| S2 Ozaki II | FP64 | int8 TC | 56-80 TF (GH200) | exact (enough slices) |
| S3 DGEMM IMMU | FP64 | int8 TC | up to 6x vs cuBLAS | exact |
| S4 Ozaki perf | FP64 | int8 TC | accumulation dominates | exact |
| S5 MARLIN | weight-only 4-bit | FP16xINT4 | 1.5-4x | approximate |
| S6 LUT-GEMM | 3-bit weights | LUT | 2.1x | exact for stored format |

Key variance: S2/S3 are about *raising* precision with int8; S5 lowers it for speed. For our
"model unchanged" constraint the relevant ones are S2/S3/S6 (exact expansions), not S5.

## Practical implications for this frame

1. Direction: keep the model's E4M3 weights at 1 byte in device memory and expand to FP16 in
   registers via a 256-entry LUT (S6 adapted). Mechanism: load 1 byte/weight, index a
   register-resident LUT, feed DPAS. Attacks: the 292 MB of FP16 weights streamed per frame
   (weights are the dominant DRAM traffic in a bandwidth-bound frame). Ceiling: halves weight
   bytes -> up to ~2x on the weight-streaming portion; exact, no quality change. Effort: medium.
   Risk: low. FIRST falsifiable experiment: measure the per-frame DRAM bytes for weights under
   (a) FP16 decode pass and (b) direct 1-byte LUT decode in the GEMM; compare total bytes.
2. Direction: emulate the E4M3 x FP16 product on DG2's int8->int32 DPAS (S2/S3 mechanism).
   Mechanism: split the E4M3 operand into int8 slices with per-exponent scaling, accumulate in
   int32. Attacks: uses the int8 DPAS at 2x the FP16 rate. Ceiling: 2x DPAS throughput *if* the
   number of slices stays low (E4M3's small mantissa helps); the reduction cost (S4) bounds it.
   Effort: high. Risk: high (numerics). FIRST falsifiable experiment: for a representative E4M3
   weight tile, find the minimum slice count that reproduces FP16 accumulation bit-for-bit; if
   <=2 slices, the win is real.
3. Direction: do NOT quantise activations (S5 is approximate + S2/S4 warn outlier cost). Keep the
   model unchanged per the brief.

## Excluded leads

- FP8 hardware paths (Hopper/Blackwell FP8 TC) - DG2 has none (S1).
- SmoothQuant/QuaRot activation quantisation - approximate, violates "model unchanged".
- FP16-accumulate DPAS via coopmat - not advertised on DG2 (W1 S1); only reachable via OpenCL.