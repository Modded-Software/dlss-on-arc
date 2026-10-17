# W1 - Xe-HPG/DG2 matrix-engine programming

Author: orchestrator (Wave-1 subagent for W1 stalled on the provider endpoint; this file
was compiled by the orchestrator from primary driver/compiler source and vendor docs,
date 2026-10-17).

## Scope and search log

Queries / retrievals run (2026-10-17), all fetched and confirmed:
- Local primary source: Mesa 25.2.8 (`mesa-25.2.8` @ `f97d7dcca0e51d7e20ed4342bb5ea8afea7db7fa`),
  files `src/intel/dev/intel_device_info.c`, `src/intel/vulkan/anv_physical_device.c`,
  `src/intel/compiler/brw_nir_lower_cooperative_matrix.c`, `src/intel/compiler/brw_lower_dpas.cpp`.
- `vulkaninfo` on this box (GPU0/GPU1 = Arc A770 DG2).
- Intel oneAPI GPU Optimization Guide: "Boost Matrix Multiplication Performance with Intel Xe
  Matrix Extensions"; "Programming Intel XMX Using SYCL Joint Matrix Extension" (2024-1, 2025-0).
- intel/tiny-dpcpp-nn (GitHub) - joint_matrix network on Arc 770.
- oneDNN benchdnn GEMM/matmul strategy commits (DG2/Xe2, f16).
- web search: oneDNN A770 XMX, SYCL joint_matrix DPAS Arc, XeTLA/sycl-tla, ESIMD DPAS DG2.

Counts: found ~20 / screened 12 / included 7 (below) / excluded 5 (PVC-only or no DG2 data).

## Sources

### S1. Mesa 25.2.8 cooperative-matrix device configuration (DG2)
- URL: local tree `refs/dlss-on-arc/notes/mesa-25.2.8-src/src/intel/dev/intel_device_info.c`; upstream
  https://gitlab.freedesktop.org/mesa/mesa/-/blob/mesa-25.2.8/src/intel/dev/intel_device_info.c ; Retrieved: yes
- Tier: T1 - compiler/driver source.
- Key claims (verbatim from source):
  - `DG2_CONFIG` -> `XEHP_FEATURES` -> `CMAT_XEHP_CONFIGURATIONS`:
    - `{ SUBGROUP, 8, 8, 16, FLOAT16, FLOAT16, FLOAT32, FLOAT32 }`
    - `{ SUBGROUP, 8, 8, 16, BFLOAT16, BFLOAT16, FLOAT32, FLOAT32 }`
    - `{ SUBGROUP, 8, 8, 32, SINT8, SINT8, SINT32, SINT32 }`
    - `{ SUBGROUP, 8, 8, 32, UINT8, UINT8, UINT32, UINT32 }`
  - The older `CMAT_PRE_XEHP_CONFIGURATIONS` (pre-Xe-HPG) *does* contain an
    `{ 8,8,16, FLOAT16,FLOAT16,FLOAT16,FLOAT16 }` (fp16 accumulate) entry, but the Xe-HPG set
    used by DG2 **does not**. So on DG2, ANV advertises no f16-accumulate coopmat.
  - Xe2 (`CMAT_XE2_CONFIGURATIONS`) uses 8x16x16, i.e. a *larger* DPAS tile than DG2.
- Transferability to DG2/A770: yes - this *is* DG2.
- Exact vs approximate: exact (type tiles only, no numerics change).
- Recency flag: 25.2.8 is the shipped version on this box; re-verify if ANV bumps to 26.x.
- Conflicts: none.

### S2. ANV cooperative-matrix exposure
- URL: `.../src/intel/vulkan/anv_physical_device.c`; `.../src/intel/vulkan/anv_device.c`; Retrieved: yes
- Tier: T1.
- Key claims:
  - `VK_KHR_cooperative_matrix` is advertised iff
    `devinfo->cooperative_matrix_configurations[0].scope != INTEL_CMAT_SCOPE_NONE`.
  - `cooperativeMatrixSupportedStages = VK_SHADER_STAGE_COMPUTE_BIT` (compute only).
  - `convert_component_type` maps INTEL types to `VK_COMPONENT_TYPE_{FLOAT16,FLOAT32,SINT32,SINT8,UINT32,UINT8,BFLOAT16}_KHR`
    -> there is no FP8 / E4M3/E5M2 component type in the Vulkan coopmat path.
  - A helper `anv_device_has_bfloat16_cooperative_matrix` exists; bf16 is advertised where the
    device config lists BFLOAT16.
- Transferability: yes - DG2.
- Recency: 25.2.8.

### S3. Live driver capability on this box
- URL: `vulkaninfo` output captured to `/tmp/opencode/orch-vulkaninfo.txt`; Retrieved: yes (2026-10-17)
- Tier: T1 - measured on the actual device used by the project.
- Key claims:
  - GPU0 and GPU1 both `deviceName = Intel(R) Arc(tm) A770 Graphics (DG2)`;
    `driverInfo = Mesa 25.2.8-0ubuntu0.24.04.2`.
  - `VK_KHR_cooperative_matrix : extension revision 2`, `cooperativeMatrix = true`,
    `cooperativeMatrixRobustBufferAccess = false`.
  - `subgroupSize = 32`, `minSubgroupSize = 8`, `maxSubgroupSize = 32`, `subgroupSizeControl = true`.
  - No `VK_EXT_shader_float8`, no `VK_KHR_shader_bfloat16`, no coopmat2 observed.
- Transferability: yes - this is the review's target hardware.
- Exact vs approximate: exact.

### S4. Intel oneAPI GPU Optimization Guide - XMX / joint_matrix
- URL: https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-0/programming-intel-xmx-using-sycl-joint-matrix.html
  and https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2024-1/xmx.html ; Retrieved: yes
- Tier: T1 (vendor).
- Key claims:
  - "DPAS (Dot Product and Accumulate Systolic) is the name of the elementary operations done in Intel XMX."
  - "There is no emulation or alternate path in the joint matrix implementation."
  - Tuning guidance: "One sub-group can perform multiple DPAS operations"; block over i,j,k;
    example `256x256x32` global with one sub-group doing `64x32x32`; "Large register file per
    thread gives the best results" (`-ze-opt-large-register-file`).
  - XMX peak table (PVC 1550, not DG2): fp32 26 TF (98% of fp32 peak); bf16 151 TF; bf16x2 74 TF;
    bf16x3 42 TF.
- Transferability to DG2/A770: partial - guidance is generic Xe, **peak numbers are PVC**; and
  `-ze-opt-large-register-file` is NOT available on DG2 (no 256-GRF mode).
- Exact vs approximate: guidance only.

### S5. intel/tiny-dpcpp-nn - joint_matrix on Arc 770
- URL: https://github.com/intel/tiny-dpcpp-nn ; Retrieved: yes
- Tier: T2 - vendor repo with Arc 770 (`ACM`) target and benchmarks.
- Key claims:
  - Uses SYCL `joint_matrix` (wrapper over DPAS) with activations/weights staged in L1 and registers.
  - "We reach up to 20x compared to PyTorch" on Arc 770 (ACM); up to 60x on PVC.
  - `TARGET_DEVICE` includes `ACM` for Arc A-series (A570, A770) and Flex 140/170.
- Transferability to DG2/A770: yes - explicitly targets ACM. Caveat: tiny MLP (training/inference),
  not the skinny GEMM shapes of this frame; the 20x is vs PyTorch, not vs DPAS peak.
- Exact vs approximate: exact.

### S6. FlashAttention-2 (context for tile/work partitioning; see W4)
- URL: https://arxiv.org/abs/2307.08691 ; Retrieved: yes
- Tier: T1 (peer-reviewed/venue paper, code).
- Key claims: reaches 50-73% of theoretical max FLOPs/s on A100; parallelise over sequence length;
  "typically choose blocks of size {64,128}x{64,128}" for head dim 64/128.
- Transferability: partial (see W4).

### S7. oneDNN DG2/Xe2 GEMM strategies (f16)
- URL: https://github.com/uxlfoundation/oneDNN/pull/3479 (Xe2 f16) ; Retrieved: yes
- Tier: T1 (vendor library source/CI perf).
- Key claims: benchdnn `f16:f16:f16` matmul `128x8192:8192x1024` => ~38.3 TFLOP/s (Xe2 benches);
  strategies are added per-arch (`DG2+Xe2 2nd token dynamic quant strategies`).
- Transferability: partial - numbers are Xe2 (tile 8x16x16), used only as an order-of-magnitude
  reference for library-grade f16 GEMM on Intel GPUs.

## Variance table

| Claim | S1/S2 (ANV 25.2.8) | Implication |
|---|---|---|
| Shared-memory commit point | subgroup-scope configs only | coopmat must be compute-stage, subgroup 32 |
| f16 accumulate | not in DG2 advertised set | f16-acc DPAS must go via OpenCL/IGC or a compiler path |
| FP8/E4M3 | absent | FP8 must be emulated (see W5) |
| Tile | 8x8x16 (f16), 8x8x32 (int8) | skinny N is quantised hard by 8/16 |

No source contradicts another; the oneDNN/XMX peak numbers are PVC/Xe2, not DG2 (flagged).

## Practical implications for this frame

1. Direction: restrict the Vulkan coopmat proof-of-concept to what DG2 actually advertises.
   Mechanism: use `{8,8,16 f16->f32}` and `{8,8,32 int8->int32}` only; do not attempt
   `f16 accumulate` or FP8 through coopmat. Attacks: avoids wasted PoC effort. Ceiling: none
   directly. Effort: low. Risk: low. First falsifiable experiment: compile a coopMatMulAdd with
   c/result = f16 on this box and confirm the pipeline fails validation / is not in the
   advertised set.
2. Direction: obtain f16-accumulate DPAS (2x the f32-acc accumulate density) via OpenCL/IGC rather
   than Vulkan. Mechanism: `cl_intel_subgroup_matrix_multiply_accumulate` already available (no
   `2d_block_io`, no `buffer_prefetch` on DG2). Attacks: the DPAS-issue ceiling. Ceiling: up to
   ~2x the f32-acc DPAS rate if accuracy allows (see W5). Effort: medium (two runtime paths).
   Risk: accuracy. First falsifiable experiment: OpenCL microbench of f16-acc vs f32-acc DPAS at
   frame shapes; measure TFLOP/s and output error.
3. Direction: keep the register-resident approach; the "large register file" advice does not apply
   to DG2. Mechanism: stay at 128 GRF and instead raise DPASs-in-flight per sub-group. Attacks:
   issue-bound loop. Ceiling: the register-resident probe (~86 TF) is the realistic upper bound.
   Effort: low. Risk: low.

## Excluded leads

- PVC/Xe2-only oneDNN/XMX peak tables (151 TF bf16) - wrong arch, kept only as context (S4/S7).
- `-ze-opt-large-register-file` - DG2 has no 256-GRF mode (already closed).
- AMD zen/gfx coopmat papers - not Xe.