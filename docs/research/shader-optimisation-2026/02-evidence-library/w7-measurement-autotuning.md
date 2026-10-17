# W7 - Measurement, roofline and autotuning on Intel GPUs

Author: orchestrator (Wave-1 W7 subagent stalled on the provider endpoint; compiled from
verified primary sources, 2026-10-17).

## Scope and search log

Queries: Intel Advisor GPU Roofline / XMX support; VTune/PTI-GPU tools on Linux; GPU hardware
counters on Vulkan/OpenCL; Mesa driver debug tools; autotuning / LM kernel generation.
Retrieved and confirmed: Intel PTI-GPU (unitrace, gpuinfo, instcount, cl_gpu_metrics,
ze_metric_query), Intel Advisor GPU Roofline + release notes (XMX on Arc), Mesa
`VK_KHR_pipeline_executable_properties`, KernelBench (arXiv 2502.10517).
Counts: found ~15 / screened 10 / included 6 / excluded 4 (CPU-only or NVIDIA-only tooling).

## Sources

### S1. Intel PTI-GPU (Profiling Tools Interfaces for GPU)
- https://github.com/intel/pti-gpu ; unitrace README https://github.com/intel/pti-gpu/blob/master/tools/unitrace/README.md
  docs https://intel.github.io/pti-gpu/ . Retrieved: yes.
- Tier: T1 (Intel vendor tool, open source).
- Key claims:
  - `unitrace` "traces and profiles host/device activities ... for Intel GPU applications"; works
    with Level Zero AND OpenCL ("To enable OpenCL tracing and profiling, please use --opencl").
  - `--device-timing`, `--kernel-submission` (queued/submit/execute intervals),
    `--device-timeline`, `--chrome-kernel-logging` (Perfetto JSON).
  - Toolset includes `gpuinfo` ("list of HW metrics one can collect"), `instcount` ("prints GPU
    kernel assembly (GEN ISA) annotated by instruction execution count"), `sysmon`, and
    `cl_gpu_metrics` / `ze_metric_query` ("hottest GPU kernels along with percent of cycles it was
    active, stall and idle").
  - "you can run the tool unitrace to profile GPU kernels at **instruction level**, pinpoint
    instructions that stall the execution unit or the vector engine and **report reasons of the
    stalls**."
  - Enabling metrics may need `/proc/sys/dev/i915/perf_stream_paranoid` (i915) or
    `/proc/sys/dev/xe/observation_paranoid` (xe) set to 0.
- Transferability to DG2/A770: **yes - this is the primary measurement toolkit for our box**, and it
  spans both our runtimes (Vulkan via Level Zero? note: our Vulkan path is ANV, not L0 - see caveat;
  OpenCL via cl_gpu_metrics directly). Actionable now.
- Caveat: unitrace's device metrics are Level Zero/OpenCL; for the Vulkan (ANV) path we may need
  Mesa's own counters (S3) or correlate via `sysmon`/`intel_gpu_top`.

### S2. Intel Advisor GPU Roofline
- https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2024-1/advisor-roofline.html
  and release notes https://www.intel.com/content/www/us/en/developer/articles/release-notes/advisor-release-notes.html . Retrieved: yes.
- Tier: T1 (vendor).
- Key claims:
  - "Intel Advisor can generate a roofline model for kernels running on Intel GPUs", showing
    whether a kernel is bounded by L3 bandwidth, HBM, or compute (CARM dot = registers<->L1).
  - Report shows **XVE usage** per GPU kernel.
  - Release notes: GPU roofline "added support for **Xe Matrix Extensions (XMX) for Intel Arc
    Graphics**".
  - Known caveat in notes: "Advisor Tripcounts collection might not run as expected on Intel Iris
    XE Graphics GPU and Intel Data Center GPU Flex Series devices ... may result in not generating
    GPU Roofline."
- Transferability to DG2/A770: yes (Arc XMX supported). Requires DPC++/oneAPI. Gives the exact
  roofline verdict (DRAM vs L3 vs compute) that our frame needs.

### S3. Mesa ANV / IGC introspection
- From W1: ANV exposes `VK_KHR_pipeline_executable_properties` (compiler stats) and honors
  `INTEL_DEBUG` (e.g. `shader`, `scheduler`, `perf`); IGC/OpenCL exposes shader dumps and vISA
  stats (spills, SWSB) - see the project's prior notes (`phase*`, `research-xe-hpg-xmx.md`).
- Also `intel_gpu_top` (i915/xe) for real-time EU/engine utilisation on Linux.
- Transferability: yes, available on this box; complements PTI.

### S4. KernelBench - LM-generated GPU kernels
- arXiv: 2502.10517 (Stanford, 2025). Retrieved: yes.
- Tier: T2 (arXiv, open framework).
- Key claims:
  - 250 PyTorch workloads; metric `fast_p` (fraction correct and >p faster than baseline).
  - "leading reasoning models ... still fall short ... matching the PyTorch baseline in **less
    than 20%** of cases"; iterative refinement with execution+profiler feedback boosts
    `fast_1` from 36% to 72% on Level 2.
  - "one-shot LM-generated kernels may not generalize well across hardware" - **hardware-specific
    in-context details help**.
- Transferability to DG2/A770: partial. LM kernel generation is not hardware-portable and our
  target is Vulkan/OpenCL on Xe, not CUDA. Useful conclusion: automated search is viable only with
  a fast *correctness + profiler* feedback loop, which requires S1 first.

## Variance table

| Tool | Scope | Vulkan(ANV)? | Metrics |
|---|---|---|---|
| S1 unitrace | host+device trace, HW metrics, instr-level stalls | via L0/OpenCL (caveat) | EU active/stall/idle, ISA counts |
| S2 Advisor | roofline (L3/HBM/compute), XVE usage, XMX | via DPC++ | roofline dots |
| S3 Mesa/IGC | compiler stats, ISA dump, live EU util | yes (ANV) | spills, SWSB, EU busy |
| S4 KernelBench | LM kernel search | n/a | correctness + speed |

No contradictions; they are complementary. The gap: a first-class **Vulkan/ANV hardware-counter**
path is weaker than the L0/OpenCL path - so our measurement plan should lean on the OpenCL
microbenchmarks (already the project's 33 TF / 86 TF probes) plus PTI, and use ANV compiler stats
for the Vulkan side.

## Practical implications for this frame

1. Direction: build a per-kernel attribution protocol using S1 + S3.
   Mechanism: for each of the ~20 frame GEMM shapes and the epilogues, collect
   `cl_gpu_metrics`/`ze_metric_query` (EU active/stall/idle), `instcount` (ISA mix), and ANV
   `VK_KHR_pipeline_executable_properties` (spills/occupancy); classify each as
   DPAS-issue / load-issue / SLM / L3 / DRAM / latency. Attacks: eliminates guessing about what
   bounds each op (the brief already has an ISA census showing the GEMM is mov/address bound).
   Ceiling: none directly - it multiplies the hit-rate of every other direction. Effort: medium.
   Risk: low. FIRST falsifiable experiment: run `cl_gpu_metrics` on the existing OpenCL DPAS
   microbench and confirm it reports the same bottleneck the ISA census found.
2. Direction: a fast correctness+profiler feedback loop for kernel variant search (S4).
   Mechanism: generate variants across (tile, SLM staging, sub-group size, split-K), gate on an
   exact-output check, score with S1/S2. Attacks: the hand-tuning bottleneck for ~20 shapes.
   Ceiling: whatever the roofline allows; the loop only finds it. Effort: medium-high.
   Risk: medium. FIRST falsifiable experiment: run a 10-variant search on one shape and check it
   finds the same best variant a human did.
3. Direction: roofline the frame with Advisor (S2) to get the DRAM-vs-compute verdict per level.
   Mechanism: `advisor --collect=survey --profile-gpu` + tripcounts. Attacks: prioritisation.
   Risk: Advisor Tripcounts has a known Iris-Xe/Flex failure mode; verify it runs on Arc A770 at
   all before depending on it.

## Excluded leads

- nvprof/nsys/Nsight - NVIDIA only.
- CUPTI - NVIDIA only.
- Pure CPU roofline - irrelevant.
- AlphaEvolve/autonomous kernel search without a verified feedback loop - S4 shows the loop is the
  whole value.