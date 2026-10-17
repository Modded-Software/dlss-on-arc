# 04 - Gap analysis and Wave-2 gap-fill

Author: orchestrator, 2026-10-17. Sits between the evidence library and the synthesis
(`03-synthesis-guide.md`) and records what the review did **not** capture, why, and what was done
about it. Intended as an honesty artefact: the reader should be able to see the seams.

## 1. What happened to the intended method

The protocol (§5) planned seven parallel research subagents (Wave 1, W1-W7) plus an orchestrator
cross-check. Reality:

| Wave-1 session | Launched | Outcome |
|---|---|---|
| W1 | yes | stalled on model endpoint; no output |
| W2 | yes | stalled; no output |
| W3 | yes | stalled; no output |
| W4 | yes | stalled; no output |
| W5 | yes | stalled; no output |
| W6 | yes | stalled; no output |
| W7 | yes | stalled; no output |
| W1b (relaunch) | yes | stalled; no output |

Symptom: process alive in `epoll_wait`, CPU time flat over 20 s, log files frozen for 9-10 minutes;
the orchestrator's own web tools kept working (with intermittent Exa HTTP 429). All sessions were
killed. **No subagent produced a single source.** Every evidence file was assembled by the
orchestrator from directly retrieved sources. This is a deviation from the protocol and is declared.

Consequence: the review is **single-reviewer** (already declared in protocol §10) and the source
budget is smaller than planned (44 vs 60-91). Wave 2 below recovers the explicitly-named leads.

## 2. Coverage matrix - what each workstream does and does not establish

| WS | Establishes | Does **not** establish |
|---|---|---|
| W1 | DG2 coopmat limits from Mesa source + live `vulkaninfo` (rev 2, compute-only, subgroup 32, tile set, no FP8 / no f16-accum) | IGC vISA-level DPAS lowering quality; GRF-mode availability on DG2 (no 256-GRF) |
| W2 | Stream-K / split-K / packing / epilogue-fusion mechanisms and their ceilings | Any measurement on Intel Xe; CUTLASS EVT not portable (no CUTLASS on Xe) |
| W3 | megakernel / dataflow / graph-compiler mechanisms (MPK, Welder, Souffle, DeFiNES) | None run on DG2; all NVIDIA/A100-evidenced |
| W4 | FA2/FA3/NATTEN mechanisms; the llama.cpp A770 evidence (coopmat ~no gain) | A DG2-specific FA kernel; Intel's two-phase split not implemented by us |
| W5 | FP8/4-bit weight-dequant mechanisms (Ozaki, MARLIN, LUT-GEMM) | 4-bit dequant on DG2's int8 DPAS; our own accuracy budget for 4-bit |
| W6 | temporal-reuse and 2-GPU split mechanisms | PCIe byte-cost on this box; 2-GPU quality at the halo |
| W7 | Intel profiler/roofline tooling; LM-kernel generation state | Any actual profile of our frame (protocol says no new measurement) |

## 3. Wave-2 gap-fill (named leads)

Leads named in Wave 1 but not retrieved at the time were re-searched and resolved:

| Lead | Resolved to | Verdict |
|---|---|---|
| DeltaCNN | arXiv 2203.03996, CVPR 2022 | **excluded as implementation** - CUDA-only and explicitly tensor-core-free; the 7x is vs non-XMX cuDNN. Mechanism noted. |
| Skip-Convolutions | arXiv 2104.11487, CVPR 2021 | **partial** - training-free Norm gate adds 3-4x with no accuracy drop and keeps 8x8 block sparsity (XMX-compatible), but only for the CNN stages, not attention. |
| MCFuser | arXiv 2506.22169, SC 2024 | **method-transfer** - analytic cost model over a tiling search space for fusing MBCI chains; feeds the D4 fusion + W7 autotuning path. |
| Gemini "compute-on-the-fly" | (3DGS rendering line, e.g. CTRL-GS 2505.18306) | **out of scope** - not a training-free CNN/ViT redundancy method. Excluded after inspection, not refuted. |

Wave 2 also surfaced two adjacent video-SR redundancy methods (MDPI Sensors 23(18):7880; ACM MM
2023 kernel-split) as directional corroboration of the feature-cache mechanism (S8 in W6).

**Wave-2 verdict:** the outstanding leads are concept-adjacent, not adoptable as-is. They do **not**
change the D1-D9 ranking in `03-synthesis-guide.md`.

## 4. Source-count and shrinkage

| Stage | Sources |
|---|---|
| Wave 1 (orchestrator recovery) | 40 |
| Wave 2 gap-fill | +4 (S7 W2; S6,S7,S8 W6) |
| **Total** | **44** |

Against a 60-91 plan this is a ~30-50 % shortfall. The shortfall is *depth per workstream*, not the
existence of the key mechanisms: every workstream has at least one T1 anchor (Mesa source / CVPR /
ICLR / OSDI / SC) and the top-ranked directions (D1-D4) rest on T1 or >=2 independent T2.

## 5. Verification performed

- **arXiv ids:** all cited ids re-fetched via `http://export.arxiv.org/api/query?id_list=...`;
  **18/18** (Wave 1) resolved to the expected titles; Wave-2 ids 2203.03996, 2104.11487, 2506.22169
  resolved the same way.
- **Non-arXiv URLs:** 33 URLs status-checked with `curl -L`. 29 x HTTP 200. Three non-200 are
  **bot-protection artefacts, not dead links**: `doi.org/10.1145/3617232.3624858`,
  `doi.org/10.1177/1094342020965661` and `intel.com/.../2024-1/advisor-roofline.html` return 403/404
  to `curl` but their content was independently retrieved (arXiv/IJHPCA mirrors; the Intel page body
  was returned by the search engine and is quoted verbatim in W7 S2). No citation was left with an
  unverifiable target.
- **On-box facts** (strongest tier, T1): Mesa 25.2.8 source at `notes/mesa-25.2.8-src/` and live
  `vulkaninfo` on the A770.

## 6. Residual gaps and confidence

- **Highest-confidence claims (act on these):** DG2 coopmat tile set / no-FP8 / compute-only (source
  + live readback); Stream-K removes tail quantisation; epilogue fusion removes one activation
  round-trip; LUT dequant of 1-byte weights is bandwidth-exact-logic. These underpin D1-D4.
- **Medium-confidence (prototype before committing):** the exact ceiling of the fused per-window
  kernel on DG2 (no Xe measurement exists in the literature); DGC benefit (merged in Mesa 26.2, we
  run 25.2.8; spec says DGC is not async).
- **Low-confidence / must-measure:** 4-bit dequant on int8 DPAS accuracy; 2-GPU halo PCIe cost;
  whether FA on DG2 can beat a plain attention at head-dim 16-32.
- **Known bias:** easily-retrieved, English, arXiv-heavy; vendor blog claims flagged T3; no access to
  vendor-internal kernels (so "what oneDNN/XeTLA actually do" is inferred from docs + source, not
  from a vendor engineer).
- **Out of scope by design:** retraining-time methods (context only), any new GPU measurement.

## 7. What a Wave 3 would do (if resumed)

1. Retrieve the ~4 vendor/system T1 sources per workstream that Wave 1 missed (e.g. XeTLA repo
   internals, oneDNN Xe kernels, ANV DGC MR thread).
2. Measure the two "must-measure" items in §6 with the W7 tooling (PTI unitrace on the Vulkan/ANV
   path caveat).
3. Run the D1 fused-kernel microbenchmark (the single highest-value falsifiable experiment).