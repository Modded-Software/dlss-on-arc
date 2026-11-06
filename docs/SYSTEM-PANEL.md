# System panel: requirements and specification

Date: 2026-11-06
Status: implemented as `gpu_overlay --system`

## Purpose

Add a second overlay panel that reports the CPU package(s) and system memory.
It reuses the GPU overlay's visual language: same window, font, palette,
spacing, tabs, compact mode, hint, and footer.

## Requirements

### R1. Separate panel

- New mode `--system` on the existing overlay binary.
- Do not mix CPU or memory rows into the GPU panel.
- GPU mode behavior is unchanged.

### R2. No privileges

- Read only procfs, sysfs, and hwmon.
- Never write.
- Missing files render as `--`.

### R3. Same interaction

- `GRAPH` and `DETAILS` tabs.
- Click metrics to select the graph series.
- Range selector: 1m, 15min, 1h.
- Double-click toggles compact mode.
- Hover shows `Double-click for details.` in compact mode.
- Pointer cursor over clickable items.

### R4. CPU information

Top metrics:

| Metric | Unit | Source |
|---|---|---|
| LOAD | % | `/proc/stat` delta |
| FREQ | GHz | `cpufreq/scaling_cur_freq` average |
| TMP | C | `coretemp` package sensor |
| PWR | W | RAPL energy when readable, else `--` |

RAPL energy is root-only on many systems. The details tab then shows the
readable PL1 and PL2 limits instead.

Details:

- Model, vendor.
- Family, model, stepping, microcode.
- Cores, threads, sockets.
- Min, current, max frequency; base frequency.
- Governor, driver, energy preference.
- L1d, L1i, L2, L3 cache sizes.
- NUMA node count.
- Package temperature.
- Load average (1, 5, 15 min).

### R5. Memory information

Top metrics:

| Metric | Unit | Source |
|---|---|---|
| USED | % | `/proc/meminfo` |
| USED | GiB | `/proc/meminfo` |
| AVL | GiB | `MemAvailable` |
| SWAP | % | `SwapTotal` and `SwapFree` |

Details:

- Total, used, free, available.
- Buffers, cached, shared.
- Active, inactive, dirty, writeback.
- Slab, reclaimable, unreclaimable.
- Kernel stack, page tables.
- Swap total, used, free.
- DIMM temperatures from `spd5118` hwmon.

### R6. Layout

- Default position: left quarter of the output, so the GPU panel can stay on the right.
- Per-block title: large `CPU0` or `MEM0` plus model text.
- Metrics row: four slots, labels above values, centered as a group.
- Separator, then tabs.
- Graph tab: selected metric, range selector, scale label.
- Details tab: left-aligned two-column rows with fixed offsets.
- Footer: `PWR <W|--> | MEM <used>G/<total>G` and `SWAP <used>G/<total>G`.
- Compact mode: title, four metrics, hidden hover hint, footer.
- Compact and expanded keep the same width.

### R7. History

- Four series per block, one sample per refresh interval.
- Ring buffer of 3600 samples.
- Range selector limits samples to 60, 900, or 3600.

## Sources

- `/proc/stat`, `/proc/loadavg`, `/proc/cpuinfo`, `/proc/meminfo`
- `/sys/devices/system/cpu/cpu*/cpufreq/*`
- `/sys/devices/system/cpu/cpu0/cache/index*/*`
- `/sys/class/hwmon/*/name`, `coretemp`, `spd5118`
- `/sys/class/powercap/intel-rapl:0/*`

## Verification

- Build with `make -C src/overlay build`.
- Run headless tests with `make test-gpu-overlay`.
- Run `work/gpu_overlay --system --probe`.
- Screenshot expanded graph, details, and compact modes.
