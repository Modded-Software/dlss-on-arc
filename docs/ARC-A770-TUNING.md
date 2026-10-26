# Arc A770 Linux tuning and overclocking capability report

Date: 2026-10-23
Target: GPU1 only

## Summary

- GPU1 is an Arc A770 on the `i915` driver.
- Current Linux stack exposes frequency clamps and a PL1 power cap.
- It does not expose voltage offset, a voltage-frequency curve, memory overclocking,
  fan PWM control, or a temperature limit.
- The Level Zero Sysman overclocking API reports `UNSUPPORTED_FEATURE` on GPU1.
- Intel states it has no plans to ship Arc Control or these tuning APIs on Linux.
- Upstream fan control (series 168027) is `xe` only and does not work on DG2.
- All write tests were blocked because the session is unprivileged.
- No GPU settings were changed.

## Test system

- OS: Ubuntu 24.04.4 LTS
- Kernel: 7.1.8
- GPU1: `0000:09:00.0`, `card2` / `renderD129`, DG2 Arc A770, 16 GiB
- Driver: `i915`. `xe` is loaded but not bound to GPU1.
- Mesa: 25.2.8
- Vulkan driver: 25.2.8
- OpenCL driver: 26.27.39122.12
- Level Zero: 1.28.6
- XPU Manager: 2.0.0

## Local capability matrix

| Capability | Exposed on GPU1 | Can read | Can write | Notes |
|---|---:|---:|---:|---|
| GPU min/max frequency | Yes | Yes | Root only | 300 to 2400 MHz |
| GPU frequency lock | Yes | Yes | Root only | Set min = max |
| GPU requested/actual frequency | Yes | Yes | No | sysfs and XPU-SMI |
| PL1 sustained power limit | Yes | Yes | Root only | 190 W |
| PL1 averaging interval | Yes | Yes | Root only | 28 s |
| PL2 / burst power limit | No | No | No | Not exposed by this i915 hwmon |
| Reactive critical power / current | No | No | No | No `power1_crit` or `curr1_crit` |
| Voltage offset | No | No | No | Level Zero OC API unsupported |
| Voltage-frequency curve | No | No | No | No i915 interface |
| Frequency offset above 2400 MHz | No | No | No | `gt_max_freq_mhz` is capped at RP0 |
| VRAM frequency or voltage offset | No | No | No | No interface |
| Temperature limit | No | No | No | Only `temp1_input` |
| Fan RPM | Yes | Yes | No | `fan1_input`, read-only |
| Fan PWM / fan curve | No | No | No | Sysman fan `canControl=0` |
| Scheduler mode | Yes | Yes | Root only | `xpu-smi config` |
| Standby mode | Yes | Yes | Root only | `xpu-smi config` |

## Current GPU1 values

```text
gt_min_freq_mhz      300
gt_max_freq_mhz      2400
gt_boost_freq_mhz    2400
gt_RP0_freq_mhz      2400
gt_RP1_freq_mhz      600
gt_RPn_freq_mhz      300
gt_cur_freq_mhz      variable
gt_act_freq_mhz      variable

power1_max           190000000  (190 W, microwatts)
power1_max_interval  28000      (28 s)
power1_rated_max     0
temp1_input          51000      (51 C)
fan1_input           0          (RPM)
in0_input            0          (mV)
```

`xpu-smi config --device 1` reported:

- Package sustained limit: 190 W
- Card power limit: N/A
- GPU min frequency: 300 MHz
- GPU max frequency: 2400 MHz
- Standby mode: default
- Scheduler mode: timeslice, interval 1000 us, yield timeout 640000 us
- Fan 0: N/A / N/A
- Memory ECC: disabled
- PCIe Gen4 downgrade: N/A

## Tests performed

### 1. Direct sysfs frequency write

```sh
printf '300\n' > /sys/class/drm/card2/gt_min_freq_mhz
```

Result: `Permission denied`.

The file is `-rw-r--r-- root:root`.

### 2. Direct sysfs power-limit write

```sh
printf '190000000\n' > /sys/class/drm/card2/device/hwmon/hwmon8/power1_max
```

Result: `Permission denied`.

The file is `-rw-rw-r-- root:root`.

### 3. XPU-SMI frequency write

```sh
xpu-smi config --device 1 --tile 0 --frequencyrange 300,2400
```

Result:

```text
[Error] Failed to set frequency range for device-level GPU domain.
0x70010000 (ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS)
```

### 4. XPU-SMI power-limit write

```sh
xpu-smi config --device 1 --powerlimit 190
```

Result:

```text
[Error] Failed to set extended power limits.
0x70010000 (ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS)
```

### 5. Level Zero Sysman overclock query

A read-only Level Zero query was compiled against the current headers and run
against device index 1 (`0000:09:00.0`).

Result:

```text
DEVICE 1: Intel(R) Arc(TM) A770 Graphics
  fans: SUCCESS n=1
    FAN 0: canControl=0 maxRPM=-1 maxPoints=0
  power domains: SUCCESS n=1
    POWER 0: canControl=1 default=0 mW min=-1 max=0
  freq domains: SUCCESS n=1
  FREQ 0 (GPU): canControl=1 min=300 max=2400
    state: request=2400 actual=2400 tdp=-1 efficient=600 voltage=-1.000
           throttle=0x0
    OC capabilities: UNSUPPORTED_FEATURE
           supported=0 maxOcFreq=0 factoryFreq=0 factoryVolt=0.000
           maxOcVolt=0.000 voltOffset=[0.000, 0.000]
           tjMax=0 iccMax=0 highVolt=0/0 extended=0 fixed=0
```

This is the key result. The driver can clamp frequency and power, but it does
not expose the overclocking controls that the Windows driver exposes.

### 6. `intel_gpu_frequency`

`intel_gpu_frequency` has no device selector. It opened `/dev/dri/card1`
(GPU0) and failed because debugfs is not accessible:

```text
Failed assertion: debugfs_root
Last errno: 1, Operation not permitted
```

It aborted before any read or write. It was not used further.

### 7. `intel_gpu_top` against GPU1

```sh
intel_gpu_top -d drm:/dev/dri/card2 -J -s 1000
```

Result: assertion failure in `get_num_gts` and a core dump. Not usable for
GPU1 telemetry on this build.

### 8. XPU-SMI telemetry

XPU-SMI reads GPU1 correctly:

```text
Device ID: 1
PCI BDF: 0000:09:00.0
GPU Power: about 41 W idle in one sample
GPU Frequency: variable
Power Limit: 190 W
```

Some telemetry is incomplete. Temperature is N/A in XPU-SMI even though
`temp1_input` reports 51 C. Memory-used values were inconsistent between
samples. For reliable values, read hwmon and sysfs directly.

## What can be tuned today

All writes require root. Use explicit device paths and change one thing at a
time.

### Frequency clamp

```sh
# GPU1: clamp to 1800 MHz max
echo 1800 | sudo tee /sys/class/drm/card2/gt_max_freq_mhz

# restore
echo 2400 | sudo tee /sys/class/drm/card2/gt_max_freq_mhz
```

To lock a fixed frequency, set min and max to the same value:

```sh
echo 1800 | sudo tee /sys/class/drm/card2/gt_min_freq_mhz
echo 1800 | sudo tee /sys/class/drm/card2/gt_max_freq_mhz
```

Allowed range: 300 to 2400 MHz. Values above 2400 MHz are rejected because
`gt_max_freq_mhz` is bounded by RP0.

XPU-SMI equivalent:

```sh
sudo xpu-smi config --device 1 --tile 0 --frequencyrange 1800,1800
sudo xpu-smi config --device 1 --tile 0 --frequencyrange 300,2400
```

### Power limit

```sh
# GPU1: set PL1 to 150 W
echo 150000000 | sudo tee /sys/class/drm/card2/device/hwmon/hwmon8/power1_max

# restore
echo 190000000 | sudo tee /sys/class/drm/card2/device/hwmon/hwmon8/power1_max
```

XPU-SMI equivalent:

```sh
sudo xpu-smi config --device 1 --powerlimit 150
sudo xpu-smi config --device 1 --powerlimit 190
```

Only PL1 is exposed. There is no PL2, no PL4, and no IccMax control on this
i915 interface. The max settable value is the board's default 190 W here.

### Power-limit averaging interval

```sh
echo 28000 | sudo tee /sys/class/drm/card2/device/hwmon/hwmon8/power1_max_interval
```

Keep the default unless testing transient behavior.

### Scheduler and standby

These are not overclocking controls, but XPU-SMI exposes them:

```sh
sudo xpu-smi config --device 1 --scheduler "timeslice,1000,640000"
sudo xpu-smi config --device 1 --standby never
sudo xpu-smi config --device 1 --standby default
```

## What cannot be tuned today

### Voltage and VF curve

No voltage offset. No voltage-frequency curve. No undervolt. No overvolt.
The Level Zero OC capability query returns `UNSUPPORTED_FEATURE`. The current
kernel driver does not expose the PCODE transaction needed for VF changes.

### Memory overclocking

No VRAM frequency or voltage controls. GDDR6 stays at stock.

### Fan control

Fan RPM is readable. Fan PWM and fan curves are not exposed. Sysman reports
one fan with `canControl=0`. An A770 owner in compute-runtime #885 reports the
same: `zesDeviceEnumFans()` succeeds but returns `fanCount = 0`. The upstream
fan patch is `xe` only, is still under review, and failed on an A380 test.

### Temperature limit

Only `temp1_input` is present. There is no `temp1_max`, `temp1_crit`, or
writable thermal limit.

### Frequency above 2400 MHz

Not possible. The i915 sysfs store rejects values above the hardware RP0
maximum. There is no frequency-offset interface on Linux.

## Windows comparison

Windows Intel Graphics Software exposes the full tuning set:

- Frequency offset, including positive offset.
- Voltage limit or voltage offset.
- Voltage-frequency curve in advanced mode.
- Power limit as a percentage.
- Temperature limit.
- Fan target speed and fan curve.
- Memory speed adjustment on supported cards.

Intel documents the basic Arc tuning controls as GPU Performance Boost,
GPU Voltage Offset, GPU Power Limit, and GPU Temperature Limit. Third-party
IGCL tools can also use frequency/voltage lock modes.

For A770, published overclocking results reached about 2600 to 2800 MHz with
raised voltage and power. Those results are Windows and IGCL based. They do
not apply to the current Linux i915 stack.

## Why Linux differs

- Windows uses Intel IGCL and the Windows kernel driver. IGCL exposes the VF
  curve and fan controls.
- Linux i915 exposes only the generic DRM/sysfs and hwmon controls.
- Level Zero Sysman has OC APIs, but the current Intel Linux implementation
  marks them unsupported for this device.
- The `xe` driver has its own frequency API and newer hwmon power limits, but
  stock `xe` still lacks voltage and fan control. Fan PWM support is still
  under review upstream as of September 2026.
- Intel support stated on 2026-01-14 that Arc Control will not come to Linux
  and gave no commitment for undervolting, fan, or overclocking interfaces.
- LACT maintainer (2026-07-14): neither `i915` nor `xe` provides fan control or
  overclocking on any hardware. The upcoming `xe` fan patch was tested on an
  A380 and did not work. Kernel-side overclocking is not planned for DG2.

## Community projects

These were checked against their own documentation.

| Project | What it does | Works on A770? |
|---|---|---:|
| `exzile/intel-arc-pro-fan-control` | Fan curves, VF curve, VRAM speed, temp limit, power/clock tuning via patched `xe` | No. `xe` only; tested on B60/B70/B580 |
| `PerkyZZ999/XeDriver_FanPatch` | Kernel patch for fan PWM/curves | No. `xe` only |
| `Trollicus-Rex/arc-monitor` | D-Bus daemon and CLI for monitoring, claims power-limit and fan PWM | Unverified. Fan path needs Sysman fan handles, which DG2 does not populate |
| `ilya-zlobintsev/LACT` | GPU control GUI | No Intel fan/OC yet; waiting on kernel support |
| `YamsSE/Arc-Power` | Full tuning, fan curves, VF, VRAM | No. Windows only |
| `xpu-smi` | Power limit, frequency clamp, scheduler | Read/write attempted; needs root |

Important: the patched-driver tools issue PCODE mailbox transactions that stock
`i915` does not perform. They do not apply to the A770 as long as it stays on
`i915`. A DG2 port would be new, out-of-tree work, not a configuration change.

## Practical recommendations

1. For maximum stable performance, leave frequency on auto and keep the 190 W
   PL1. The card already boosts to 2400 MHz when conditions allow.
2. For repeatable benchmarks, clamp min and max to 2400 MHz with root. Watch
   temperatures and power.
3. For lower idle power or heat, lower `gt_max_freq_mhz` first. Lowering PL1
   also works, but it affects sustained load more broadly.
4. Do not expect real overclocking on i915. There is no voltage, memory, fan,
   or thermal-limit control.
5. If overclocking is required, use Windows with Intel Graphics Software, or
   accept a patched/out-of-tree Linux driver with the associated risk.
6. Persist any root setting with a systemd unit or udev rule. Settings are not
   preserved across reboot by the driver.

## Safety notes

- Overclocking voids warranties and can damage the GPU.
- Do not raise PL1 without checking board VRM and cooling limits.
- Do not pin maximum frequency for long periods without thermal monitoring.
- The A770 has a 190 W software PL1 here. Published Windows overclocks used
  228 W or more and aftermarket cooling.
- This report did not apply any settings. Every write attempt was rejected by
  permissions.

## Sources

- Intel, "How Can I Overclock My Intel Arc GPU Using Intel Arc Control?"
  https://www.intel.com/content/www/us/en/support/articles/000092590/graphics.html
- Intel Graphics Software product page
  https://www.intel.com/content/www/us/en/products/details/discrete-gpus/arc/software/graphics-software.html
- Intel XPU Manager, `xpu-smi config`
  https://intel.github.io/xpumanager/2.0/xpu-smi/config.html
- Level Zero Sysman programming guide, frequency/voltage overclocking
  https://oneapi-src.github.io/level-zero-spec/level-zero/latest/sysman/PROG.html
- Linux kernel ABI, i915 hwmon (`power1_max`, `power1_max_interval`,
  `fan1_input`)
  https://www.kernel.org/doc/html/latest/gpu/i915.html
- Linux kernel Xe GT frequency management
  https://docs.kernel.org/gpu/xe/xe_gt_freq.html
- Linux kernel Xe hwmon ABI
  https://github.com/torvalds/linux/blob/master/Documentation/ABI/testing/sysfs-driver-intel-xe-hwmon
- Intel compute-runtime issue 885, "[GSD-12198] No Fan control in Linux for
  Battlemage GPUs" (A770/DG2 confirmed affected)
  https://github.com/intel/compute-runtime/issues/885
- Intel Community, "Will the i915 or Xe driver on Linux support undervolting &
  overclocking?" (2026-01-14 answer)
  https://community.intel.com/t5/Intel-Arc-Discrete-Graphics/Will-the-i915-or-Xe-driver-on-Linux-support-undervolting-amp/m-p/1733656
- LACT issue 1107, "Fan control for Intel ARC"
  https://github.com/ilya-zlobintsev/LACT/issues/1107
- LACT issue 1111, "Will LACT work properly with Intel Arc A series GPUs"
  https://github.com/ilya-zlobintsev/LACT/issues/1111
- `exzile/intel-arc-pro-fan-control` (Battlemage only)
  https://github.com/exzile/intel-arc-pro-fan-control
- `PerkyZZ999/XeDriver_FanPatch` (Battlemage fan patch)
  https://github.com/PerkyZZ999/XeDriver_FanPatch
- `Trollicus-Rex/arc-monitor`
  https://github.com/Trollicus-Rex/arc-monitor
- Linux kernel patch series 168027, `drm/xe/hwmon` fan control
  https://patchwork.freedesktop.org/series/168027/
- SkatterBencher, Arc A770 overclocking to 2795 MHz
  https://skatterbencher.com/2023/07/29/skatterbencher-64-intel-arc-a770-overclocked-to-2795-mhz/
- TechPowerUp, Arc A770 overclocking
  https://www.techpowerup.com/review/intel-arc-a770/40.html
- Phoronix, i915 versus Xe on Arc A-Series
  https://www.phoronix.com/review/intel-i915-xe-linux-2025
