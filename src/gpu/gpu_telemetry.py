#!/usr/bin/env python3
"""What this host can and cannot tell us about the Arc A770s, and the power reading.

`frame_profile.py` says where a frame's time goes; this says what the card was doing while
it went there. The sources on this box are uneven, so the module records which ones
actually answer instead of pretending to a uniform interface:

  * `i915` sysfs is the only source that works here. Each card carries
    `/sys/class/drm/cardN/device/hwmon/hwmonM/` (cumulative `energy1_input`, `temp1_input`,
    `in0_input`, `fan1_input`, `power1_max`, `power1_rated_max`) and
    `/sys/class/drm/cardN/gt/gt0/` (`rc6_residency_ms`, `rps_act_freq_mhz`,
    `punit_req_freq_mhz`, `slpc_power_profile`, and the nine `throttle_reason_*` latches).
    `energy1_input` is a microjoule counter, not a gauge: average power over an interval is
    a delta between two reads, which is what `power_w` computes.
  * `xpu-smi` is installed but lists only "Intel Arc B series" as supported - it is the
    Battlemage/datacenter tool, and on an Alchemist A770 it answers "Device handle not
    found". It cannot serve this box.
  * `intel_gpu_top` aborts at start-up here (`get_num_gts: Assertion '!errno || errno ==
    ENOENT' failed`, igpu-tools reading the GT directory) and `gputop` exits without a
    sample. The `i915` perf PMU is not registered and `perf_event_paranoid` is 2, so the
    EU active/stall/idle counters those tools would read are closed to a non-root user
    anyway. Per-engine busyness and the EU array counters are therefore unavailable.
  * `unitrace` / `cl_gpu_metrics` (Intel PTI-GPU; see
    `docs/research/shader-optimisation-2026/02-evidence-library/w7-measurement-autotuning.md`)
    are not installed, and the driver's metric streams need root.

What is readable without root is: energy (hence power), temperature, voltage, fan, the
power limits, the requested and actual clocks, RC6 residency, and the throttle reasons.

    uv run python src/gpu/gpu_telemetry.py probe            # every readable field, both cards
    uv run python src/gpu/gpu_telemetry.py sample 10 0.5    # average power over 10 seconds
"""

import argparse
import json
import pathlib
import shutil
import sys
import time

DRM = pathlib.Path("/sys/class/drm")

# hwmon fields, with the scale from the file's raw integer to the reported unit.
HWMON = {
    "temperature_c": ("temp1_input", 0.001),
    "voltage_mv": ("in0_input", 1.0),
    "fan_rpm": ("fan1_input", 1.0),
    "power_limit_w": ("power1_max", 1e-6),
    "power_rated_w": ("power1_rated_max", 1e-6),
    "power_interval_ms": ("power1_max_interval", 0.001),
}
# gt_* is the card-level clock view; gt/gt0 is the per-GT one (RPS request, RC6).
FREQ = {
    "freq_act_mhz": "gt_act_freq_mhz",
    "freq_cur_mhz": "gt_cur_freq_mhz",
    "freq_max_mhz": "gt_max_freq_mhz",
    "freq_rp0_mhz": "gt_RP0_freq_mhz",
}
GT0 = {
    "rc6_residency_ms": ("rc6_residency_ms", 1.0),
    "rps_act_mhz": ("rps_act_freq_mhz", 1.0),
    "rps_cur_mhz": ("rps_cur_freq_mhz", 1.0),
    "rps_max_mhz": ("rps_max_freq_mhz", 1.0),
    "punit_req_mhz": ("punit_req_freq_mhz", 1.0),
    "slpc_profile": ("slpc_power_profile", None),
}
THROTTLES = ["pl1", "pl2", "pl4", "prochot", "ratl", "status", "thermal", "vr_tdc", "vr_thermalert"]

# The sources that do not answer on this host, with the observation behind each entry, so
# that "why does xpu-smi show nothing?" is answered in one place rather than re-derived.
UNAVAILABLE = [
    (
        "xpu-smi",
        "installed, but supports only the Arc B series (Battlemage); on an Alchemist A770 "
        "it reports 'Device handle not found'",
    ),
    (
        "intel_gpu_top",
        "aborts at start-up: get_num_gts assertion '!errno || errno == ENOENT' failed",
    ),
    ("gputop", "exits without producing a sample"),
    (
        "i915 perf PMU",
        "not registered, and perf_event_paranoid is 2: the EU active/stall/idle counters "
        "are closed without root",
    ),
    ("unitrace / cl_gpu_metrics", "Intel PTI-GPU, not installed; its metric streams need root"),
]


def _read(path):
    """The file's text, or None when the driver does not expose it.

    The existence test covers the normal case: a card exposes only some of these leaves.
    A leaf that exists but cannot be read is a real fault, so it is named on stderr rather
    than reported as an absent sensor.
    """
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError as error:
        sys.stderr.write("gpu telemetry: cannot read %s: %s\n" % (path, error))
        return None


def _number(path, scale):
    """The file's first integer, scaled, or None when it is absent or not a number."""
    raw = _read(path)
    if raw is None:
        return None
    head = raw.split()
    # Not a number: slpc_power_profile reads "[base]    power_saving". A text value has no
    # numeric reading, and inventing one would misreport the card.
    if not head or not head[0].lstrip("+-").isdigit():
        return None
    return float(head[0]) * scale


class Card:
    """One DRM card, with the sysfs roots this host exposes for it."""

    def __init__(self, index):
        self.index = index
        self.name = "card%d" % index
        self.root = DRM / self.name
        self.gt0 = self.root / "gt" / "gt0"
        self.hwmon = _first("device", "hwmon", "hwmon*", under=self.root)

    def pci(self):
        """The PCI BDF of the device behind the card, or None when it cannot be resolved."""
        link = self.root / "device"
        if not link.exists():
            return None
        return link.resolve().name


def _first(*parts, under):
    """The first path matching `under/parts[0]/parts[1]/parts[2]`, where the leaf may glob."""
    base = under.joinpath(*parts[:-1])
    for found in sorted(base.glob(parts[-1])):
        return found
    return None


def cards():
    """Every DRM card that has a device node, lowest index first (`card0` is absent here)."""
    found = []
    for child in sorted(DRM.glob("card[0-9]*")):
        suffix = child.name[4:]
        if suffix.isdigit() and (child / "device").exists():
            found.append(Card(int(suffix)))
    return found


def read(card):
    """A snapshot of every field this host exposes for one card.

    A field the driver does not answer is left out rather than reported as zero: an
    all-zero snapshot would read as an idle GPU rather than as a missing sensor.
    """
    seen = {"card": card.name, "pci": card.pci()}
    if card.hwmon is not None:
        for key, (leaf, scale) in HWMON.items():
            value = _number(card.hwmon / leaf, scale)
            if value is not None:
                seen[key] = value
    for key, leaf in FREQ.items():
        value = _number(card.root / leaf, 1.0)
        if value is not None:
            seen[key] = value
    for key, (leaf, scale) in GT0.items():
        value = _read(card.gt0 / leaf) if scale is None else _number(card.gt0 / leaf, scale)
        if value is not None:
            seen[key] = value
    throttle = {}
    for reason in THROTTLES:
        value = _read(card.gt0 / ("throttle_reason_" + reason))
        if value is not None:
            throttle[reason] = value
    if throttle:
        seen["throttle"] = throttle
    return seen


def energy_uj(card):
    """The card's cumulative energy counter in microjoules, or None when it is absent.

    The value is a counter, so only a difference between two reads means anything.
    """
    if card.hwmon is None:
        return None
    raw = _read(card.hwmon / "energy1_input")
    if raw is None:
        return None
    head = raw.split()
    if not head or not head[0].isdigit():
        return None
    return int(head[0])


def power_w(before, after, seconds):
    """Average watts between two `energy_uj` reads, or None if the counter did not advance."""
    if before is None or after is None or seconds <= 0 or after < before:
        return None
    return (after - before) / 1e6 / seconds


def _mean(values):
    return sum(values) / len(values) if values else None


def frame_power(cards_seen, before, seconds, frames):
    """Per-card watts and joules per frame between two `energy_uj` snapshots.

    `before` maps a card name to the `energy_uj` taken before the frames and `seconds` is
    the wall time they took, so the result is the card's average draw across the timed
    run, not a sample of it. A card whose counter is missing or did not advance is left
    out: a zero would read as an idle card rather than as a sensor that cannot answer.
    """
    report = {}
    for card in cards_seen:
        watt = power_w(before.get(card.name), energy_uj(card), seconds)
        if watt is not None:
            report[card.name] = {"w": watt, "j_per_frame": watt * seconds / frames}
    return report


def _rc6(card):
    return _number(card.gt0 / "rc6_residency_ms", 1.0)


def sample(card, seconds, interval):
    """Power and clock statistics for `card` over `seconds`, sampled every `interval`.

    Power is the mean of the per-interval energy deltas, so a card that idles and then
    bursts reports the burst's share; RC6 is the fraction of the interval the GT spent
    powered down, which separates "busy" from "busy and clocked up".
    """
    first_energy = energy_uj(card)
    if first_energy is None:
        return {"card": card.name, "available": False}
    start = last_t = time.monotonic()
    first_rc6 = _rc6(card)
    last_energy = first_energy
    watts = []
    freqs = []
    temps = []
    while True:
        now = time.monotonic()
        left = seconds - (now - start)
        if left <= 0:
            break
        time.sleep(min(interval, left))
        taken = energy_uj(card)
        at = time.monotonic()
        watt = power_w(last_energy, taken, at - last_t)
        if watt is not None:
            watts.append(watt)
        if taken is not None:
            last_energy, last_t = taken, at
        freq = _number(card.root / "gt_act_freq_mhz", 1.0)
        if freq is not None:
            freqs.append(freq)
        temp = _number(card.hwmon / "temp1_input", 0.001) if card.hwmon is not None else None
        if temp is not None:
            temps.append(temp)
    elapsed = time.monotonic() - start
    last_rc6 = _rc6(card)
    rc6_percent = None
    if first_rc6 is not None and last_rc6 is not None and elapsed > 0:
        rc6_percent = 100.0 * (last_rc6 - first_rc6) / 1000.0 / elapsed
    return {
        "card": card.name,
        "available": True,
        "seconds": elapsed,
        "energy_j": (last_energy - first_energy) / 1e6,
        "power_w_min": min(watts) if watts else None,
        "power_w_avg": _mean(watts),
        "power_w_max": max(watts) if watts else None,
        "freq_mhz_min": min(freqs) if freqs else None,
        "freq_mhz_avg": _mean(freqs),
        "freq_mhz_max": max(freqs) if freqs else None,
        "temperature_c_max": max(temps) if temps else None,
        "rc6_percent": rc6_percent,
    }


def probe():
    """Print every readable field for each card, then the sources that do not answer."""
    found = cards()
    if not found:
        print("no DRM cards with a device node under %s" % DRM)
        return
    for card in found:
        print("%s  pci %s" % (card.name, card.pci() or "?"))
        for key, value in read(card).items():
            if key in ("card", "pci"):
                continue
            if isinstance(value, dict):
                for inner, inner_value in value.items():
                    print("    throttle %-14s %s" % (inner, inner_value))
            else:
                print("    %-20s %s" % (key, value))
        print("    %-20s %s" % ("energy1_input_uj", energy_uj(card)))
    print("\nnot available on this host:")
    for tool, why in UNAVAILABLE:
        where = shutil.which(tool.split()[0])
        print("  %-26s %s (%s)" % (tool, why, "installed" if where else "absent"))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=["probe", "sample"], default="probe", nargs="?")
    parser.add_argument("seconds", type=float, nargs="?", default=5.0)
    parser.add_argument("interval", type=float, nargs="?", default=0.5)
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    args = parser.parse_args()
    if args.mode == "probe":
        if args.json:
            print(json.dumps([read(card) for card in cards()], indent=2))
        else:
            probe()
        return
    results = [sample(card, args.seconds, args.interval) for card in cards()]
    if args.json:
        print(json.dumps(results, indent=2))
        return
    for result in results:
        if not result["available"]:
            print("%s  no energy counter" % result["card"])
            continue
        print(
            "%s  %.1f W (min %.1f, max %.1f) over %.1f s, %.3f J, "
            "clock %.0f-%.0f MHz (avg %.0f), temp<=%.0f C, rc6 %.0f%%"
            % (
                result["card"],
                result["power_w_avg"],
                result["power_w_min"],
                result["power_w_max"],
                result["seconds"],
                result["energy_j"],
                result["freq_mhz_min"],
                result["freq_mhz_max"],
                result["freq_mhz_avg"],
                result["temperature_c_max"],
                result["rc6_percent"],
            )
        )


if __name__ == "__main__":
    main()
