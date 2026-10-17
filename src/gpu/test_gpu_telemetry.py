#!/usr/bin/env python3
"""Lock the telemetry arithmetic: the power figure is a counter difference, so an error
here would misreport every benchmark rather than fail loudly.

The readings themselves come from the driver and cannot be fabricated, so the tests build
a temporary sysfs tree and drive the reader against it: scaling, a missing sensor, a
counter that did not advance, and the joules-per-frame division.
"""

import pathlib
import tempfile

import gpu_telemetry


class FakeCard:
    """A card whose sysfs roots point at a temporary tree instead of /sys/class/drm."""

    name = "card9"

    def __init__(self, root):
        self.root = root
        self.gt0 = root / "gt" / "gt0"
        self.hwmon = root / "hwmon"

    def pci(self):
        return "0000:00:00.0"


def temp_card(root):
    """A FakeCard with an empty hwmon tree, ready for a counter to be written into it."""
    card = FakeCard(root)
    card.hwmon.mkdir(parents=True)
    return card


def test_power_w():
    # 2 J over 2 s is 1 W, and the counter's units are microjoules.
    assert gpu_telemetry.power_w(1_000_000, 3_000_000, 2.0) == 1.0
    assert gpu_telemetry.power_w(3_000_000, 1_000_000, 2.0) is None, "counter went down"
    assert gpu_telemetry.power_w(None, 3_000_000, 2.0) is None
    assert gpu_telemetry.power_w(1_000_000, None, 2.0) is None
    assert gpu_telemetry.power_w(1_000_000, 3_000_000, 0.0) is None


def test_number():
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "temp1_input"
        path.write_text("56000\n")
        assert gpu_telemetry._number(path, 0.001) == 56.0
        assert gpu_telemetry._number(pathlib.Path(tmp) / "absent", 1.0) is None
        path.write_text("[base]    power_saving\n")
        assert gpu_telemetry._number(path, 1.0) is None, "a text field is not a number"


def test_read_leaves_out_absent_fields():
    with tempfile.TemporaryDirectory() as tmp:
        card = temp_card(pathlib.Path(tmp))
        (card.hwmon / "temp1_input").write_text("56000\n")
        seen = gpu_telemetry.read(card)
        assert seen["temperature_c"] == 56.0
        assert "fan_rpm" not in seen, "an absent sensor must be left out, not reported as 0"


def test_frame_power():
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        card = temp_card(root)
        counter = card.hwmon / "energy1_input"
        counter.write_text("1000000\n")
        before = {"card9": gpu_telemetry.energy_uj(card)}
        counter.write_text("3000000\n")
        report = gpu_telemetry.frame_power([card], before, 2.0, 10)
        assert report["card9"]["w"] == 1.0
        assert report["card9"]["j_per_frame"] == 0.2
        # a card with no counter at all is left out of the report rather than read as idle
        assert gpu_telemetry.frame_power([FakeCard(root / "gone")], {}, 1.0, 1) == {}


def main():
    test_power_w()
    test_number()
    test_read_leaves_out_absent_fields()
    test_frame_power()
    live = gpu_telemetry.cards()
    if live:
        print(
            "gpu telemetry: counter arithmetic exact on a temporary tree; %d live card(s), "
            "energy counter %s" % (len(live), gpu_telemetry.energy_uj(live[0]) is not None)
        )
    else:
        print("gpu telemetry: counter arithmetic exact on a temporary tree (no live card)")


if __name__ == "__main__":
    main()
