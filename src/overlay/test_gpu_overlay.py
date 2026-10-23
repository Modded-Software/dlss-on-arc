#!/usr/bin/env python3
"""Headless checks for the C GPU overlay.

The overlay needs X and Vulkan to draw, so these tests stay on the paths that do not open
a window: the power/RC6 arithmetic, argument handling, and the telemetry probe. The probe
is driven against a temporary sysfs tree so the JSON contract is tested without a GPU.
"""

import json
import pathlib
import subprocess
import tempfile

BIN = pathlib.Path("work/gpu_overlay")

CARD_KEYS = {
    "card",
    "pci",
    "power_w",
    "power_limit_w",
    "temp_c",
    "fan_rpm",
    "voltage_mv",
    "act_mhz",
    "max_mhz",
    "req_mhz",
    "pcie_gen",
    "pcie_width",
    "rc6_pct",
    "engine_busy_pct",
    "engine_render_pct",
    "engine_copy_pct",
    "engine_video_pct",
    "engine_video_enhance_pct",
    "engine_compute_pct",
    "mem_resident_mb",
    "mem_total_mb",
    "throttle",
    "energy_uj",
}


def run(*args, env=None):
    return subprocess.run([str(BIN), *args], capture_output=True, text=True, env=env, check=False)


def test_selftest():
    result = run("--selftest")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "selftest ok"


def test_help():
    result = run("--help")
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
    assert "--draggable" in result.stdout


def test_probe_empty_tree():
    with tempfile.TemporaryDirectory() as tmp:
        result = run("--probe", "--json", env={"GPU_OVERLAY_SYSFS": tmp})
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout) == []


def test_probe_json_contract():
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        card = root / "card0"
        (card / "device").mkdir(parents=True)
        (card / "gt" / "gt0").mkdir(parents=True)
        (card / "gt_act_freq_mhz").write_text("1200\n")
        (card / "gt_max_freq_mhz").write_text("2400\n")

        result = run(
            "--probe",
            "--json",
            env={"GPU_OVERLAY_SYSFS": str(root)},
        )
        assert result.returncode == 0, result.stderr
        cards = json.loads(result.stdout)
        assert len(cards) == 1, cards
        assert set(cards[0]) == CARD_KEYS, cards[0]
        assert cards[0]["card"] == "card0"
        assert cards[0]["act_mhz"] == 1200
        assert cards[0]["max_mhz"] == 2400
        assert cards[0]["power_w"] is None


def test_probe_client_memory():
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        upstream = root / "pci0000:00" / "0000:00:01.0" / "0000:01:00.0"
        upstream.mkdir(parents=True)
        (upstream / "current_link_speed").write_text("16.0 GT/s PCIe\n")
        (upstream / "current_link_width").write_text("16\n")
        pci = upstream / "0000:02:01.0" / "0000:03:00.0"
        pci.mkdir(parents=True)
        card = root / "card0"
        card.mkdir()
        (card / "device").symlink_to(pci)
        (card / "gt" / "gt0").mkdir(parents=True)

        fdinfo = root / "proc" / "100" / "fdinfo"
        fdinfo.mkdir(parents=True)
        content = (
            "drm-driver:\ti915\n"
            "drm-client-id:\t4242\n"
            "drm-pdev:\t0000:03:00.0\n"
            "drm-total-local0:\t6 MiB\n"
            "drm-resident-local0:\t4 MiB\n"
            "drm-engine-render:\t1000000000 ns\n"
            "drm-engine-copy:\t0 ns\n"
            "drm-engine-video:\t0 ns\n"
            "drm-engine-video-enhance:\t0 ns\n"
            "drm-engine-compute:\t0 ns\n"
        )
        (fdinfo / "7").write_text(content)
        (fdinfo / "8").write_text(content)

        smi = root / "xpu-smi"
        smi.write_text("#!/bin/sh\nprintf '8, 0\\n'\n")
        smi.chmod(0o755)

        result = run(
            "--probe",
            "--json",
            env={
                "GPU_OVERLAY_SYSFS": str(root),
                "GPU_OVERLAY_PROC": str(root / "proc"),
                "GPU_OVERLAY_XPU_SMI": str(smi),
            },
        )
        assert result.returncode == 0, result.stderr
        cards = json.loads(result.stdout)
        assert len(cards) == 1, cards
        assert cards[0]["mem_resident_mb"] == 4.0, cards[0]
        assert cards[0]["mem_total_mb"] == 8.0, cards[0]
        assert cards[0]["engine_busy_pct"] == 0.0, cards[0]
        assert cards[0]["pcie_gen"] == 4, cards[0]
        assert cards[0]["pcie_width"] == 16, cards[0]


def test_live_probe_when_present():
    if not list(pathlib.Path("/sys/class/drm").glob("card[0-9]*")):
        return
    result = run("--probe", "--json")
    assert result.returncode == 0, result.stderr
    cards = json.loads(result.stdout)
    for card in cards:
        assert set(card) == CARD_KEYS, card


def main():
    test_selftest()
    test_help()
    test_probe_empty_tree()
    test_probe_json_contract()
    test_probe_client_memory()
    test_live_probe_when_present()
    print("gpu overlay: selftest, help and telemetry JSON contract pass")


if __name__ == "__main__":
    main()
