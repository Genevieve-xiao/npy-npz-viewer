"""
Verify deterministic uxcase test data.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


DATA_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = DATA_DIR / "uxcase_manifest.json"


def fail(message):
    print(f"FAIL: {message}")
    sys.exit(1)


def check(condition, message):
    if not condition:
        fail(message)
    print(f"OK: {message}")


def load_manifest():
    if not MANIFEST_PATH.exists():
        fail("uxcase_manifest.json exists")
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def assert_finite(name, array):
    check(np.isfinite(array).all(), f"{name} has only finite values")


def verify_heartbeat(array):
    check(array.ndim == 1 and array.shape[0] == 2400, "heartbeat shape is (2400,)")
    check(float(array.max()) > 0.8 and float(array.min()) < -0.2, "heartbeat has visible QRS peaks")
    peak_count = int(np.count_nonzero(array > 0.75))
    check(20 <= peak_count <= 80, "heartbeat peak count is plausible")


def verify_well_log(array):
    check(array.shape == (2400, 6), "well log shape is (2400, 6)")
    depth = array[:, 0]
    gamma = array[:, 1]
    porosity = array[:, 3]
    impedance = array[:, 5]
    check(bool(np.all(np.diff(depth) > 0)), "well log depth is strictly increasing")
    check(float(gamma.max() - gamma.min()) > 60.0, "gamma curve has layered contrast")
    check(float(porosity.max() - porosity.min()) > 0.12, "porosity has reservoir intervals")
    check(float(impedance.max() - impedance.min()) > 500.0, "impedance has interpretable range")
    corr = float(np.corrcoef(gamma, porosity)[0, 1])
    check(corr < -0.35, "gamma and porosity are anti-correlated")


def verify_fault_image(array):
    check(array.shape == (512, 512), "fault slice shape is (512, 512)")
    check(float(array.max() - array.min()) > 1.5, "fault slice has strong amplitude contrast")
    left = array[:, :256].mean(axis=1)
    right = array[:, 256:].mean(axis=1)
    check(float(np.std(left - right)) > 0.03, "fault slice has lateral discontinuity")


def verify_rgb(array):
    check(array.shape == (256, 256, 3), "landcover RGB shape is (256, 256, 3)")
    check(float(array.min()) >= 0.0 and float(array.max()) <= 1.0, "RGB values are normalized")
    channel_means = array.reshape(-1, 3).mean(axis=0)
    check(float(channel_means.max() - channel_means.min()) > 0.05, "RGB channels have distinct means")


def verify_volume(array, expected_shape, name):
    check(array.shape == expected_shape, f"{name} shape is {expected_shape}")
    check(float(array.max() - array.min()) > 1.5, f"{name} has seismic amplitude contrast")
    mid_slice = array[:, :, array.shape[2] // 2]
    check(float(mid_slice.std()) > 0.25, f"{name} middle slice has visible structure")


def verify_reservoir(array):
    check(array.shape == (48, 64, 40, 4), "reservoir 4D shape is (48, 64, 40, 4)")
    check(float(array[..., 1].min()) >= 0.0 and float(array[..., 1].max()) <= 1.0, "semblance channel is normalized")
    check(float(array[..., 2].min()) >= 0.0 and float(array[..., 2].max()) <= 0.4, "porosity channel is physical")
    check(float(array[..., 3].min()) >= 0.0 and float(array[..., 3].max()) <= 1.0, "facies channel is probability-like")
    check(float(array[..., 3].max() - array[..., 3].min()) > 0.7, "facies channel has high contrast")


def verify_npz(path):
    expected = {
        "heartbeat": (600,),
        "well_log": (600, 6),
        "fault_image": (128, 128),
        "landcover_rgb": (128, 128, 3),
        "seismic_volume": (32, 32, 40),
        "reservoir_4d": (24, 32, 20, 4),
    }
    with np.load(path) as data:
        check(set(data.keys()) == set(expected), "NPZ contains expected keys")
        for key, shape in expected.items():
            array = data[key]
            check(array.shape == shape, f"NPZ key {key} has shape {shape}")
            assert_finite(f"NPZ key {key}", array)


def main():
    manifest = load_manifest()
    entries = {entry["file"]: entry for entry in manifest["files"]}

    for name, entry in entries.items():
        path = DATA_DIR / name
        check(path.exists(), f"{name} exists")

        if entry["kind"] == "npz":
            verify_npz(path)
            continue

        array = np.load(path, mmap_mode="r")
        check(list(array.shape) == entry["shape"], f"{name} matches manifest shape")
        check(str(array.dtype) == entry["dtype"], f"{name} matches manifest dtype")
        assert_finite(name, array)

        if name == "uxcase_signal_heartbeat_1d.npy":
            verify_heartbeat(array)
        elif name == "uxcase_well_log_table_2400x6.npy":
            verify_well_log(array)
        elif name == "uxcase_fault_slice_image_512x512.npy":
            verify_fault_image(array)
        elif name == "uxcase_landcover_rgb_256x256x3.npy":
            verify_rgb(array)
        elif name == "uxcase_seismic_volume_96x128x80.npy":
            verify_volume(array, (96, 128, 80), name)
        elif name == "uxcase_reservoir_4d_48x64x40x4.npy":
            verify_reservoir(array)
        elif name == "uxcase_stress_seismic_cube_192x256x160.npy":
            verify_volume(array, (192, 256, 160), name)

    print("\nAll uxcase data checks passed.")


if __name__ == "__main__":
    main()
