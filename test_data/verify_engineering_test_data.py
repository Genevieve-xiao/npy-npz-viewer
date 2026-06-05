"""
Verify deterministic engcase engineering test data.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


DATA_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = DATA_DIR / "engcase_manifest.json"


EXPECTED_SHAPES = {
    "engcase_bearing_vibration_4096.npy": (4096,),
    "engcase_bridge_sensor_table_2400x6.npy": (2400, 6),
    "engcase_fem_stress_plate_512x512.npy": (512, 512),
    "engcase_industrial_ct_volume_96x128x96.npy": (96, 128, 96),
    "engcase_cfd_wake_24x64x48x4.npy": (24, 64, 48, 4),
}


def fail(message):
    print(f"FAIL: {message}")
    sys.exit(1)


def check(condition, message):
    if not condition:
        fail(message)
    print(f"OK: {message}")


def load_manifest():
    if not MANIFEST_PATH.exists():
        fail("engcase_manifest.json exists; run create_engineering_test_data.py first")
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def assert_finite(name, array):
    check(np.isfinite(array).all(), f"{name} has only finite values")


def verify_bearing(array):
    check(array.shape == (4096,), "bearing vibration shape is (4096,)")
    centered = array - float(array.mean())
    crest_factor = float(np.max(np.abs(centered)) / np.std(centered))
    high_tail = int(np.count_nonzero(array > float(array.mean() + 2.0 * array.std())))
    check(float(array.max() - array.min()) > 1.0, "bearing vibration has strong impact contrast")
    check(crest_factor > 4.0, "bearing vibration has impulsive crest factor")
    check(25 <= high_tail <= 260, "bearing vibration has plausible repeated impact samples")


def verify_bridge(array):
    check(array.shape == (2400, 6), "bridge sensor table shape is (2400, 6)")
    minutes, load, strain, temp, displacement, stress = array.T
    check(bool(np.all(np.diff(minutes) > 0)), "bridge time column is strictly increasing")
    check(float(temp.max() - temp.min()) > 12.0, "bridge temperature has daily cycle")
    check(float(load.max() - load.min()) > 110.0, "bridge load has truck crossing events")
    check(float(np.corrcoef(load, strain)[0, 1]) > 0.88, "bridge load and strain are strongly correlated")
    check(float(np.corrcoef(load, displacement)[0, 1]) > 0.88, "bridge load and displacement are strongly correlated")
    check(float(np.corrcoef(strain, stress)[0, 1]) > 0.98, "bridge strain and stress are nearly linear")


def verify_fem(array):
    check(array.shape == (512, 512), "FEM stress plate shape is (512, 512)")
    size = array.shape[0]
    y, x = np.mgrid[-1.0:1.0:complex(size), -1.0:1.0:complex(size)]
    radius = np.sqrt((x + 0.22) ** 2 + (y - 0.02) ** 2)
    hole = radius < 0.16
    ring = (radius > 0.18) & (radius < 0.27)
    crack_tip = (x > -0.03) & (x < 0.12) & (y > -0.04) & (y < 0.08)
    check(float(array.max() - array.min()) > 0.95, "FEM stress plate has normalized contrast")
    check(float(array[hole].mean()) < 0.04, "FEM stress plate has low-valued hole")
    check(float(array[ring].mean()) > float(array.mean() + 0.12), "FEM stress ring is brighter than average")
    check(float(array[crack_tip].max()) > 0.85, "FEM stress plate has crack-tip hotspot")


def verify_ct_volume(array):
    check(array.shape == (96, 128, 96), "industrial CT volume shape is (96, 128, 96)")
    nz, ny, nx = array.shape
    z, y, x = np.mgrid[
        -1.0:1.0:complex(nz),
        -1.0:1.0:complex(ny),
        -1.0:1.0:complex(nx),
    ]
    body_mask = (x / 0.74) ** 2 + (y / 0.62) ** 2 + (z / 0.82) ** 2 < 0.90
    background_mask = (x / 0.86) ** 2 + (y / 0.76) ** 2 + (z / 0.94) ** 2 > 1.30
    body = array[body_mask]
    background = array[background_mask]
    check(float(body.mean()) > float(background.mean() + 0.45), "industrial CT body is denser than background")
    check(float(body.min()) < 0.18, "industrial CT contains low-density pores or crack")
    check(float(body.max()) > 0.88, "industrial CT contains bright inclusions")
    check(float(array[:, :, nx // 2].std()) > 0.16, "industrial CT middle slice has visible structure")


def verify_cfd(array):
    check(array.shape == (24, 64, 48, 4), "CFD wake shape is (24, 64, 48, 4)")
    velocity = array[..., 0]
    pressure = array[..., 1]
    vorticity = array[..., 2]
    temperature = array[..., 3]
    check(float(velocity.max() - velocity.min()) > 5.0, "CFD velocity has wake deficit contrast")
    check(99.0 < float(pressure.mean()) < 103.0, "CFD pressure has plausible engineering scale")
    check(float(vorticity.max() - vorticity.min()) > 6.0, "CFD vorticity has shear-layer contrast")
    check(float(temperature.max() - temperature.min()) > 3.5, "CFD temperature has thermal plume contrast")
    time_delta = float(np.mean(np.abs(array[-1, :, :, 0] - array[0, :, :, 0])))
    check(time_delta > 0.20, "CFD wake evolves over time")


def verify_npz(path):
    expected = {
        "bearing_vibration": (4096,),
        "bridge_sensor_table": (600, 6),
        "fem_stress_plate": (128, 128),
        "industrial_ct_volume": (48, 64, 48),
        "cfd_wake": (8, 32, 24, 4),
    }
    with np.load(path) as data:
        check(set(data.keys()) == set(expected), "NPZ contains expected engineering keys")
        for key, shape in expected.items():
            array = data[key]
            check(array.shape == shape, f"NPZ key {key} has shape {shape}")
            check(str(array.dtype) == "float32", f"NPZ key {key} is float32")
            assert_finite(f"NPZ key {key}", array)


def main():
    manifest = load_manifest()
    entries = {entry["file"]: entry for entry in manifest["files"]}
    check(set(EXPECTED_SHAPES).issubset(entries), "manifest contains all engcase NPY files")

    for name, entry in entries.items():
        path = DATA_DIR / name
        check(path.exists(), f"{name} exists")

        if entry["kind"] == "npz":
            verify_npz(path)
            continue

        array = np.load(path, mmap_mode="r")
        check(tuple(array.shape) == EXPECTED_SHAPES[name], f"{name} has expected shape")
        check(list(array.shape) == entry["shape"], f"{name} matches manifest shape")
        check(str(array.dtype) == "float32", f"{name} is float32")
        check(str(array.dtype) == entry["dtype"], f"{name} matches manifest dtype")
        assert_finite(name, array)

        if name == "engcase_bearing_vibration_4096.npy":
            verify_bearing(array)
        elif name == "engcase_bridge_sensor_table_2400x6.npy":
            verify_bridge(array)
        elif name == "engcase_fem_stress_plate_512x512.npy":
            verify_fem(array)
        elif name == "engcase_industrial_ct_volume_96x128x96.npy":
            verify_ct_volume(array)
        elif name == "engcase_cfd_wake_24x64x48x4.npy":
            verify_cfd(array)

    print("\nAll engcase engineering data checks passed.")


if __name__ == "__main__":
    main()
