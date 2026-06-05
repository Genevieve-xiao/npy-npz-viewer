"""
Create deterministic engineering test data for the NPY/NPZ viewer.

The engcase_ suite is designed for realistic visual QA and course reports.
It avoids np.random so generated arrays are reproducible across machines.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


OUT_DIR = Path(__file__).resolve().parent


def as_float32(array):
    return np.asarray(array, dtype=np.float32)


def normalize(array, low=0.0, high=1.0):
    array = np.asarray(array, dtype=np.float32)
    amin = float(array.min())
    amax = float(array.max())
    if amax == amin:
        return np.full_like(array, low, dtype=np.float32)
    scaled = (array - amin) / (amax - amin)
    return as_float32(low + scaled * (high - low))


def sigmoid(array):
    return 1.0 / (1.0 + np.exp(-array))


def save_npy(name, array, semantic, plots, features, recommended=None):
    path = OUT_DIR / name
    array = as_float32(array)
    np.save(path, array)
    return {
        "file": name,
        "kind": "npy",
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "elements": int(array.size),
        "semantic": semantic,
        "recommended_plots": plots,
        "recommended_operations": recommended or [],
        "expected_visible_features": features,
        "size_bytes": path.with_suffix(".npy").stat().st_size,
    }


def create_bearing_vibration(n=4096):
    """1D rotating machinery vibration with deterministic bearing impacts."""
    duration = 4.0
    t = np.linspace(0.0, duration, n, endpoint=False, dtype=np.float32)

    shaft_hz = 24.0
    speed_ramp_hz = 1.8
    phase = 2.0 * np.pi * (shaft_hz * t + 0.5 * speed_ramp_hz / duration * t * t)
    signal = 0.16 * np.sin(phase)
    signal += 0.055 * np.sin(2.0 * phase + 0.45)
    signal += 0.032 * np.sin(4.0 * phase + 1.20)

    for center in np.arange(0.12, duration, 0.137, dtype=np.float32):
        dt = t - center
        active = dt >= 0.0
        ring = np.zeros_like(t)
        ring[active] = np.exp(-dt[active] / 0.024) * np.sin(2.0 * np.pi * 245.0 * dt[active])
        signal += 0.58 * ring
        signal += 0.22 * np.exp(-((dt - 0.002) / 0.0045) ** 2)

    deterministic_noise = 0.018 * np.sin(2.0 * np.pi * 93.7 * t + 0.4 * np.sin(2.0 * np.pi * 1.3 * t))
    deterministic_noise += 0.010 * np.sin(2.0 * np.pi * 217.0 * t + 0.7)
    signal += deterministic_noise
    signal += 0.03 * np.sin(2.0 * np.pi * 0.45 * t)

    return as_float32(signal)


def create_bridge_sensor_table(rows=2400):
    """2D structural health monitoring table for a bridge span."""
    hours = np.linspace(0.0, 24.0, rows, dtype=np.float32)
    temperature = 17.5 + 7.2 * np.sin(2.0 * np.pi * (hours - 6.0) / 24.0)
    temperature += 0.7 * np.sin(2.0 * np.pi * hours / 3.7)

    morning = np.exp(-((hours - 8.2) / 1.7) ** 2)
    noon = 0.55 * np.exp(-((hours - 13.1) / 2.5) ** 2)
    evening = 1.15 * np.exp(-((hours - 18.0) / 2.0) ** 2)
    traffic_density = 0.18 + morning + noon + evening
    load_kn = 32.0 + 42.0 * traffic_density

    for center, amplitude, width in [
        (1.3, 80.0, 0.045),
        (2.7, 56.0, 0.035),
        (5.9, 65.0, 0.050),
        (7.6, 95.0, 0.040),
        (8.4, 120.0, 0.032),
        (9.1, 74.0, 0.038),
        (12.4, 62.0, 0.050),
        (15.2, 70.0, 0.045),
        (17.5, 110.0, 0.036),
        (18.3, 132.0, 0.032),
        (19.4, 88.0, 0.040),
        (22.6, 58.0, 0.050),
    ]:
        load_kn += amplitude * np.exp(-((hours - center) / width) ** 2)

    slow_drift = 1.8 * np.sin(2.0 * np.pi * hours / 12.0 + 0.35)
    strain_micro = 6.0 + 0.73 * load_kn + 1.55 * (temperature - 17.5) + slow_drift
    strain_micro += 1.2 * np.sin(2.0 * np.pi * hours * 3.0)
    displacement_mm = 0.42 + 0.028 * load_kn + 0.035 * (temperature - 17.5)
    displacement_mm += 0.06 * np.sin(2.0 * np.pi * hours / 1.8)
    stress_mpa = 0.205 * strain_micro + 0.018 * load_kn + 0.12 * np.sin(2.0 * np.pi * hours / 6.0)

    return as_float32(np.column_stack([
        hours * 60.0,
        load_kn,
        strain_micro,
        temperature,
        displacement_mm,
        stress_mpa,
    ]))


def create_fem_stress_plate(size=512):
    """2D finite-element-like stress field for a loaded plate with a hole."""
    y, x = np.mgrid[-1.0:1.0:complex(size), -1.0:1.0:complex(size)]
    hole_x, hole_y, hole_radius = -0.22, 0.02, 0.18
    dx = x - hole_x
    dy = y - hole_y
    radius = np.sqrt(dx * dx + dy * dy)
    theta = np.arctan2(dy, dx)

    edge_load = 0.65 + 0.35 * (x + 1.0) / 2.0 + 0.08 * np.sin(2.0 * np.pi * y)
    stress_ring = 2.6 * np.exp(-((radius - hole_radius) / 0.065) ** 2) * (0.45 + np.sin(theta) ** 2)
    crack_tip = 2.2 * np.exp(-(((x - 0.04) / 0.070) ** 2 + ((y - 0.02) / 0.018) ** 2))
    support_hotspot = 1.1 * np.exp(-(((x + 0.82) / 0.080) ** 2 + ((y + 0.72) / 0.140) ** 2))
    support_hotspot += 0.85 * np.exp(-(((x + 0.82) / 0.080) ** 2 + ((y - 0.72) / 0.140) ** 2))
    mesh_texture = 0.045 * np.sin(36.0 * x) * np.sin(34.0 * y)

    plate_mask = sigmoid((radius - hole_radius) / 0.012)
    stress = (edge_load + stress_ring + crack_tip + support_hotspot + mesh_texture) * plate_mask
    stress = normalize(stress, 0.0, 1.0)
    stress[radius < hole_radius * 0.94] = 0.0

    return as_float32(stress)


def create_industrial_ct_volume(shape=(96, 128, 96)):
    """3D industrial CT volume for a cast metal part with internal defects."""
    nz, ny, nx = shape
    z, y, x = np.mgrid[
        -1.0:1.0:complex(nz),
        -1.0:1.0:complex(ny),
        -1.0:1.0:complex(nx),
    ]
    ellipsoid = (x / 0.74) ** 2 + (y / 0.62) ** 2 + (z / 0.82) ** 2
    body = sigmoid((1.0 - ellipsoid) / 0.025)
    core_gradient = 0.58 + 0.10 * (1.0 - ellipsoid)
    scan_rings = 0.028 * np.sin(42.0 * z) + 0.016 * np.cos(28.0 * y)
    volume = 0.035 + body * (core_gradient + scan_rings)

    voids = [
        (-0.28, -0.18, -0.08, 0.100),
        (0.16, 0.22, 0.24, 0.085),
        (0.36, -0.08, -0.30, 0.070),
        (-0.02, 0.34, -0.42, 0.060),
    ]
    for cx, cy, cz, radius in voids:
        defect = np.exp(-(((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) / (radius * radius)))
        volume -= 0.62 * defect * body

    crack_sheet = np.exp(-((x - (0.10 + 0.32 * z)) / 0.018) ** 2)
    crack_sheet *= np.exp(-((y + 0.08) / 0.120) ** 2)
    crack_sheet *= sigmoid((z + 0.58) / 0.06) * sigmoid((0.48 - z) / 0.06)
    volume -= 0.42 * crack_sheet * body

    inclusions = [
        (-0.48, 0.12, 0.18, 0.052),
        (0.42, -0.28, 0.06, 0.045),
        (0.04, -0.36, -0.52, 0.040),
    ]
    for cx, cy, cz, radius in inclusions:
        inclusion = np.exp(-(((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) / (radius * radius)))
        volume += 0.35 * inclusion * body

    volume = np.clip(volume, 0.0, 1.0)
    return as_float32(volume)


def create_cfd_wake(shape=(24, 64, 48, 4)):
    """4D transient CFD wake field: time, height, crosswind, channels."""
    nt, ny, nx, channels = shape
    if channels != 4:
        raise ValueError("create_cfd_wake expects four channels")

    t = np.linspace(0.0, 1.0, nt, dtype=np.float32)[:, None, None]
    y = np.linspace(0.0, 1.0, ny, dtype=np.float32)[None, :, None]
    x = np.linspace(-1.0, 1.0, nx, dtype=np.float32)[None, None, :]

    downstream_gate = sigmoid((y - 0.16) / 0.035)
    wake_center = 0.18 * np.sin(2.0 * np.pi * (1.15 * t + 0.62 * y))
    wake_width = 0.080 + 0.260 * y
    wake = np.exp(-((x - wake_center) / wake_width) ** 2) * downstream_gate
    shear_layer = np.exp(-((np.abs(x - wake_center) - 0.55 * wake_width) / 0.055) ** 2) * downstream_gate

    inlet_ripple = 0.35 * np.sin(2.0 * np.pi * (2.4 * t + 1.6 * y + 0.25 * x))
    velocity = 12.0 - 5.2 * wake + 0.55 * shear_layer + inlet_ripple
    pressure = 101.3 + 0.44 * wake - 0.025 * velocity
    pressure += 0.18 * np.cos(2.0 * np.pi * (t + 0.8 * y))
    vorticity = 7.0 * np.sin(10.0 * (x - wake_center)) * shear_layer
    vorticity += 1.2 * np.sin(2.0 * np.pi * (3.0 * t + 0.5 * y))
    temperature = 22.0 + 3.8 * np.exp(-((x - wake_center - 0.08) / (1.25 * wake_width)) ** 2) * downstream_gate
    temperature += 0.7 * np.sin(2.0 * np.pi * (t - 0.35 * y))

    return as_float32(np.stack([velocity, pressure, vorticity, temperature], axis=-1))


def create_manifest(entries, npz_entry):
    manifest = {
        "suite": "engcase",
        "version": 1,
        "description": "Deterministic engineering application data for NPY/NPZ Viewer demos.",
        "naming_rule": "engcase_<scenario>_<shape>.npy or engcase_mixed_suite.npz",
        "randomness": "No np.random calls; arrays are generated from analytic functions.",
        "files": entries + [npz_entry],
    }
    path = OUT_DIR / "engcase_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main():
    print("Creating deterministic engcase engineering test data...")
    entries = []

    bearing = create_bearing_vibration()
    entries.append(save_npy(
        "engcase_bearing_vibration_4096.npy",
        bearing,
        "1D sequence: bearing vibration diagnosis",
        ["line chart", "histogram", "sampled statistics"],
        ["shaft harmonics", "periodic impact pulses", "high-frequency resonance decay"],
        ["Use line chart first; histogram should show a long positive tail."],
    ))

    bridge = create_bridge_sensor_table()
    entries.append(save_npy(
        "engcase_bridge_sensor_table_2400x6.npy",
        bridge,
        "2D table: bridge structural health monitoring",
        ["multi-line chart", "single-column histogram", "scatter plot", "correlation heatmap"],
        ["daily traffic peaks", "truck crossing pulses", "load-strain-deflection correlation"],
        ["Use column 0 as time in minutes; compare load, strain, displacement, and stress."],
    ))

    fem = create_fem_stress_plate()
    entries.append(save_npy(
        "engcase_fem_stress_plate_512x512.npy",
        fem,
        "2D image/matrix: finite element stress cloud",
        ["heatmap", "image display"],
        ["central hole", "stress concentration ring", "crack-tip hotspot", "support hotspots"],
        ["Use heatmap with viridis, plasma, or inferno colormap."],
    ))

    ct_volume = create_industrial_ct_volume()
    entries.append(save_npy(
        "engcase_industrial_ct_volume_96x128x96.npy",
        ct_volume,
        "3D volume: industrial CT defect inspection",
        ["slice heatmap", "projection", "3D scatter", "3D voxel", "slice stack"],
        ["ellipsoid casting body", "low-density pores", "thin crack sheet", "bright inclusions"],
        ["Slice near the middle axes; use max projection to expose inclusions."],
    ))

    cfd = create_cfd_wake()
    entries.append(save_npy(
        "engcase_cfd_wake_24x64x48x4.npy",
        cfd,
        "4D transient field: CFD wake channels",
        ["slice heatmap", "projection", "channel comparison"],
        ["velocity deficit", "pressure recovery", "vorticity shear layer", "thermal plume"],
        ["Treat axis 3 as channel axis; inspect channels 0-3 separately."],
    ))

    npz_path = OUT_DIR / "engcase_mixed_suite.npz"
    np.savez_compressed(
        npz_path,
        bearing_vibration=bearing,
        bridge_sensor_table=bridge[::4],
        fem_stress_plate=fem[::4, ::4],
        industrial_ct_volume=ct_volume[::2, ::2, ::2],
        cfd_wake=cfd[::3, ::2, ::2, :],
    )
    npz_entry = {
        "file": "engcase_mixed_suite.npz",
        "kind": "npz",
        "dtype": "mixed float32 arrays",
        "semantic": "Mixed engineering suite for NPZ key switching",
        "recommended_plots": ["all supported plot groups by key"],
        "recommended_operations": ["Switch every key and confirm semantic controls refresh."],
        "expected_visible_features": ["key-specific engineering scenario, shape, and dimensionality changes"],
        "keys": {
            "bearing_vibration": list(bearing.shape),
            "bridge_sensor_table": list(bridge[::4].shape),
            "fem_stress_plate": list(fem[::4, ::4].shape),
            "industrial_ct_volume": list(ct_volume[::2, ::2, ::2].shape),
            "cfd_wake": list(cfd[::3, ::2, ::2, :].shape),
        },
        "size_bytes": npz_path.stat().st_size,
    }

    manifest = create_manifest(entries, npz_entry)
    print(f"Created {len(manifest['files'])} engcase files plus engcase_manifest.json")
    for entry in manifest["files"]:
        if entry["kind"] == "npy":
            print(f"  {entry['file']}: shape={tuple(entry['shape'])}, dtype={entry['dtype']}")
        else:
            print(f"  {entry['file']}: keys={', '.join(entry['keys'].keys())}")


if __name__ == "__main__":
    main()
