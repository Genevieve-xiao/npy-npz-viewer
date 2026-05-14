"""
Create deterministic UX test data for the NPY/NPZ viewer.

All files use the uxcase_ prefix and avoid random numbers so visual
regressions are easy to compare by eye.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


OUT_DIR = Path(__file__).resolve().parent


def as_float32(array):
    return np.asarray(array, dtype=np.float32)


def normalize(array, low=-1.0, high=1.0):
    array = np.asarray(array, dtype=np.float32)
    amin = float(array.min())
    amax = float(array.max())
    if amax == amin:
        return np.full_like(array, low, dtype=np.float32)
    scaled = (array - amin) / (amax - amin)
    return as_float32(low + scaled * (high - low))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


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


def create_heartbeat_signal():
    t = np.linspace(0.0, 12.0, 2400, dtype=np.float32)
    signal = 0.06 * np.sin(2.0 * np.pi * 0.33 * t)
    signal += 0.025 * np.sin(2.0 * np.pi * 3.2 * t)

    for center in np.arange(0.7, 12.0, 0.82, dtype=np.float32):
        signal += 0.18 * np.exp(-((t - (center - 0.18)) / 0.045) ** 2)
        signal -= 0.32 * np.exp(-((t - (center - 0.025)) / 0.015) ** 2)
        signal += 1.05 * np.exp(-((t - center) / 0.018) ** 2)
        signal -= 0.22 * np.exp(-((t - (center + 0.035)) / 0.018) ** 2)
        signal += 0.34 * np.exp(-((t - (center + 0.23)) / 0.09) ** 2)

    return as_float32(signal)


def create_well_log_table():
    depth = np.linspace(1000.0, 3399.0, 2400, dtype=np.float32)
    layer = np.sin(depth / 115.0) + 0.35 * np.sin(depth / 37.0)
    sand_a = np.exp(-((depth - 1580.0) / 145.0) ** 2)
    sand_b = 0.75 * np.exp(-((depth - 2360.0) / 210.0) ** 2)
    sand_c = 0.55 * np.exp(-((depth - 3090.0) / 120.0) ** 2)
    sand = sand_a + sand_b + sand_c
    shale_step = 0.5 * (1.0 + np.tanh((depth - 2200.0) / 35.0))

    gamma = 84.0 + 28.0 * layer - 48.0 * sand + 14.0 * shale_step
    density = 2.42 + 0.055 * layer - 0.16 * sand + 0.035 * shale_step
    porosity = 0.11 + 0.12 * sand - 0.022 * layer - 0.025 * shale_step
    velocity = 2650.0 + 460.0 * density - 900.0 * porosity + 80.0 * np.sin(depth / 260.0)
    impedance = density * velocity

    return as_float32(np.column_stack([depth, gamma, density, porosity, velocity, impedance]))


def create_fault_slice_image(size=512):
    y, x = np.mgrid[-1.0:1.0:complex(size), -1.0:1.0:complex(size)]
    fault_line = 0.18 * np.sin(2.4 * y) - 0.05
    throw = np.where(x > fault_line, 0.22, -0.04)
    warped_y = y + throw + 0.05 * np.sin(5.0 * x)

    reflectors = np.sin(45.0 * warped_y + 4.0 * np.sin(5.0 * x))
    reflectors += 0.55 * np.sin(28.0 * warped_y + 2.5 * np.cos(4.0 * x))
    channel = 1.7 * np.exp(-((warped_y + 0.30 + 0.12 * np.sin(3.0 * x)) / 0.055) ** 2)
    bright_spot = 2.1 * np.exp(-(((x - 0.34) / 0.18) ** 2 + ((y + 0.18) / 0.12) ** 2))
    shadow = -1.3 * np.exp(-(((x + 0.45) / 0.16) ** 2 + ((y - 0.28) / 0.18) ** 2))
    fault_marker = 0.45 * np.exp(-((x - fault_line) / 0.012) ** 2)

    return normalize(reflectors + channel + bright_spot + shadow + fault_marker)


def create_landcover_rgb(size=256):
    y, x = np.mgrid[0.0:1.0:complex(size), 0.0:1.0:complex(size)]
    terrain = 0.5 + 0.5 * np.sin(8.0 * x + 4.0 * y) * np.cos(5.0 * y)
    river_center = 0.52 + 0.10 * np.sin(8.0 * x)
    river = np.exp(-((y - river_center) / 0.035) ** 2)
    vegetation = sigmoid(7.0 * (terrain - 0.42)) * (1.0 - 0.75 * river)
    urban_grid = ((np.sin(70.0 * x) > 0.82) | (np.sin(62.0 * y) > 0.86)).astype(np.float32)
    urban_mask = np.exp(-(((x - 0.70) / 0.22) ** 2 + ((y - 0.25) / 0.17) ** 2))
    urban = urban_grid * sigmoid(8.0 * (urban_mask - 0.35))
    bare = sigmoid(9.0 * (0.42 - terrain)) * (1.0 - river)

    red = 0.15 + 0.34 * bare + 0.42 * urban + 0.10 * vegetation
    green = 0.18 + 0.62 * vegetation + 0.18 * bare + 0.25 * urban
    blue = 0.18 + 0.75 * river + 0.12 * urban + 0.05 * bare
    rgb = np.stack([red, green, blue], axis=-1)
    return as_float32(np.clip(rgb, 0.0, 1.0))


def create_seismic_volume(shape=(96, 128, 80), dtype=np.float32):
    n0, n1, n2 = shape
    inline = np.linspace(-1.0, 1.0, n0, dtype=dtype)[:, None, None]
    xline = np.linspace(-1.0, 1.0, n1, dtype=dtype)[None, :, None]
    depth = np.linspace(-1.0, 1.0, n2, dtype=dtype)[None, None, :]

    fault_surface = 0.15 * np.sin(3.0 * inline) - 0.08
    throw = np.where(xline > fault_surface, 0.18, -0.05).astype(dtype)
    structure = depth + throw + 0.08 * np.sin(2.5 * inline) + 0.06 * np.cos(3.2 * xline)

    reflectors = np.sin(34.0 * structure) + 0.45 * np.sin(57.0 * structure + 2.0 * xline)
    channel_path = 0.30 * np.sin(2.5 * inline) - 0.18
    channel = np.exp(-((xline - channel_path) / 0.10) ** 2 - ((depth + 0.18) / 0.12) ** 2)
    gas = np.exp(-((inline - 0.32) / 0.22) ** 2 - ((xline + 0.28) / 0.18) ** 2 - ((depth + 0.02) / 0.12) ** 2)
    salt = -1.6 * np.exp(-((inline + 0.48) / 0.18) ** 2 - ((xline - 0.35) / 0.20) ** 2 - ((depth - 0.12) / 0.25) ** 2)
    fault_marker = 0.35 * np.exp(-((xline - fault_surface) / 0.025) ** 2)

    volume = reflectors + 1.4 * channel + 2.2 * gas + salt + fault_marker
    return normalize(volume)


def create_reservoir_4d():
    base = create_seismic_volume((48, 64, 40))
    n0, n1, n2 = base.shape
    inline = np.linspace(-1.0, 1.0, n0, dtype=np.float32)[:, None, None]
    xline = np.linspace(-1.0, 1.0, n1, dtype=np.float32)[None, :, None]
    depth = np.linspace(-1.0, 1.0, n2, dtype=np.float32)[None, None, :]

    reservoir = np.exp(-((inline - 0.10) / 0.45) ** 2 - ((xline + 0.15) / 0.32) ** 2 - ((depth + 0.10) / 0.20) ** 2)
    channel = np.exp(-((xline - 0.22 * np.sin(3.0 * inline)) / 0.11) ** 2 - ((depth + 0.24) / 0.10) ** 2)
    grad0, grad1, grad2 = np.gradient(base)
    discontinuity = np.sqrt(grad0 * grad0 + grad1 * grad1 + grad2 * grad2)
    semblance = normalize(1.0 / (1.0 + 7.0 * discontinuity), 0.0, 1.0)
    porosity = np.clip(0.08 + 0.16 * reservoir + 0.08 * channel - 0.03 * depth, 0.02, 0.34)
    facies = sigmoid(28.0 * (porosity - 0.17) + 1.8 * base)

    return as_float32(np.stack([
        normalize(base),
        semblance,
        porosity,
        facies,
    ], axis=-1))


def create_manifest(entries, npz_entry):
    manifest = {
        "suite": "uxcase",
        "version": 1,
        "description": "Deterministic semantic UX test data for NPY/NPZ Viewer v2.2.",
        "naming_rule": "uxcase_<semantic>_<shape>.npy or uxcase_mixed_suite.npz",
        "randomness": "No np.random calls; all arrays are generated from deterministic analytic functions.",
        "files": entries + [npz_entry],
    }
    path = OUT_DIR / "uxcase_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main():
    print("Creating deterministic uxcase test data...")
    entries = []

    heartbeat = create_heartbeat_signal()
    entries.append(save_npy(
        "uxcase_signal_heartbeat_1d.npy",
        heartbeat,
        "1D sequence: heartbeat / vibration signal",
        ["折线图", "直方图"],
        ["sharp periodic QRS-like spikes", "low baseline drift", "secondary recovery waves"],
        ["Use line chart first; histogram should show long positive tail."],
    ))

    well_log = create_well_log_table()
    entries.append(save_npy(
        "uxcase_well_log_table_2400x6.npy",
        well_log,
        "2D tabular: synthetic well log",
        ["多折线图", "单列直方图", "散点图", "相关性热力图"],
        ["monotonic depth column", "three sand-rich intervals", "porosity and gamma anti-correlation"],
        ["X axis: column 0 depth; compare gamma, porosity, impedance."],
    ))

    fault_image = create_fault_slice_image()
    entries.append(save_npy(
        "uxcase_fault_slice_image_512x512.npy",
        fault_image,
        "2D image/matrix: seismic fault slice",
        ["热力图", "图像显示"],
        ["offset reflectors across a curved fault", "bright channel and gas spot", "dark shadow anomaly"],
        ["Try seismic/coolwarm colormap, swap axes, and descending Y axis."],
    ))

    rgb = create_landcover_rgb()
    entries.append(save_npy(
        "uxcase_landcover_rgb_256x256x3.npy",
        rgb,
        "3D multichannel: RGB landcover image",
        ["通道热力图", "通道图像"],
        ["blue meandering river", "green vegetation", "red-gray urban grid"],
        ["Inspect channels 0, 1, and 2 separately."],
    ))

    seismic = create_seismic_volume()
    entries.append(save_npy(
        "uxcase_seismic_volume_96x128x80.npy",
        seismic,
        "3D volume: seismic cube",
        ["切片热力图", "投影图", "3D散点图", "3D表面图", "3D线框图", "3D等高线图", "3D体素图", "3D切片堆叠图"],
        ["faulted reflectors", "sinuous channel", "bright gas anomaly", "negative salt-like body"],
        ["Slice axis 2 near index 40; projection method max; fast quality for 3D plots."],
    ))

    reservoir = create_reservoir_4d()
    entries.append(save_npy(
        "uxcase_reservoir_4d_48x64x40x4.npy",
        reservoir,
        "4D volume: reservoir attributes",
        ["切片热力图", "投影图"],
        ["channel 0 amplitude", "channel 1 semblance", "channel 2 porosity", "channel 3 facies probability"],
        ["Channel axis 3; inspect channels 0-3 and switch projection mean/max."],
    ))

    stress = create_seismic_volume((192, 256, 160))
    entries.append(save_npy(
        "uxcase_stress_seismic_cube_192x256x160.npy",
        stress,
        "3D stress volume: larger seismic cube",
        ["切片热力图", "投影图", "3D散点图", "3D切片堆叠图"],
        ["same deterministic seismic semantics at larger scale", "tests mmap, preview, sampled stats, and downsampling"],
        ["Open by drag-and-drop; use auto preview, cancel refresh, and one-click 2D view."],
    ))

    npz_path = OUT_DIR / "uxcase_mixed_suite.npz"
    np.savez(
        npz_path,
        heartbeat=heartbeat[:600],
        well_log=well_log[::4],
        fault_image=fault_image[::4, ::4],
        landcover_rgb=rgb[::2, ::2, :],
        seismic_volume=seismic[::3, ::4, ::2],
        reservoir_4d=reservoir[::2, ::2, ::2, :],
    )
    npz_entry = {
        "file": "uxcase_mixed_suite.npz",
        "kind": "npz",
        "dtype": "mixed float32 arrays",
        "semantic": "Mixed semantic suite for NPZ key switching",
        "recommended_plots": ["all supported plot groups by key"],
        "recommended_operations": ["Switch each key and confirm UI parameters refresh."],
        "expected_visible_features": ["key-specific shape and semantic changes"],
        "keys": {
            "heartbeat": list(heartbeat[:600].shape),
            "well_log": list(well_log[::4].shape),
            "fault_image": list(fault_image[::4, ::4].shape),
            "landcover_rgb": list(rgb[::2, ::2, :].shape),
            "seismic_volume": list(seismic[::3, ::4, ::2].shape),
            "reservoir_4d": list(reservoir[::2, ::2, ::2, :].shape),
        },
        "size_bytes": npz_path.stat().st_size,
    }

    manifest = create_manifest(entries, npz_entry)
    print(f"Created {len(manifest['files'])} uxcase files plus uxcase_manifest.json")
    for entry in manifest["files"]:
        if entry["kind"] == "npy":
            print(f"  {entry['file']}: shape={tuple(entry['shape'])}, dtype={entry['dtype']}")
        else:
            print(f"  {entry['file']}: keys={', '.join(entry['keys'].keys())}")


if __name__ == "__main__":
    main()
