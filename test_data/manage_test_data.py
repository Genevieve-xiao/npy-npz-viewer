"""
Manage deterministic engineering test data for NPY/NPZ/Zarr Viewer.

Commands:
  generate     Generate canonical NPY files plus equivalent NPZ and Zarr suites.
  verify-data  Verify engcase data features and cross-format equality.
  verify-core  Run lightweight non-GUI smoke checks.
  verify-all   Run verify-data and verify-core.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DATA_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = DATA_DIR / "engcase_manifest.json"
NPZ_PATH = DATA_DIR / "engcase_suite.npz"
ZARR_PATH = DATA_DIR / "engcase_suite.zarr"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


SCENARIOS = [
    {
        "key": "bearing_vibration",
        "file": "engcase_bearing_vibration_4096.npy",
        "semantic": "1D sequence: bearing vibration diagnosis",
        "plots": ["line chart", "histogram", "sampled statistics"],
        "features": ["shaft harmonics", "periodic impact pulses", "high-frequency resonance decay"],
        "recommended": ["Use line chart first; histogram should show a long positive tail."],
        "chunks": (1024,),
    },
    {
        "key": "bridge_sensor_table",
        "file": "engcase_bridge_sensor_table_2400x6.npy",
        "semantic": "2D table: bridge structural health monitoring",
        "plots": ["multi-line chart", "single-column histogram", "scatter plot", "correlation heatmap"],
        "features": ["daily traffic peaks", "truck crossing pulses", "load-strain-deflection correlation"],
        "recommended": ["Use column 0 as time in minutes; compare load, strain, displacement, and stress."],
        "chunks": (600, 6),
    },
    {
        "key": "fem_stress_plate",
        "file": "engcase_fem_stress_plate_512x512.npy",
        "semantic": "2D image/matrix: finite element stress cloud",
        "plots": ["heatmap", "image display"],
        "features": ["central hole", "stress concentration ring", "crack-tip hotspot", "support hotspots"],
        "recommended": ["Use heatmap with viridis, plasma, or inferno colormap."],
        "chunks": (128, 128),
    },
    {
        "key": "industrial_ct_volume",
        "file": "engcase_industrial_ct_volume_96x128x96.npy",
        "semantic": "3D volume: industrial CT defect inspection",
        "plots": ["slice heatmap", "projection", "3D scatter", "3D voxel", "slice stack"],
        "features": ["ellipsoid casting body", "low-density pores", "thin crack sheet", "bright inclusions"],
        "recommended": ["Slice near the middle axes; use max projection to expose inclusions."],
        "chunks": (16, 32, 32),
    },
    {
        "key": "cfd_wake",
        "file": "engcase_cfd_wake_24x64x48x4.npy",
        "semantic": "4D transient field: CFD wake channels",
        "plots": ["slice heatmap", "projection", "channel comparison"],
        "features": ["velocity deficit", "pressure recovery", "vorticity shear layer", "thermal plume"],
        "recommended": ["Treat axis 3 as channel axis; inspect channels 0-3 separately."],
        "chunks": (4, 32, 24, 4),
    },
]


EXPECTED_SHAPES = {
    "engcase_bearing_vibration_4096.npy": (4096,),
    "engcase_bridge_sensor_table_2400x6.npy": (2400, 6),
    "engcase_fem_stress_plate_512x512.npy": (512, 512),
    "engcase_industrial_ct_volume_96x128x96.npy": (96, 128, 96),
    "engcase_cfd_wake_24x64x48x4.npy": (24, 64, 48, 4),
}


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


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    print(f"OK: {message}")


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


def generate_arrays():
    return {
        "bearing_vibration": create_bearing_vibration(),
        "bridge_sensor_table": create_bridge_sensor_table(),
        "fem_stress_plate": create_fem_stress_plate(),
        "industrial_ct_volume": create_industrial_ct_volume(),
        "cfd_wake": create_cfd_wake(),
    }


def save_npy_atomic(path, array):
    array = as_float32(array)
    if path.exists():
        try:
            existing = np.load(path)
            if existing.shape == array.shape and existing.dtype == array.dtype and np.array_equal(existing, array):
                return
        except Exception:
            pass

    temp_path = path.with_name(f"{path.stem}.tmp.npy")
    np.save(temp_path, array)
    try:
        temp_path.replace(path)
    except PermissionError:
        if path.exists():
            existing = np.load(path)
            if existing.shape == array.shape and existing.dtype == array.dtype and np.array_equal(existing, array):
                temp_path.unlink(missing_ok=True)
                return
        raise


def scenario_by_key():
    return {scenario["key"]: scenario for scenario in SCENARIOS}


def arrays_from_npy():
    scenarios = scenario_by_key()
    arrays = {}
    for key, scenario in scenarios.items():
        path = DATA_DIR / scenario["file"]
        arrays[key] = np.load(path, mmap_mode="r")
    return arrays


def create_zarr_suite(arrays):
    try:
        import zarr
    except ImportError as exc:
        raise RuntimeError("Zarr support is not installed; install project dependencies first.") from exc

    if ZARR_PATH.exists():
        shutil.rmtree(ZARR_PATH)

    root = zarr.open_group(str(ZARR_PATH), mode="w")
    for key, array in arrays.items():
        chunks = scenario_by_key()[key]["chunks"]
        if hasattr(root, "create_array"):
            root.create_array(key, data=np.asarray(array), chunks=chunks)
        else:
            root.create_dataset(key, data=np.asarray(array), chunks=chunks)


def create_manifest(arrays):
    scenarios = scenario_by_key()
    npy_entries = []
    for key, array in arrays.items():
        scenario = scenarios[key]
        path = DATA_DIR / scenario["file"]
        npy_entries.append({
            "file": scenario["file"],
            "kind": "npy",
            "key": key,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "elements": int(array.size),
            "semantic": scenario["semantic"],
            "recommended_plots": scenario["plots"],
            "recommended_operations": scenario["recommended"],
            "expected_visible_features": scenario["features"],
            "size_bytes": path.stat().st_size,
        })

    keyed_shapes = {key: list(array.shape) for key, array in arrays.items()}
    keyed_chunks = {key: list(scenario_by_key()[key]["chunks"]) for key in arrays}
    files = npy_entries + [
        {
            "file": NPZ_PATH.name,
            "kind": "npz",
            "dtype": "float32 arrays",
            "semantic": "Canonical engineering suite for NPZ key switching",
            "recommended_plots": ["all supported plot groups by key"],
            "recommended_operations": ["Switch every key and compare with the matching NPY file."],
            "expected_visible_features": ["same data as canonical NPY files, grouped by engineering scenario"],
            "keys": keyed_shapes,
            "source_files": {key: scenario_by_key()[key]["file"] for key in arrays},
            "size_bytes": NPZ_PATH.stat().st_size,
        },
        {
            "file": ZARR_PATH.name,
            "kind": "zarr",
            "dtype": "float32 arrays",
            "semantic": "Canonical engineering suite for Zarr group key switching",
            "recommended_plots": ["all supported plot groups by key"],
            "recommended_operations": ["Open with the Zarr directory button and compare each key with NPY/NPZ."],
            "expected_visible_features": ["same data as canonical NPY files, stored as chunked Zarr arrays"],
            "keys": keyed_shapes,
            "chunks": keyed_chunks,
            "source_files": {key: scenario_by_key()[key]["file"] for key in arrays},
        },
    ]

    manifest = {
        "suite": "engcase",
        "version": 2,
        "description": "Deterministic engineering application data for NPY/NPZ/Zarr Viewer demos.",
        "naming_rule": "Five canonical engcase_*.npy files plus engcase_suite.npz and engcase_suite.zarr suites.",
        "randomness": "No np.random calls; arrays are generated from analytic functions.",
        "canonical_format": "npy",
        "equivalent_formats": [NPZ_PATH.name, ZARR_PATH.name],
        "files": files,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def generate():
    print("Creating deterministic engcase engineering test data...")
    arrays = generate_arrays()
    scenarios = scenario_by_key()

    for key, array in arrays.items():
        save_npy_atomic(DATA_DIR / scenarios[key]["file"], array)

    arrays = {key: np.load(DATA_DIR / scenarios[key]["file"], mmap_mode="r") for key in arrays}
    np.savez_compressed(NPZ_PATH, **{key: np.asarray(array) for key, array in arrays.items()})
    create_zarr_suite(arrays)
    manifest = create_manifest(arrays)

    print(f"Created {len(manifest['files'])} engcase entries plus engcase_manifest.json")
    for entry in manifest["files"]:
        if entry["kind"] == "npy":
            print(f"  {entry['file']}: shape={tuple(entry['shape'])}, dtype={entry['dtype']}")
        else:
            print(f"  {entry['file']}: keys={', '.join(entry['keys'].keys())}")


def load_manifest():
    if not MANIFEST_PATH.exists():
        raise AssertionError("engcase_manifest.json is missing; run manage_test_data.py generate first")
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


def load_zarr_arrays():
    try:
        import zarr
    except ImportError as exc:
        raise RuntimeError("Zarr support is not installed; install project dependencies first.") from exc

    root = zarr.open_group(str(ZARR_PATH), mode="r")
    return {key: root[key] for key in root.keys()}


def verify_cross_format_equality(canonical_arrays):
    with np.load(NPZ_PATH) as npz_data:
        npz_keys = set(npz_data.files)
        check(npz_keys == set(canonical_arrays), "NPZ contains exactly the canonical engcase keys")
        for key, npy_array in canonical_arrays.items():
            npz_array = npz_data[key]
            check(npz_array.shape == npy_array.shape, f"NPZ key {key} shape matches NPY")
            check(str(npz_array.dtype) == "float32", f"NPZ key {key} is float32")
            check(np.array_equal(np.asarray(npy_array), npz_array), f"NPZ key {key} equals canonical NPY")

    zarr_arrays = load_zarr_arrays()
    check(set(zarr_arrays) == set(canonical_arrays), "Zarr contains exactly the canonical engcase keys")
    for key, npy_array in canonical_arrays.items():
        zarr_array = zarr_arrays[key]
        expected_chunks = scenario_by_key()[key]["chunks"]
        check(tuple(zarr_array.shape) == npy_array.shape, f"Zarr key {key} shape matches NPY")
        check(str(zarr_array.dtype) == "float32", f"Zarr key {key} is float32")
        check(tuple(zarr_array.chunks) == expected_chunks, f"Zarr key {key} chunks match manifest policy")
        check(np.array_equal(np.asarray(npy_array), zarr_array[:]), f"Zarr key {key} equals canonical NPY")


def verify_manifest(canonical_arrays):
    manifest = load_manifest()
    entries = {entry["file"]: entry for entry in manifest["files"]}
    check(manifest["version"] == 2, "manifest version is 2")
    check(manifest["canonical_format"] == "npy", "manifest records NPY as canonical format")
    check(set(EXPECTED_SHAPES).issubset(entries), "manifest contains all canonical NPY files")
    check(NPZ_PATH.name in entries, "manifest contains NPZ suite")
    check(ZARR_PATH.name in entries, "manifest contains Zarr suite")

    for key, array in canonical_arrays.items():
        scenario = scenario_by_key()[key]
        npy_entry = entries[scenario["file"]]
        check(npy_entry["key"] == key, f"manifest maps {scenario['file']} to key {key}")
        check(npy_entry["shape"] == list(array.shape), f"manifest shape for {key} matches NPY")
        check(npy_entry["dtype"] == "float32", f"manifest dtype for {key} is float32")

        npz_entry = entries[NPZ_PATH.name]
        zarr_entry = entries[ZARR_PATH.name]
        check(npz_entry["keys"][key] == list(array.shape), f"manifest NPZ shape for {key} matches")
        check(zarr_entry["keys"][key] == list(array.shape), f"manifest Zarr shape for {key} matches")
        check(zarr_entry["chunks"][key] == list(scenario["chunks"]), f"manifest Zarr chunks for {key} matches")


def verify_data():
    check(NPZ_PATH.exists(), "engcase_suite.npz exists")
    check(ZARR_PATH.exists(), "engcase_suite.zarr exists")
    canonical_arrays = {}

    for scenario in SCENARIOS:
        name = scenario["file"]
        key = scenario["key"]
        path = DATA_DIR / name
        check(path.exists(), f"{name} exists")
        array = np.load(path, mmap_mode="r")
        canonical_arrays[key] = array
        check(tuple(array.shape) == EXPECTED_SHAPES[name], f"{name} has expected shape")
        check(str(array.dtype) == "float32", f"{name} is float32")
        assert_finite(name, array)

        if key == "bearing_vibration":
            verify_bearing(array)
        elif key == "bridge_sensor_table":
            verify_bridge(array)
        elif key == "fem_stress_plate":
            verify_fem(array)
        elif key == "industrial_ct_volume":
            verify_ct_volume(array)
        elif key == "cfd_wake":
            verify_cfd(array)

    verify_cross_format_equality(canonical_arrays)
    verify_manifest(canonical_arrays)
    print("\nAll engcase NPY/NPZ/Zarr data checks passed.")


def verify_core():
    from npy_npz_viewer.core.array_session import ArraySession
    from npy_npz_viewer.core.dimension_filter import apply_dimension_filter
    from npy_npz_viewer.core.loaders import ArrayLoader
    from npy_npz_viewer.core.slicing import ArraySlicer
    from npy_npz_viewer.core.stats import ArrayStats
    from npy_npz_viewer.core.task_result import TaskResult
    from npy_npz_viewer.utils.helpers import PreviewHelper

    loader = ArrayLoader()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        npy_path = tmp_path / "smoke_1d.npy"
        npz_path = tmp_path / "smoke_suite.npz"
        np.save(npy_path, np.arange(12, dtype=np.float32))
        np.savez(npz_path, first=np.arange(6), second=np.arange(12).reshape(3, 4))

        result = loader.load_file(str(npy_path))
        check(result["success"], "NPY smoke fixture loads")

        result = loader.load_file(str(npz_path))
        check(result["success"] and len(result["keys"]) >= 1, "NPZ smoke fixture loads")
        loader.close()

    array = np.arange(5000, dtype=np.float32).reshape(100, 50)
    stats = ArrayStats.compute_stats(array)
    check(stats["shape"] == (100, 50), "statistics report shape")
    check(stats["min"] == 0.0 and stats["max"] == 4999.0, "statistics report range")

    sliced = ArraySlicer.apply_slice(array, [":10", ":20"])
    check(sliced["success"] and sliced["array"].shape == (10, 20), "slicing works")

    filtered = apply_dimension_filter(
        array,
        [
            {"axis": 0, "mode": "keep", "spec": "0:10:2"},
            {"axis": 1, "mode": "drop", "spec": "0,2,-1"},
        ],
    )
    check(
        filtered["success"]
        and filtered["array"].shape == (5, 47)
        and filtered["axis_index_maps"][0] == [0, 2, 4, 6, 8],
        "dimension filter keep/drop works",
    )

    empty = apply_dimension_filter(array, [{"axis": 1, "mode": "drop", "spec": ":"}])
    check(not empty["success"], "empty dimension filter result is rejected")

    seismic_like = np.zeros((571551, 1, 288, 1), dtype=np.float32)
    singleton = apply_dimension_filter(
        seismic_like,
        [
            {"axis": 0, "mode": "keep", "spec": "0:2000:2"},
            {"axis": 1, "mode": "drop", "spec": ":"},
            {"axis": 3, "mode": "drop", "spec": ":"},
        ],
    )
    check(
        singleton["success"] and singleton["array"].shape == (1000, 288),
        "singleton axes can be removed",
    )

    preview, message = PreviewHelper.get_preview_slice(array)
    check(preview is not None and "100" in message, "basic preview works")

    cube = np.arange(20 * 12 * 8, dtype=np.float32).reshape(20, 12, 8)
    df, _ = PreviewHelper.build_preview(cube, mode="slice")
    check(df is not None and df.shape == (20, 12), "3D slice preview works")

    df, _ = PreviewHelper.build_preview(cube, mode="summary")
    check(df is not None and len(df) == 3, "axis summary preview works")

    session = ArraySession()
    loaded = session.load_array(cube, source_path="demo.npy")
    filtered = session.apply_filters(
        [
            {"axis": 0, "mode": "keep", "spec": "0:10:2"},
            {"axis": 2, "mode": "drop", "spec": "0,-1"},
        ]
    )
    check(
        loaded.success and filtered.success and session.current_array.shape == (5, 12, 6),
        "ArraySession state flow works",
    )

    task_result = TaskResult.ok(data={"value": 1}, sampled=True)
    check(task_result.success and task_result.sampled and task_result.data["value"] == 1, "TaskResult works")

    print("\nAll core checks passed.")


def verify_all():
    verify_data()
    verify_core()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Manage deterministic engcase test data.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("generate", help="Generate NPY, NPZ, Zarr, and manifest files.")
    subparsers.add_parser("verify-data", help="Verify engcase data and cross-format equality.")
    subparsers.add_parser("verify-core", help="Run non-GUI core smoke checks.")
    subparsers.add_parser("verify-all", help="Run verify-data and verify-core.")
    args = parser.parse_args(argv)

    if args.command == "generate":
        generate()
    elif args.command == "verify-data":
        verify_data()
    elif args.command == "verify-core":
        verify_core()
    elif args.command == "verify-all":
        verify_all()
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
