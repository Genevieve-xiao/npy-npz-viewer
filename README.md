# NPY/NPZ Viewer

A PySide6 desktop viewer for NumPy `.npy` / `.npz` files and local `.zarr`
stores. It is focused on large multidimensional scientific arrays such as
seismic volumes, 1D signals, 2D tables, images, multichannel data, and 4D
attribute volumes.

## Quick Start

```bash
conda create -n npy-viewer python=3.11 -y
conda activate npy-viewer
pip install -e ".[dev]"
python main.py
```

After installation you can also launch with:

```bash
npy-npz-viewer
```

Open `.npy` / `.npz` files from the left panel, drag them into the window, or
use the Zarr directory button for local `.zarr` stores.

## Main Features

- Standard `src/npy_npz_viewer` package layout with a console entry point.
- Read `.npy` files through read-only memory mapping.
- Browse `.npz` keys and `.zarr` group arrays from the same key-selection UI.
- Use Dask-backed lazy arrays for large `.npy` files and all `.zarr` arrays.
- Compute previews, sampled statistics, projections, and plot inputs through
  bounded compute helpers so the UI avoids accidental full-array materialization.
- Apply non-destructive dimension filters before slicing, previewing, plotting,
  statistics, and CSV export.
- Slice arbitrary axes with Python-style slice syntax.
- Preview high-dimensional data as flattened values, bounded 2D slices, or axis
  summaries.
- Plot semantic data types:
  - 1D sequence: line chart, histogram
  - 2D table: multi-line chart, column histogram, scatter, correlation heatmap
  - 2D image/matrix: heatmap, image display with axis swap/order controls
  - 3D volume: slice heatmap, projection, 3D scatter/surface/wireframe/contour,
    voxel, slice stack
  - 3D multichannel: channel heatmap/image
  - 4D volume: channel selection plus slice/projection
- Run file loading, statistics, and preview refresh in background Qt tasks.
- Generate benchmark CSV/Markdown output for performance-report material.

## Project Layout

```text
npy-npz-Viewer/
|-- main.py                       # Application launcher
|-- pyproject.toml                # Package metadata and console script
|-- src/npy_npz_viewer/
|   |-- app.py                    # Main PySide6 application
|   |-- config.py                 # Central runtime thresholds
|   |-- logging_config.py         # Logging setup
|   |-- core/                     # Loading, handles, compute, slicing, stats, plotting
|   |-- ui/                       # PySide6 widgets and panels
|   `-- utils/                    # Preview and large-data helpers
|-- tests/                        # Pytest coverage for Dask/Zarr engine
|-- scripts/benchmark_large_arrays.py
|-- test_data/                    # Tracked engcase data and generators
`-- docs/
```

`main.py` is the single local launcher. New code should import from
`npy_npz_viewer`.

## Large Data Notes

- `.npy` remains best for large local arrays because it supports memory mapping.
- `.zarr` is supported for chunked multidimensional stores and is loaded lazily
  through Dask.
- Start with dimension filtering, singleton-axis removal, slicing, or one-click
  2D view before expensive plots.
- Statistics use bounded sampling for large arrays.
- Projections and 3D plots use quality controls and downsampling to keep the UI
  responsive.
- CSV export is intended for current 1D/2D views, not raw high-dimensional
  volumes.

## Verification

```bash
python -m compileall -q main.py src test_data tests scripts
pytest
python test_data/verify_functions.py
python test_data/create_engineering_test_data.py
python test_data/verify_engineering_test_data.py
```

Manual testing is documented in `docs/TESTING_GUIDE.md`.

Run a small benchmark smoke test:

```bash
python scripts/benchmark_large_arrays.py --shape 96x128x80
```

Benchmark output is written under `benchmark_results/`, which is ignored by git.

## Test Data Policy

The repository tracks the compact `engcase_` engineering test suite:

| File | Scenario | First useful view |
| --- | --- | --- |
| `test_data/engcase_bearing_vibration_4096.npy` | 1D bearing vibration diagnosis | Line chart |
| `test_data/engcase_bridge_sensor_table_2400x6.npy` | 2D bridge structural health table | Multi-line chart or correlation heatmap |
| `test_data/engcase_fem_stress_plate_512x512.npy` | 2D finite-element stress cloud | Heatmap |
| `test_data/engcase_industrial_ct_volume_96x128x96.npy` | 3D industrial CT inspection | Slice heatmap or projection |
| `test_data/engcase_cfd_wake_24x64x48x4.npy` | 4D transient CFD wake channels | Channel slice heatmap |
| `test_data/engcase_mixed_suite.npz` | Mixed NPZ engineering suite | Switch each key |

`test_data/engcase_manifest.json` describes shapes, semantics, recommended
plots, and expected visible features. Regenerate or verify the suite with:

```bash
python test_data/create_engineering_test_data.py
python test_data/verify_engineering_test_data.py
```

Detailed click-by-click manual validation is in `docs/TESTING_GUIDE.md`.

Arbitrary large `.npy/.npz/.zarr` data is intentionally not tracked in git. Put
very large local samples under `local_data/` or attach them as GitHub Release
assets instead of committing them.

## License

MIT. See [LICENSE](LICENSE).
