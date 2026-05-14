# NPY/NPZ Viewer

A PySide6 desktop viewer for NumPy `.npy` and `.npz` files, focused on large
seismic/volume data as well as 1D signals, 2D tables, images, multichannel data,
and 4D attribute volumes.

## Quick Start

```bash
conda create -n npy-viewer python=3.11 -y
conda activate npy-viewer
pip install -r requirements.txt
python main_v2.2.py
```

You can open files with the button in the left panel or drag a `.npy` / `.npz`
file directly into the window.

## Main Features

- Read `.npy` with read-only memory mapping for faster large-file startup.
- Browse `.npz` keys without changing the main workflow.
- Apply non-destructive dimension filters before slicing, previewing, plotting,
  statistics, and CSV export.
- Slice arbitrary axes with Python-style slice syntax.
- Preview high-dimensional data as flattened values, 2D middle slices, or axis
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
- Use one-click 2D view for large/high-dimensional arrays.

## Project Layout

```text
npy-npz-Viewer/
├── main_v2.2.py              # Recommended application entrypoint
├── core/                     # Loading, filtering, slicing, stats, semantics, plotting
├── ui/                       # PySide6 widgets and panels for the v2.2 UI
├── utils/                    # Preview and large-data helpers
├── test_data/                # Tiny fixtures plus deterministic data generators
├── docs/                     # Testing and performance guides
├── legacy/                   # Archived v1/v2.1 code kept for reference only
├── requirements.txt
└── README.md
```

Only `main_v2.2.py` is recommended for current use. Files under `legacy/` are
not part of the maintained runtime path.

## Test Data Policy

Large binary `.npy/.npz` data is intentionally not tracked in git. The repository
keeps only tiny smoke fixtures and deterministic generators.

Generate semantic UX data locally:

```bash
python test_data/create_uxcase_test_data.py
python test_data/verify_uxcase_data.py
```

This creates `uxcase_*.npy`, `uxcase_mixed_suite.npz`, and
`uxcase_manifest.json` in `test_data/`. These generated files are ignored by
git. Put very large local samples such as F3 seismic cubes under `local_data/`
or attach them as GitHub Release assets instead of committing them.

See [test_data/README.md](test_data/README.md) for the exact fixture policy and
[docs/TESTING_GUIDE.md](docs/TESTING_GUIDE.md) for a GUI test checklist.

## Large Data Notes

- Prefer `.npy` over `.npz` for very large arrays because `.npy` can be memory
  mapped.
- Start with dimension filtering, singleton-axis removal, slicing, or one-click
  2D view before expensive plots.
- Projections and 3D plots use downsampling/quality controls to keep the UI
  responsive.
- CSV export is intended for current 1D/2D views, not raw high-dimensional
  volumes.

See [docs/PERFORMANCE_GUIDE.md](docs/PERFORMANCE_GUIDE.md) for more detail.

## Verification

```bash
python -m py_compile main_v2.2.py core/*.py ui/*.py utils/*.py test_data/*.py
python test_data/verify_functions.py
python test_data/create_uxcase_test_data.py
python test_data/verify_uxcase_data.py
```

## GitHub Publishing Checklist

- Do not commit generated `uxcase_*.npy`, generated `.npz`, or local F3 data.
- Keep large local samples in `local_data/` or publish them as Release assets.
- Confirm `.DS_Store`, `._*`, `__pycache__/`, and `*.pyc` are absent.
- Use `python main_v2.2.py` as the public launch command.

## License

MIT. See [LICENSE](LICENSE).
