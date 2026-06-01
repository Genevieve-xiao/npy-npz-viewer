# Performance Guide

This project is optimized for quick inspection of NumPy and Zarr arrays, with
special attention to seismic and other large volume data.

## Recommended Workflow

1. Open `.npy` files directly when possible. They use read-only memory mapping.
2. Use `.zarr` stores when data is already chunked or larger than memory.
3. Use dimension filtering to keep/drop axes or columns before plotting.
4. Use slicing or one-click 2D view to reduce high-dimensional data.
5. Preview with "auto", "2D slice", or "axis summary" before running heavy plots.
6. Use fast/medium/high quality controls for volume projections and 3D plots.

## Large File Policy

Large sample data should not be committed to git. Keep local-only data under
`local_data/` or publish it as a GitHub Release asset. The repository's
`test_data/` folder contains tiny smoke fixtures and deterministic generators.

## Data Format Notes

- `.npy` is best for large arrays because it supports memory mapping.
- `.npz` is good for bundled demos and key-switching tests, but very large keys
  may need to be decompressed into memory.
- `.zarr` is best when arrays are chunked or grouped. Zarr arrays are wrapped in
  Dask so previews, sampled statistics, and projections can stay bounded.
- CSV export is guarded for large views and is only available after converting
  data to a 1D/2D current view.

## Benchmarking

Generate performance-report material with:

```powershell
python scripts/benchmark_large_arrays.py --shape 96x128x80
```

The script writes CSV and Markdown results under `benchmark_results/`, which is
ignored by git.

## Common Scenarios

- `601 x 951 x 288` cube: open as `.npy`, inspect middle-slice preview, then use
  one-click 2D view or slice axis 2 before plotting.
- `(N, C)` table: keep the meaningful columns, use column 0 as X if it is depth,
  time, or another monotonic coordinate.
- 4D attribute volume: choose the channel axis first, then slice/projection
  controls operate on the selected 3D volume.

## Troubleshooting

- If the UI feels slow after opening a large volume, wait for the background
  preview/statistics task to complete or cancel it.
- If a plot is unreadable or slow, first reduce the current view with slicing,
  filtering, or lower plot quality.
- If a file is too large for GitHub, do not commit it. Regenerate it locally or
  publish it as a release artifact.
