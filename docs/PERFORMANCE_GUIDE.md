# 性能指南 / Performance Guide

本项目面向 NumPy 和 Zarr 数组的快速查看与可视分析，尤其关注地震体、工业 CT、CFD 场等较大的多维工程数据。

This project is optimized for quick inspection and visual analysis of NumPy and Zarr arrays, with special attention to large multidimensional engineering data such as seismic volumes, industrial CT, and CFD fields.

## 推荐工作流 / Recommended Workflow

1. 优先直接打开 `.npy` 文件；它们会使用只读内存映射。
2. Prefer opening `.npy` files directly; they use read-only memory mapping.
3. 当数据已经分块或大于内存时，优先使用 `.zarr`。
4. Use `.zarr` stores when data is already chunked or larger than memory.
5. 绘图前先用维度筛选保留或剔除轴、列和通道。
6. Use dimension filtering to keep or drop axes, columns, and channels before plotting.
7. 对高维数据先使用切片、投影或一键二维化。
8. Reduce high-dimensional data with slicing, projection, or quick 2D view.
9. 在重型绘图前，先用 `自动`、`2D 切片` 或 `轴摘要` 预览。
10. Preview with `auto`, `2D slice`, or `axis summary` before running heavy plots.
11. 对体数据投影和 3D 绘图使用快速、较精细、高质量等质量控制。
12. Use fast, medium, or high quality controls for volume projections and 3D plots.

## 大文件策略 / Large File Policy

大型样例数据不应直接提交到 Git。建议把本地数据放在 `local_data/` 下，或者作为 GitHub Release 资产发布。

Large sample data should not be committed to git. Keep local-only data under `local_data/` or publish it as a GitHub Release asset.

仓库中的 `test_data/` 只保留紧凑的 `engcase_` 工程测试套件和确定性生成脚本。

The repository's `test_data/` folder keeps only the compact `engcase_` engineering suite and deterministic generators.

## 数据格式说明 / Data Format Notes

- `.npy` 适合较大的本地数组，因为它支持内存映射。
- `.npy` is suitable for large local arrays because it supports memory mapping.
- `.npz` 适合打包演示数据和多 key 切换测试，但较大的 key 可能需要解压到内存。
- `.npz` is good for bundled demos and key-switching tests, but very large keys may need to be decompressed into memory.
- `.zarr` 适合分块或分组数组。Zarr 数组会包装为 Dask 数组，使预览、采样统计和投影保持有限规模。
- `.zarr` is best when arrays are chunked or grouped. Zarr arrays are wrapped in Dask so previews, sampled statistics, and projections can stay bounded.
- CSV 导出主要面向当前 1D/2D 视图；高维数据应先筛选、切片或投影。
- CSV export is intended for current 1D/2D views; high-dimensional data should be filtered, sliced, or projected first.

## Benchmark / Benchmarking

生成性能实验材料：

Generate performance-report material with:

```powershell
python scripts/benchmark_large_arrays.py --shape 96x128x80
```

脚本会在 `benchmark_results/` 下写入 CSV 和 Markdown 结果，该目录已被 Git 忽略。

The script writes CSV and Markdown results under `benchmark_results/`, which is ignored by git.

## 常见场景 / Common Scenarios

- `601 x 951 x 288` 体数据：作为 `.npy` 打开，先查看中间切片预览，再使用一键二维化或沿轴 2 切片后绘图。
- `601 x 951 x 288` cube: open as `.npy`, inspect the middle-slice preview, then use quick 2D view or slice axis 2 before plotting.
- `(N, C)` 表格：先保留有意义的列；如果第 0 列是深度、时间或其他单调坐标，可作为 X 轴。
- `(N, C)` table: keep meaningful columns; use column 0 as X if it is depth, time, or another monotonic coordinate.
- 4D 属性体：先选择通道轴，再对选中的 3D 体使用切片或投影控制。
- 4D attribute volume: choose the channel axis first, then slice or project the selected 3D volume.
- 3D 工业 CT：优先使用切片或最大值投影；`3D体素图` 会自动降采样和限制体素数量。
- 3D industrial CT: prefer slices or max projection; `3D体素图` automatically downsamples and limits voxel count.

## 故障排查 / Troubleshooting

- 打开大体数据后如果界面较慢，等待后台预览/统计任务完成，或点击取消任务。
- If the UI feels slow after opening a large volume, wait for the background preview/statistics task to complete or cancel it.
- 如果图表不可读或绘制缓慢，先用切片、筛选或较低绘图质量缩小当前视图。
- If a plot is unreadable or slow, first reduce the current view with slicing, filtering, or lower plot quality.
- 如果文件太大，不要提交到 GitHub；应在本地重新生成，或作为 Release 资产发布。
- If a file is too large for GitHub, do not commit it. Regenerate it locally or publish it as a release artifact.
