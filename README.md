# NPY/NPZ Viewer

NPY/NPZ Viewer 是一个基于 PySide6 的桌面端多维数组查看与可视分析工具，支持 NumPy `.npy` / `.npz` 文件和本地 `.zarr` 目录。项目面向 1D 信号、2D 表格、2D 图像、3D 体数据、多通道数组和 4D 工程场数据，并加入 Dask/Zarr 大数组延迟计算能力。

NPY/NPZ Viewer is a PySide6 desktop tool for browsing and visualizing multidimensional array data. It supports NumPy `.npy` / `.npz` files and local `.zarr` stores, with Dask/Zarr-backed lazy computation for larger arrays.

## 快速开始 / Quick Start

```bash
conda create -n npy-viewer python=3.11 -y
conda activate npy-viewer
pip install -e ".[dev]"
python main.py
```

安装后也可以使用命令行入口启动：

After installation, the console entry point is also available:

```bash
npy-npz-viewer
```

启动后可以通过左侧 `打开文件` 打开 `.npy` / `.npz` 文件，也可以将文件拖入窗口；本地 `.zarr` 目录请使用 `打开 Zarr 目录`。

Open `.npy` / `.npz` files from the left panel, drag files into the window, or use `Open Zarr Directory` for local `.zarr` stores.

## 主要功能 / Main Features

- 标准 `src/npy_npz_viewer` Python 包结构，并提供 `npy-npz-viewer` 命令行入口。
  Standard `src/npy_npz_viewer` package layout with a console script.
- `.npy` 文件使用只读内存映射加载，适合较大的本地数组。
  `.npy` files are loaded through read-only memory mapping.
- `.npz` 多 key 文件和 `.zarr` group 数组共用同一套 key 选择界面。
  `.npz` keys and `.zarr` group arrays are browsed through a unified key-selection UI.
- 大 `.npy` 文件和 `.zarr` 数组可使用 Dask lazy array，避免不必要的全量加载。
  Large `.npy` files and `.zarr` arrays can use Dask-backed lazy arrays.
- 预览、采样统计、投影和绘图输入通过有限规模计算路径生成，减少 UI 阻塞。
  Preview, sampled statistics, projections, and plot inputs use bounded compute helpers.
- 支持非破坏式维度筛选、通用切片、CSV 导出和高维数组快速二维化。
  Supports non-destructive dimension filtering, universal slicing, CSV export, and quick 2D views.
- 支持按语义绘图：1D 折线/直方图、2D 表格图、2D 热力图、3D 切片/投影/体素图、4D 通道切片等。
  Provides semantic plotting for 1D sequences, 2D tables/images, 3D volumes, multichannel arrays, and 4D channel data.
- 提供 benchmark 脚本，可生成性能实验所需的 CSV/Markdown 结果。
  Includes a benchmark script that writes CSV/Markdown results for performance reports.

## 项目结构 / Project Layout

```text
npy-npz-Viewer/
|-- main.py                       # 本地启动入口 / Local launcher
|-- pyproject.toml                # 包元数据和依赖 / Package metadata and dependencies
|-- requirements.txt              # 兼容安装入口 / Compatibility install entry
|-- src/npy_npz_viewer/
|   |-- app.py                    # PySide6 主应用 / Main PySide6 application
|   |-- config.py                 # 运行时阈值配置 / Runtime thresholds
|   |-- logging_config.py         # 日志配置 / Logging setup
|   |-- core/                     # 加载、计算、切片、统计、绘图 / Core logic
|   |-- ui/                       # PySide6 控件和面板 / UI widgets and panels
|   `-- utils/                    # 预览和大数据辅助函数 / Utility helpers
|-- tests/                        # pytest 测试 / Pytest tests
|-- scripts/benchmark_large_arrays.py
|-- test_data/                    # engcase 工程测试数据 / Engineering test data
`-- docs/                         # 测试、性能和实验文档 / Guides and report docs
```

`main.py` 是唯一的本地启动脚本。新代码应从 `npy_npz_viewer` 包内导入。

`main.py` is the single local launcher. New code should import from the `npy_npz_viewer` package.

## 大数据说明 / Large Data Notes

- `.npy` 适合较大的本地数组，因为它支持内存映射。
  `.npy` remains suitable for large local arrays because it supports memory mapping.
- `.zarr` 适合分块多维数据，当前通过 Dask lazy array 读取。
  `.zarr` is supported for chunked multidimensional stores and is loaded lazily through Dask.
- 对高维数据建议先做维度筛选、切片、投影或一键二维化，再进行昂贵的 3D 绘图。
  For high-dimensional data, use filtering, slicing, projection, or quick 2D view before expensive 3D plots.
- 统计信息会对大数组使用采样策略。
  Statistics use bounded sampling for large arrays.
- 3D 绘图包含降采样和默认体素数量限制，避免界面长时间卡顿。
  3D plots use downsampling and bounded voxel rendering to keep the UI responsive.
- CSV 导出主要面向当前 1D/2D 视图，不建议直接导出原始高维体数据。
  CSV export is intended for current 1D/2D views, not raw high-dimensional volumes.

## 验证 / Verification

```bash
python -m compileall -q main.py src test_data tests scripts
pytest
python test_data/verify_functions.py
python test_data/create_engineering_test_data.py
python test_data/verify_engineering_test_data.py
```

手动测试步骤见：

Manual click-by-click validation is documented in:

```text
docs/TESTING_GUIDE.md
```

运行一个小型 benchmark：

Run a small benchmark smoke test:

```bash
python scripts/benchmark_large_arrays.py --shape 96x128x80
```

benchmark 输出会写入 `benchmark_results/`，该目录已被 `.gitignore` 忽略。

Benchmark output is written under `benchmark_results/`, which is ignored by git.

## 测试数据 / Test Data

仓库提交了一组紧凑的 `engcase_` 工程测试数据，用于课程展示、功能验证和手动测试。

The repository tracks a compact `engcase_` engineering test suite for demos, validation, and manual testing.

| 文件 / File | 场景 / Scenario | 推荐首图 / First Useful View |
| --- | --- | --- |
| `test_data/engcase_bearing_vibration_4096.npy` | 1D 轴承振动故障诊断 / 1D bearing vibration diagnosis | 折线图 / Line chart |
| `test_data/engcase_bridge_sensor_table_2400x6.npy` | 2D 桥梁结构健康监测表 / 2D bridge structural health table | 多折线图或相关性热力图 / Multi-line chart or correlation heatmap |
| `test_data/engcase_fem_stress_plate_512x512.npy` | 2D 有限元应力云图 / 2D finite-element stress field | 热力图 / Heatmap |
| `test_data/engcase_industrial_ct_volume_96x128x96.npy` | 3D 工业 CT 缺陷检测 / 3D industrial CT inspection | 切片热力图或投影图 / Slice heatmap or projection |
| `test_data/engcase_cfd_wake_24x64x48x4.npy` | 4D CFD 尾流场 / 4D CFD wake field | 通道切片热力图 / Channel slice heatmap |
| `test_data/engcase_mixed_suite.npz` | 混合工程 NPZ 套件 / Mixed engineering NPZ suite | 切换 key / Switch each key |

`test_data/engcase_manifest.json` 记录 shape、语义、推荐图表和预期可见特征。

`test_data/engcase_manifest.json` records shapes, semantics, recommended plots, and expected visible features.

重新生成或验证测试数据：

Regenerate or verify the test data:

```bash
python test_data/create_engineering_test_data.py
python test_data/verify_engineering_test_data.py
```

任意大型 `.npy/.npz/.zarr` 数据不应直接提交到 Git。建议放入 `local_data/`，或作为 GitHub Release 资产发布。

Arbitrary large `.npy/.npz/.zarr` data should not be committed to normal git history. Put very large local samples under `local_data/` or publish them as GitHub Release assets.

## 许可证 / License

MIT. See [LICENSE](LICENSE).
