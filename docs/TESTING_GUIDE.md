# 工程测试数据指南 / Engineering Test Data Guide

本指南只使用 `engcase_` 工程测试数据，用来验证 NPY/NPZ Viewer 的数据加载、语义判断、预览、统计和可视化能力。

This guide uses only the `engcase_` engineering test suite to validate loading, semantic inference, preview, statistics, and visualization behavior in NPY/NPZ Viewer.

## 数据位置 / Data Location

工程测试数据位于项目目录：

The engineering test data lives under:

```text
test_data/
```

| 文件 / File | 维度 / Dimension | shape | 工程场景 / Scenario | 推荐第一张图 / First View |
| --- | --- | --- | --- | --- |
| `engcase_bearing_vibration_4096.npy` | 1D | `(4096,)` | 轴承振动故障诊断 / Bearing vibration diagnosis | 折线图 / Line chart |
| `engcase_bridge_sensor_table_2400x6.npy` | 2D 表格 / 2D table | `(2400, 6)` | 桥梁结构健康监测 / Bridge structural health monitoring | 多折线图、相关性热力图 / Multi-line chart, correlation heatmap |
| `engcase_fem_stress_plate_512x512.npy` | 2D 图像 / 2D image | `(512, 512)` | 有限元板件应力云图 / FEM stress field | 热力图 / Heatmap |
| `engcase_industrial_ct_volume_96x128x96.npy` | 3D | `(96, 128, 96)` | 工业 CT 缺陷检测 / Industrial CT inspection | 切片热力图、投影图、3D体素图 / Slice, projection, voxel |
| `engcase_cfd_wake_24x64x48x4.npy` | 4D | `(24, 64, 48, 4)` | CFD 尾流场 / CFD wake field | 通道切片热力图 / Channel slice heatmap |
| `engcase_suite.npz` | NPZ 多数组 / Multi-array NPZ | 多 key，完整数据 / Multiple keys, full data | 与 NPY 基准一致 / Equivalent to canonical NPY files | 切换每个 key / Switch each key |
| `engcase_suite.zarr/` | Zarr group | 多 key，分块存储 / Multiple keys, chunked storage | 与 NPY 基准一致 / Equivalent to canonical NPY files | 打开 Zarr 目录并切换 key / Open Zarr directory and switch keys |

`engcase_manifest.json` 记录了每个数据集的说明、shape、推荐图表和预期可见特征。

`engcase_manifest.json` records each dataset's description, shape, recommended plots, and expected visible features.

## 生成与自动验证 / Generation And Automated Checks

这组数据会随项目提交。如果文件缺失，或想重新生成一份确定性数据，在项目根目录运行：

The data is tracked in the repository. If files are missing or you want to regenerate the deterministic suite, run from the repository root:

```powershell
python test_data\manage_test_data.py generate
python test_data\manage_test_data.py verify-all
```

验证成功时最后会看到：

A successful validation ends with:

```text
All engcase NPY/NPZ/Zarr data checks passed.
All core checks passed.
```

发布或提交前建议运行：

Before publishing or committing, run:

```powershell
python -m compileall -q main.py src test_data tests scripts
pytest
python test_data\manage_test_data.py generate
python test_data\manage_test_data.py verify-all
```

## 启动软件 / Launch The Application

在项目根目录运行：

Run from the repository root:

```powershell
python main.py
```

安装项目后也可以运行：

After installing the project, you can also run:

```powershell
npy-npz-viewer
```

打开数据有三种方式：

There are three ways to open data:

- 点击左侧 `打开文件`，选择 `test_data` 下的 `.npy` 或 `.npz` 文件。
- Click `打开文件` on the left panel and choose a `.npy` or `.npz` file under `test_data`.
- 点击左侧 `打开 Zarr 目录`，选择 `test_data/engcase_suite.zarr`。
- Click `打开 Zarr 目录` and choose `test_data/engcase_suite.zarr`.
- 直接把文件拖进窗口。
- Drag the file directly into the window.

## 逐项手动验证 / Manual Test Matrix

### 1D 轴承振动 / 1D Bearing Vibration

打开：

Open:

```text
test_data/engcase_bearing_vibration_4096.npy
```

操作：

Steps:

1. 在左侧 `数据解释方式` 中使用 `自动判断`，或手动选择 `一维序列`。
2. In `数据解释方式`, keep `自动判断` or manually choose `一维序列`.
3. 在 `图表类型` 中选择 `折线图`。
4. Select `折线图` in `图表类型`.
5. 点击 `生成图表`。
6. Click `生成图表`.
7. 再切换到 `直方图`，点击 `生成图表`。
8. Switch to `直方图` and click `生成图表` again.

预期现象：

Expected result:

- 折线图中能看到周期性冲击峰。
- The line chart shows periodic impact peaks.
- 冲击峰后带有高频衰减振动。
- Each impact is followed by high-frequency decaying vibration.
- 直方图中应有明显长尾，体现冲击型信号特征。
- The histogram has a visible long tail, matching impulsive vibration data.

### 2D 桥梁结构健康监测表 / 2D Bridge Monitoring Table

打开：

Open:

```text
test_data/engcase_bridge_sensor_table_2400x6.npy
```

列含义：

Column meanings:

| 列 / Column | 含义 / Meaning |
| --- | --- |
| 列 0 / Column 0 | 时间，单位分钟 / Time in minutes |
| 列 1 / Column 1 | 车辆荷载 / Vehicle load |
| 列 2 / Column 2 | 应变 / Strain |
| 列 3 / Column 3 | 温度 / Temperature |
| 列 4 / Column 4 | 挠度 / Deflection |
| 列 5 / Column 5 | 应力 / Stress |

多折线图验证：

Multi-line chart test:

1. 在 `数据解释方式` 中选择 `二维表格`。
2. Choose `二维表格` in `数据解释方式`.
3. `图表类型` 选择 `多折线图`。
4. Set `图表类型` to `多折线图`.
5. `X 轴列` 选择 `列 0`。
6. Set `X 轴列` to `列 0`.
7. 在 `Y 轴列（可多选）` 中选择 `列 1`、`列 2`、`列 4`、`列 5`。
8. In `Y 轴列（可多选）`, select `列 1`, `列 2`, `列 4`, and `列 5`.
9. 点击 `生成图表`。
10. Click `生成图表`.

预期现象：

Expected result:

- 荷载峰值对应车辆通过桥梁。
- Load peaks correspond to vehicles crossing the bridge.
- 应变、挠度、应力会随荷载同步变化。
- Strain, deflection, and stress change together with load.
- 温度列不选时不应出现在多折线图中。
- The temperature column should not appear when it is not selected.

相关性热力图回归验证：

Correlation heatmap regression test:

1. `图表类型` 选择 `相关性热力图`。
2. Set `图表类型` to `相关性热力图`.
3. 在 `Y 轴列（可多选）` 中只选择 `列 1`、`列 2`、`列 3`、`列 4`、`列 5`。
4. In `Y 轴列（可多选）`, select only `列 1`, `列 2`, `列 3`, `列 4`, and `列 5`.
5. 不选择 `列 0`。
6. Do not select `列 0`.
7. 点击 `生成图表`。
8. Click `生成图表`.

预期现象：

Expected result:

- 热力图坐标轴只显示 `列 1` 到 `列 5`。
- The heatmap axes show only `列 1` through `列 5`.
- 不应出现 `列 0`。
- `列 0` must not appear.
- 荷载、应变、挠度、应力之间相关性明显。
- Load, strain, deflection, and stress should show strong correlations.

### 2D 有限元应力云图 / 2D FEM Stress Field

打开：

Open:

```text
test_data/engcase_fem_stress_plate_512x512.npy
```

操作：

Steps:

1. 在 `数据解释方式` 中选择 `二维图像/矩阵`。
2. Choose `二维图像/矩阵` in `数据解释方式`.
3. `图表类型` 选择 `热力图`。
4. Set `图表类型` to `热力图`.
5. `色图` 可以选择 `magma`、`viridis` 或 `gray`。
6. Choose a colormap such as `magma`, `viridis`, or `gray`.
7. 点击 `生成图表`。
8. Click `生成图表`.

预期现象：

Expected result:

- 图中有一个低值圆孔。
- The image contains a low-value circular hole.
- 圆孔周围出现明显应力集中环。
- A bright stress concentration ring appears around the hole.
- 圆孔右侧有裂纹尖端热点。
- A crack-tip hotspot appears to the right of the hole.
- 左侧上下支撑区域有局部热点。
- Local support hotspots appear on the upper-left and lower-left areas.

### 3D 工业 CT 缺陷检测 / 3D Industrial CT Inspection

打开：

Open:

```text
test_data/engcase_industrial_ct_volume_96x128x96.npy
```

切片验证：

Slice test:

1. 在 `数据解释方式` 中选择 `三维体数据`。
2. Choose `三维体数据` in `数据解释方式`.
3. `图表类型` 选择 `切片热力图`。
4. Set `图表类型` to `切片热力图`.
5. `切片轴` 选择任意轴。
6. Choose any `切片轴`.
7. 将 `切片索引` 调到中间附近。
8. Move `切片索引` near the middle.
9. 点击 `生成图表`。
10. Click `生成图表`.

预期现象：

Expected result:

- 能看到铸件截面。
- A casting cross-section is visible.
- 截面内有低密度气孔。
- Low-density pores are visible inside the part.
- 中部附近有细裂纹。
- A thin crack is visible near the center.
- 局部有亮色高密度夹杂。
- Bright high-density inclusions appear locally.

投影验证：

Projection test:

1. `图表类型` 选择 `投影图`。
2. Set `图表类型` to `投影图`.
3. `投影方式` 选择 `max`。
4. Set `投影方式` to `max`.
5. `预览质量` 先选择 `快速`。
6. Set `预览质量` to `快速` first.
7. 点击 `生成图表`。
8. Click `生成图表`.

预期现象：

Expected result:

- 最大值投影更容易看到亮色夹杂。
- Max projection makes bright inclusions easier to see.
- 快速质量应能较快生成。
- Fast quality should render quickly.

3D 体素图回归验证：

3D voxel regression test:

1. `图表类型` 选择 `3D体素图`。
2. Set `图表类型` to `3D体素图`.
3. 点击 `生成图表`。
4. Click `生成图表`.

预期现象：

Expected result:

- 界面应能恢复响应，不应长时间卡死。
- The UI should become responsive again and should not freeze for a long time.
- 图表下方应显示体素图降采样或体素数量限制提示。
- A downsampling or voxel-limit message should appear below the plot.
- 体素图主要展示高密度区域，不追求完整复原 CT 体数据。
- The voxel plot emphasizes high-density regions rather than reconstructing the full CT volume.

### 4D CFD 尾流场 / 4D CFD Wake Field

打开：

Open:

```text
test_data/engcase_cfd_wake_24x64x48x4.npy
```

通道含义：

Channel meanings:

| 通道 / Channel | 含义 / Meaning |
| --- | --- |
| 通道 0 / Channel 0 | 速度 / Velocity |
| 通道 1 / Channel 1 | 压力 / Pressure |
| 通道 2 / Channel 2 | 涡量 / Vorticity |
| 通道 3 / Channel 3 | 温度 / Temperature |

操作：

Steps:

1. 在 `数据解释方式` 中选择 `四维体数据`。
2. Choose `四维体数据` in `数据解释方式`.
3. `通道轴` 选择 `轴 3`。
4. Set `通道轴` to `轴 3`.
5. `通道索引` 分别选择 `0`、`1`、`2`、`3`。
6. Try `通道索引` values `0`, `1`, `2`, and `3`.
7. `图表类型` 选择 `切片热力图` 或 `投影图`。
8. Set `图表类型` to `切片热力图` or `投影图`.
9. 点击 `生成图表`。
10. Click `生成图表`.

预期现象：

Expected result:

- 速度通道中有尾流低速区。
- The velocity channel shows a wake velocity deficit.
- 压力通道中有压力恢复区域。
- The pressure channel shows pressure recovery.
- 涡量通道中有正负剪切层。
- The vorticity channel shows positive and negative shear layers.
- 温度通道中有热羽流结构。
- The temperature channel shows a thermal plume.

### NPZ 多 key 切换 / NPZ Key Switching

打开：

Open:

```text
test_data/engcase_suite.npz
```

操作：

Steps:

1. 在左侧 NPZ key 列表中依次点击每个 key。
2. Click each key in the left-side NPZ key list.
3. 观察文件信息、shape、预览表和自动判断结果。
4. Watch the file info, shape, preview table, and auto-inferred semantic result.
5. 对不同 key 分别生成对应图表。
6. Generate the corresponding plot for each key.

预期现象：

Expected result:

- 切换 key 后 shape 会变化。
- The shape changes after switching keys.
- 自动判断结果会跟随 key 的维度和结构变化。
- The auto-inferred semantic result changes with each key's dimensionality and structure.
- 图表参数区会刷新为当前 key 对应的数据类型。
- The plot parameter panel refreshes for the current key's data type.

### Zarr group key 切换 / Zarr Group Key Switching

打开：

Open:

```text
test_data/engcase_suite.zarr
```

步骤：

Steps:

1. 点击左侧 `打开 Zarr 目录`，选择 `engcase_suite.zarr` 目录。
2. Click `打开 Zarr 目录` and choose the `engcase_suite.zarr` directory.
3. 在 key 列表中依次切换 `bearing_vibration`、`bridge_sensor_table`、`fem_stress_plate`、`industrial_ct_volume`、`cfd_wake`。
4. Switch through `bearing_vibration`, `bridge_sensor_table`, `fem_stress_plate`, `industrial_ct_volume`, and `cfd_wake`.
5. 对同一个 key，与 `.npy` 和 `.npz` 打开结果对比预览、统计和图表。
6. For the same key, compare preview, statistics, and plots with the `.npy` and `.npz` versions.

预期结果：

Expected result:

- Zarr key 的 shape 与对应 NPY 文件完全一致。
- Each Zarr key has the same shape as the matching NPY file.
- 预览、统计和图表结果与 NPY/NPZ 版本一致。
- Preview, statistics, and plots match the NPY/NPZ versions.
- Zarr 数组显示为分块/lazy 路径，适合验证大数组加载能力。
- Zarr arrays use the chunked/lazy path, which is useful for large-array loading checks.

## 验收重点 / Acceptance Focus

- `engcase_bridge_sensor_table_2400x6.npy` 的相关性热力图必须遵循 `Y 轴列（可多选）`。
- The correlation heatmap for `engcase_bridge_sensor_table_2400x6.npy` must respect `Y 轴列（可多选）`.
- `engcase_industrial_ct_volume_96x128x96.npy` 的 `3D体素图` 必须有默认性能保护。
- The `3D体素图` for `engcase_industrial_ct_volume_96x128x96.npy` must use default performance protection.
- `engcase_suite.npz` 和 `engcase_suite.zarr/` 必须能正常切换 key。
- `engcase_suite.npz` and `engcase_suite.zarr/` must support normal key switching.
- 所有图表生成后，窗口仍可继续操作、切换参数和打开其他文件。
- After any plot is generated, the window should remain usable for parameter changes and opening other files.
