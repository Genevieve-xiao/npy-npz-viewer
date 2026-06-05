# Engineering Test Data Guide

本指南只使用 `engcase_` 工程测试数据，用来验证 NPY/NPZ Viewer 的数据加载、语义判断、预览、统计和可视化能力。

## 数据位置

工程测试数据位于项目目录：

```text
test_data/
```

文件清单：

| 文件 | 维度 | shape | 工程场景 | 推荐第一张图 |
| --- | --- | --- | --- | --- |
| `engcase_bearing_vibration_4096.npy` | 1D | `(4096,)` | 轴承振动故障诊断 | 折线图 |
| `engcase_bridge_sensor_table_2400x6.npy` | 2D 表格 | `(2400, 6)` | 桥梁结构健康监测 | 多折线图、相关性热力图 |
| `engcase_fem_stress_plate_512x512.npy` | 2D 图像 | `(512, 512)` | 有限元板件应力云图 | 热力图 |
| `engcase_industrial_ct_volume_96x128x96.npy` | 3D | `(96, 128, 96)` | 工业 CT 缺陷检测 | 切片热力图、投影图、3D体素图 |
| `engcase_cfd_wake_24x64x48x4.npy` | 4D | `(24, 64, 48, 4)` | CFD 尾流场 | 通道切片热力图 |
| `engcase_mixed_suite.npz` | NPZ 多数组 | 多 key | 混合工程场景 | 切换每个 key |

`engcase_manifest.json` 记录了每个数据集的说明、shape、推荐图表和预期可见特征。

## 生成与自动验证

这组数据会随项目提交。如果文件缺失，或想重新生成一份确定性数据，在项目根目录运行：

```powershell
python test_data\create_engineering_test_data.py
python test_data\verify_engineering_test_data.py
```

验证成功时最后会看到：

```text
All engcase engineering data checks passed.
```

发布前建议同时运行：

```powershell
python -m compileall -q main.py src test_data tests scripts
pytest
python test_data\verify_functions.py
python test_data\verify_engineering_test_data.py
```

## 启动软件

在项目根目录运行：

```powershell
python main.py
```

也可以在安装项目后运行：

```powershell
npy-npz-viewer
```

打开数据有两种方式：

- 点击左侧 `打开文件`，选择 `test_data` 下的 `.npy` 或 `.npz` 文件。
- 直接把文件拖进窗口。

## 逐项手动验证

### 1D 轴承振动

打开：

```text
test_data/engcase_bearing_vibration_4096.npy
```

操作：

1. 在左侧 `数据解释方式` 中使用 `自动判断`，或手动选择 `一维序列`。
2. 在 `图表类型` 中选择 `折线图`。
3. 点击 `生成图表`。
4. 再切换到 `直方图`，点击 `生成图表`。

预期现象：

- 折线图中能看到周期性冲击峰。
- 冲击峰后带有高频衰减振动。
- 直方图中应有明显长尾，体现冲击型信号特征。

### 2D 桥梁结构健康监测表

打开：

```text
test_data/engcase_bridge_sensor_table_2400x6.npy
```

列含义：

| 列 | 含义 |
| --- | --- |
| 列 0 | 时间，单位分钟 |
| 列 1 | 车辆荷载 |
| 列 2 | 应变 |
| 列 3 | 温度 |
| 列 4 | 挠度 |
| 列 5 | 应力 |

多折线图验证：

1. 在 `数据解释方式` 中选择 `二维表格`。
2. `图表类型` 选择 `多折线图`。
3. `X 轴列` 选择 `列 0`。
4. 在 `Y 轴列（可多选）` 中选择 `列 1`、`列 2`、`列 4`、`列 5`。
5. 点击 `生成图表`。

预期现象：

- 荷载峰值对应车辆通过桥梁。
- 应变、挠度、应力会随荷载同步变化。
- 温度列不选时不应出现在多折线图中。

相关性热力图回归验证：

1. `图表类型` 选择 `相关性热力图`。
2. 在 `Y 轴列（可多选）` 中只选择 `列 1`、`列 2`、`列 3`、`列 4`、`列 5`。
3. 不选择 `列 0`。
4. 点击 `生成图表`。

预期现象：

- 热力图坐标轴只显示 `列 1` 到 `列 5`。
- 不应出现 `列 0`。
- 荷载、应变、挠度、应力之间相关性明显。

### 2D 有限元应力云图

打开：

```text
test_data/engcase_fem_stress_plate_512x512.npy
```

操作：

1. 在 `数据解释方式` 中选择 `二维图像/矩阵`。
2. `图表类型` 选择 `热力图`。
3. `色图` 可以选择 `magma`、`viridis` 或 `gray`。
4. 点击 `生成图表`。

预期现象：

- 图中有一个低值圆孔。
- 圆孔周围出现明显应力集中环。
- 圆孔右侧有裂纹尖端热点。
- 左侧上下支撑区域有局部热点。

### 3D 工业 CT 缺陷检测

打开：

```text
test_data/engcase_industrial_ct_volume_96x128x96.npy
```

切片验证：

1. 在 `数据解释方式` 中选择 `三维体数据`。
2. `图表类型` 选择 `切片热力图`。
3. `切片轴` 选择任意轴。
4. 将 `切片索引` 调到中间附近。
5. 点击 `生成图表`。

预期现象：

- 能看到铸件截面。
- 截面内有低密度气孔。
- 中部附近有细裂纹。
- 局部有亮色高密度夹杂。

投影验证：

1. `图表类型` 选择 `投影图`。
2. `投影方式` 选择 `max`。
3. `预览质量` 先选择 `快速`。
4. 点击 `生成图表`。

预期现象：

- 最大值投影更容易看到亮色夹杂。
- 快速质量应能较快生成。

3D 体素图回归验证：

1. `图表类型` 选择 `3D体素图`。
2. 点击 `生成图表`。

预期现象：

- 界面应能恢复响应，不应长时间卡死。
- 图表下方应显示体素图降采样或体素数量限制提示。
- 体素图主要展示高密度区域，不追求完整复原 CT 体数据。

### 4D CFD 尾流场

打开：

```text
test_data/engcase_cfd_wake_24x64x48x4.npy
```

通道含义：

| 通道 | 含义 |
| --- | --- |
| 通道 0 | 速度 |
| 通道 1 | 压力 |
| 通道 2 | 涡量 |
| 通道 3 | 温度 |

操作：

1. 在 `数据解释方式` 中选择 `四维体数据`。
2. `通道轴` 选择 `轴 3`。
3. `通道索引` 分别选择 `0`、`1`、`2`、`3`。
4. `图表类型` 选择 `切片热力图` 或 `投影图`。
5. 点击 `生成图表`。

预期现象：

- 速度通道中有尾流低速区。
- 压力通道中有压力恢复区域。
- 涡量通道中有正负剪切层。
- 温度通道中有热羽流结构。

### NPZ 多 key 切换

打开：

```text
test_data/engcase_mixed_suite.npz
```

操作：

1. 在左侧 NPZ key 列表中依次点击每个 key。
2. 观察文件信息、shape、预览表和自动判断结果。
3. 对不同 key 分别生成对应图表。

预期现象：

- 切换 key 后 shape 会变化。
- 自动判断结果会跟随 key 的维度和结构变化。
- 图表参数区会刷新为当前 key 对应的数据类型。

## 验收重点

- `engcase_bridge_sensor_table_2400x6.npy` 的相关性热力图必须遵循 `Y 轴列（可多选）`。
- `engcase_industrial_ct_volume_96x128x96.npy` 的 `3D体素图` 必须有默认性能保护。
- `engcase_mixed_suite.npz` 必须能正常切换 key。
- 所有图表生成后，窗口仍可继续操作、切换参数和打开其他文件。
