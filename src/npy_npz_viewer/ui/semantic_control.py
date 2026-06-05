"""
语义控制组件
提供数据语义选择和可视化参数配置
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QPushButton, QGroupBox, QSpinBox,
    QCheckBox, QListWidget, QSlider
)
from PySide6.QtCore import Signal, Qt
from npy_npz_viewer.core.data_semantics import DataSemantics, get_plot_options


class SemanticControlWidget(QWidget):
    """语义控制组件"""

    semantic_changed = Signal(str)  # 发送语义类型
    plot_requested = Signal(dict)   # 发送绘图参数

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_shape = None
        self.current_semantic = None
        self.inferred_semantic = None
        self.inferred_info = {}
        self.volume_shape = None
        self.quick_2d_btn = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # 数据语义选择
        semantic_group = QGroupBox("数据解释方式")
        semantic_layout = QVBoxLayout()

        self.semantic_combo = QComboBox()
        self.semantic_combo.addItem("自动判断")
        self.semantic_combo.addItems(DataSemantics.get_all_types())
        self.semantic_combo.currentTextChanged.connect(self.on_semantic_changed)
        semantic_layout.addWidget(self.semantic_combo)

        self.semantic_info_label = QLabel("")
        self.semantic_info_label.setWordWrap(True)
        self.semantic_info_label.setStyleSheet("color: #666; font-size: 11px;")
        semantic_layout.addWidget(self.semantic_info_label)

        semantic_group.setLayout(semantic_layout)
        layout.addWidget(semantic_group)

        # 可视化参数区域（动态内容）
        self.params_group = QGroupBox("可视化参数")
        self.params_layout = QVBoxLayout()
        self.params_group.setLayout(self.params_layout)
        layout.addWidget(self.params_group)

        # 图表类型选择
        plot_group = QGroupBox("图表类型")
        plot_layout = QVBoxLayout()

        self.plot_type_combo = QComboBox()
        plot_layout.addWidget(self.plot_type_combo)

        self.plot_btn = QPushButton("生成图表")
        self.plot_btn.clicked.connect(self.on_plot_clicked)
        plot_layout.addWidget(self.plot_btn)

        plot_group.setLayout(plot_layout)
        layout.addWidget(plot_group)

        layout.addStretch()

    def set_array_info(self, shape: tuple, semantic_info: dict):
        """设置数组信息和语义推断结果"""
        self.current_shape = shape
        self.inferred_info = semantic_info or {}
        self.inferred_semantic = self.inferred_info.get('semantic', DataSemantics.UNKNOWN)

        # 新数组/新视图默认回到自动判断，但实际生效语义使用推断结果。
        self.semantic_combo.blockSignals(True)
        self.semantic_combo.setCurrentText("自动判断")
        self.semantic_combo.blockSignals(False)
        self.apply_auto_semantic(emit=False)

    def on_semantic_changed(self, semantic_text: str):
        """语义类型改变"""
        if semantic_text == "自动判断":
            self.apply_auto_semantic(emit=True)
            return

        semantic = DataSemantics.from_string(semantic_text)
        self.apply_semantic(semantic)
        self.update_semantic_info_label(auto_mode=False, selected_semantic=semantic)
        self.semantic_changed.emit(semantic_text)

    def apply_auto_semantic(self, emit: bool = False):
        """让自动判断结果成为当前生效语义。"""
        semantic = self.inferred_semantic or DataSemantics.UNKNOWN
        self.apply_semantic(semantic)
        self.update_semantic_info_label(auto_mode=True, selected_semantic=semantic)
        if emit:
            self.semantic_changed.emit("自动判断")

    def apply_semantic(self, semantic: DataSemantics):
        """应用当前生效语义并刷新图表选项和参数区。"""
        self.current_semantic = semantic

        plot_options = get_plot_options(semantic)
        self.plot_type_combo.clear()
        self.plot_type_combo.addItems(plot_options)

        self.update_params_ui(semantic)

    def update_semantic_info_label(self, auto_mode: bool, selected_semantic: DataSemantics):
        """显示自动判断结果和当前生效解释方式。"""
        reason = self.inferred_info.get('reason', '')
        confidence = self.inferred_info.get('confidence', '')
        inferred = self.inferred_semantic or DataSemantics.UNKNOWN
        suggestions = self.inferred_info.get('suggestions') or []

        lines = [
            f"自动判断结果: {inferred.value}",
            f"原因: {reason}" if reason else "原因: 暂无",
            f"置信度: {confidence}" if confidence else "置信度: 暂无",
        ]
        if suggestions:
            lines.append("其他可能: " + ", ".join(s.value for s in suggestions))
        if not auto_mode:
            lines.insert(0, f"当前手动选择: {selected_semantic.value}")
        self.semantic_info_label.setText("\n".join(lines))

    def update_params_ui(self, semantic: DataSemantics):
        """根据语义类型更新参数 UI"""
        self.clear_layout(self.params_layout)
        self.volume_shape = None

        if semantic == DataSemantics.TABULAR_2D:
            self.create_tabular_params()
        elif semantic == DataSemantics.IMAGE_2D:
            self.create_image_params()
        elif semantic == DataSemantics.VOLUME_3D:
            self.create_volume_params()
        elif semantic == DataSemantics.VOLUME_4D:
            self.create_volume_4d_params()
        elif semantic == DataSemantics.MULTICHANNEL_3D:
            self.create_multichannel_params()

        self.create_common_workflow_params(semantic)

    def clear_layout(self, layout):
        """递归清空动态参数布局。"""
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            child_layout = item.layout()
            if widget:
                widget.deleteLater()
            elif child_layout:
                self.clear_layout(child_layout)
                child_layout.deleteLater()

    def create_tabular_params(self):
        """创建表格数据参数控件"""
        if self.current_shape is None or len(self.current_shape) != 2:
            return

        rows, cols = self.current_shape

        # X 轴列选择
        x_layout = QHBoxLayout()
        x_layout.addWidget(QLabel("X 轴列:"))
        self.x_col_combo = QComboBox()
        self.x_col_combo.addItem("行索引")
        for i in range(cols):
            self.x_col_combo.addItem(f"列 {i}")
        x_layout.addWidget(self.x_col_combo)
        self.params_layout.addLayout(x_layout)

        # Y 轴列选择（多选）
        self.params_layout.addWidget(QLabel("Y 轴列（可多选）:"))
        self.y_cols_list = QListWidget()
        self.y_cols_list.setSelectionMode(QListWidget.MultiSelection)
        for i in range(cols):
            self.y_cols_list.addItem(f"列 {i}")
        self.y_cols_list.setMaximumHeight(100)
        self.params_layout.addWidget(self.y_cols_list)

        # 反转 Y 轴（用于深度数据）
        self.invert_y_check = QCheckBox("反转 Y 轴（深度数据）")
        self.params_layout.addWidget(self.invert_y_check)

        # 单列选择（用于直方图）
        col_layout = QHBoxLayout()
        col_layout.addWidget(QLabel("单列选择:"))
        self.single_col_spin = QSpinBox()
        self.single_col_spin.setRange(0, cols - 1)
        col_layout.addWidget(self.single_col_spin)
        self.params_layout.addLayout(col_layout)

        # 散点图列选择
        scatter_layout = QHBoxLayout()
        scatter_layout.addWidget(QLabel("散点图 X:"))
        self.scatter_x_spin = QSpinBox()
        self.scatter_x_spin.setRange(0, cols - 1)
        scatter_layout.addWidget(self.scatter_x_spin)
        scatter_layout.addWidget(QLabel("Y:"))
        self.scatter_y_spin = QSpinBox()
        self.scatter_y_spin.setRange(0, cols - 1)
        self.scatter_y_spin.setValue(min(1, cols - 1))
        scatter_layout.addWidget(self.scatter_y_spin)
        self.params_layout.addLayout(scatter_layout)

    def create_image_params(self):
        """创建二维图像/矩阵参数控件。"""
        if self.current_shape is None or len(self.current_shape) != 2:
            return

        rows, cols = self.current_shape

        self.params_layout.addWidget(QLabel(f"当前矩阵: 轴 0={rows} 行，轴 1={cols} 列"))

        axes_layout = QHBoxLayout()
        axes_layout.addWidget(QLabel("X/Y 显示:"))
        self.swap_image_axes_check = QCheckBox("交换 X 轴和 Y 轴")
        axes_layout.addWidget(self.swap_image_axes_check)
        self.params_layout.addLayout(axes_layout)

        x_order_layout = QHBoxLayout()
        x_order_layout.addWidget(QLabel("X 轴方向:"))
        self.image_x_order_combo = QComboBox()
        self.image_x_order_combo.addItem("升序", "asc")
        self.image_x_order_combo.addItem("降序", "desc")
        x_order_layout.addWidget(self.image_x_order_combo)
        self.params_layout.addLayout(x_order_layout)

        y_order_layout = QHBoxLayout()
        y_order_layout.addWidget(QLabel("Y 轴方向:"))
        self.image_y_order_combo = QComboBox()
        self.image_y_order_combo.addItem("升序", "asc")
        self.image_y_order_combo.addItem("降序", "desc")
        y_order_layout.addWidget(self.image_y_order_combo)
        self.params_layout.addLayout(y_order_layout)

        color_layout = QHBoxLayout()
        color_layout.addWidget(QLabel("色图:"))
        self.image_cmap_combo = QComboBox()
        self.image_cmap_combo.addItems(["viridis", "gray", "seismic", "coolwarm", "magma", "terrain"])
        color_layout.addWidget(self.image_cmap_combo)
        self.params_layout.addLayout(color_layout)

        aspect_layout = QHBoxLayout()
        aspect_layout.addWidget(QLabel("比例:"))
        self.image_aspect_combo = QComboBox()
        self.image_aspect_combo.addItem("自动", "auto")
        self.image_aspect_combo.addItem("等比例", "equal")
        aspect_layout.addWidget(self.image_aspect_combo)
        self.params_layout.addLayout(aspect_layout)

    def create_volume_params(self, volume_shape=None):
        """创建三维体数据参数控件"""
        shape = volume_shape if volume_shape is not None else self.current_shape
        if shape is None or len(shape) != 3:
            return
        self.volume_shape = shape

        # 切片轴选择
        axis_layout = QHBoxLayout()
        axis_layout.addWidget(QLabel("切片轴:"))
        self.slice_axis_combo = QComboBox()
        self.slice_axis_combo.addItems(["轴 0", "轴 1", "轴 2"])
        self.slice_axis_combo.currentIndexChanged.connect(self.on_slice_axis_changed)
        axis_layout.addWidget(self.slice_axis_combo)
        self.params_layout.addLayout(axis_layout)

        # 切片索引滑块
        self.params_layout.addWidget(QLabel("切片索引:"))
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(shape[0] - 1)
        self.slice_slider.setValue(shape[0] // 2)
        self.params_layout.addWidget(self.slice_slider)

        self.slice_index_label = QLabel(f"{shape[0] // 2} / {shape[0] - 1}")
        self.slice_slider.valueChanged.connect(
            lambda v: self.slice_index_label.setText(
                f"{v} / {self.slice_slider.maximum()}"
            )
        )
        self.params_layout.addWidget(self.slice_index_label)

        # 投影方式
        proj_layout = QHBoxLayout()
        proj_layout.addWidget(QLabel("投影方式:"))
        self.proj_method_combo = QComboBox()
        self.proj_method_combo.addItems(["无", "mean", "max", "min"])
        proj_layout.addWidget(self.proj_method_combo)
        self.params_layout.addLayout(proj_layout)

        quality_layout = QHBoxLayout()
        quality_layout.addWidget(QLabel("预览质量:"))
        self.volume_quality_combo = QComboBox()
        self.volume_quality_combo.addItem("快速", "fast")
        self.volume_quality_combo.addItem("较精细", "medium")
        self.volume_quality_combo.addItem("高质量", "high")
        quality_layout.addWidget(self.volume_quality_combo)
        self.params_layout.addLayout(quality_layout)

    def create_common_workflow_params(self, semantic: DataSemantics):
        """创建通用工作流快捷操作。"""
        if self.current_shape is None:
            return
        if len(self.current_shape) >= 3 or 1 in self.current_shape:
            self.quick_2d_btn = QPushButton("一键变成 2D 可画视图")
            self.quick_2d_btn.clicked.connect(self.on_quick_2d_clicked)
            self.params_layout.addWidget(self.quick_2d_btn)

    def create_volume_4d_params(self):
        """创建四维数据参数控件"""
        if self.current_shape is None or len(self.current_shape) != 4:
            return

        # 通道轴选择
        ch_layout = QHBoxLayout()
        ch_layout.addWidget(QLabel("通道轴:"))
        self.channel_axis_combo = QComboBox()
        self.channel_axis_combo.addItems(["轴 0", "轴 1", "轴 2", "轴 3"])
        self.channel_axis_combo.setCurrentIndex(3)  # 默认最后一维
        self.channel_axis_combo.currentIndexChanged.connect(self.on_channel_axis_changed)
        ch_layout.addWidget(self.channel_axis_combo)
        self.params_layout.addLayout(ch_layout)

        # 通道索引
        ch_idx_layout = QHBoxLayout()
        ch_idx_layout.addWidget(QLabel("通道索引:"))
        self.channel_index_spin = QSpinBox()
        self.channel_index_spin.setRange(0, self.current_shape[3] - 1)
        ch_idx_layout.addWidget(self.channel_index_spin)
        self.params_layout.addLayout(ch_idx_layout)

        # 然后是 3D 切片参数
        self.create_volume_params(self.get_volume_shape_after_channel())

    def create_multichannel_params(self):
        """创建多通道 3D 数据参数控件"""
        if self.current_shape is None or len(self.current_shape) != 3:
            return

        # 通道轴选择
        ch_layout = QHBoxLayout()
        ch_layout.addWidget(QLabel("通道轴:"))
        self.channel_axis_combo = QComboBox()
        self.channel_axis_combo.addItems(["轴 0", "轴 1", "轴 2"])
        self.channel_axis_combo.setCurrentIndex(2)  # 默认最后一维
        self.channel_axis_combo.currentIndexChanged.connect(self.on_channel_axis_changed)
        ch_layout.addWidget(self.channel_axis_combo)
        self.params_layout.addLayout(ch_layout)

        # 通道索引
        ch_idx_layout = QHBoxLayout()
        ch_idx_layout.addWidget(QLabel("通道索引:"))
        self.channel_index_spin = QSpinBox()
        self.channel_index_spin.setRange(0, self.current_shape[2] - 1)
        ch_idx_layout.addWidget(self.channel_index_spin)
        self.params_layout.addLayout(ch_idx_layout)

    def on_slice_axis_changed(self, axis_index: int):
        """切片轴改变时更新滑块范围"""
        shape = self.volume_shape if self.volume_shape is not None else self.current_shape
        if shape and len(shape) >= 3:
            max_val = shape[axis_index] - 1
            self.slice_slider.setMaximum(max_val)
            self.slice_slider.setValue(max_val // 2)
            self.slice_index_label.setText(f"{max_val // 2} / {max_val}")

    def on_channel_axis_changed(self, axis_index: int):
        """通道轴改变时更新通道索引和 4D 降维后的切片范围。"""
        if not self.current_shape or axis_index >= len(self.current_shape):
            return

        if hasattr(self, 'channel_index_spin'):
            self.channel_index_spin.setRange(0, self.current_shape[axis_index] - 1)
            self.channel_index_spin.setValue(0)

        if self.current_semantic == DataSemantics.VOLUME_4D:
            self.volume_shape = self.get_volume_shape_after_channel()
            if hasattr(self, 'slice_axis_combo'):
                self.on_slice_axis_changed(self.slice_axis_combo.currentIndex())

    def get_volume_shape_after_channel(self):
        """获取 4D 数据选定通道轴后的 3D 形状。"""
        if not self.current_shape or len(self.current_shape) != 4:
            return None
        channel_axis = self.channel_axis_combo.currentIndex()
        return tuple(size for axis, size in enumerate(self.current_shape) if axis != channel_axis)

    def on_plot_clicked(self):
        """生成图表按钮点击"""
        plot_type = self.plot_type_combo.currentText()
        if not plot_type:
            return

        # 收集参数
        params = {
            'plot_type': plot_type,
            'semantic': self.current_semantic
        }

        # 根据语义类型收集特定参数
        if self.current_semantic == DataSemantics.TABULAR_2D:
            params.update(self.get_tabular_params(plot_type))
        elif self.current_semantic == DataSemantics.IMAGE_2D:
            params.update(self.get_image_params(plot_type))
        elif self.current_semantic == DataSemantics.VOLUME_3D:
            params.update(self.get_volume_params(plot_type))
        elif self.current_semantic == DataSemantics.VOLUME_4D:
            params.update(self.get_volume_4d_params(plot_type))
        elif self.current_semantic == DataSemantics.MULTICHANNEL_3D:
            params.update(self.get_multichannel_params(plot_type))

        self.plot_requested.emit(params)

    def on_quick_2d_clicked(self):
        """请求主窗口执行快速二维化工作流。"""
        self.plot_requested.emit({
            'plot_type': '__quick_2d__',
            'semantic': self.current_semantic,
        })

    def get_tabular_params(self, plot_type: str) -> dict:
        """获取表格数据参数"""
        params = {}

        if plot_type in ["多折线图", "相关性热力图"]:
            selected_items = self.y_cols_list.selectedItems()
            if selected_items:
                params['y_cols'] = [int(item.text().split()[1]) for item in selected_items]
            else:
                params['y_cols'] = None

        if plot_type == "多折线图":
            # X 轴列
            x_col_text = self.x_col_combo.currentText()
            if x_col_text == "行索引":
                params['x_col'] = None
            else:
                params['x_col'] = int(x_col_text.split()[1])

            params['invert_y'] = self.invert_y_check.isChecked()

        elif plot_type == "单列直方图":
            params['col_idx'] = self.single_col_spin.value()

        elif plot_type == "散点图":
            params['x_col'] = self.scatter_x_spin.value()
            params['y_col'] = self.scatter_y_spin.value()

        return params

    def get_image_params(self, plot_type: str) -> dict:
        """获取二维图像/矩阵参数。"""
        return {
            'swap_axes': self.swap_image_axes_check.isChecked(),
            'x_order': self.image_x_order_combo.currentData(),
            'y_order': self.image_y_order_combo.currentData(),
            'cmap': self.image_cmap_combo.currentText(),
            'aspect': self.image_aspect_combo.currentData(),
        }

    def get_volume_params(self, plot_type: str) -> dict:
        """获取三维体数据参数"""
        params = {}

        if plot_type == "切片热力图":
            params['axis'] = self.slice_axis_combo.currentIndex()
            params['index'] = self.slice_slider.value()

        elif plot_type == "投影图":
            params['axis'] = self.slice_axis_combo.currentIndex()
            method = self.proj_method_combo.currentText()
            params['method'] = method if method != "无" else "mean"

        if hasattr(self, 'volume_quality_combo'):
            params['quality'] = self.volume_quality_combo.currentData()

        return params

    def get_volume_4d_params(self, plot_type: str) -> dict:
        """获取四维数据参数"""
        params = {}
        params['channel_axis'] = self.channel_axis_combo.currentIndex()
        params['channel_index'] = self.channel_index_spin.value()
        params.update(self.get_volume_params(plot_type))
        return params

    def get_multichannel_params(self, plot_type: str) -> dict:
        """获取多通道 3D 数据参数"""
        params = {}
        params['channel_axis'] = self.channel_axis_combo.currentIndex()
        params['channel_index'] = self.channel_index_spin.value()
        return params
