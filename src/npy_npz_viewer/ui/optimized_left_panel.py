"""
优化后的左侧面板 - 使用折叠组和滚动区域
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QGroupBox, QListWidget, QScrollArea,
    QSizePolicy, QComboBox
)
from PySide6.QtCore import Qt, Signal
from npy_npz_viewer.ui.dimension_filter import DimensionFilterWidget
from npy_npz_viewer.ui.semantic_control import SemanticControlWidget
from npy_npz_viewer.ui.universal_slice import UniversalSliceWidget


class CollapsibleGroupBox(QGroupBox):
    """可折叠的 GroupBox"""

    def __init__(self, title: str, parent=None):
        super().__init__(title, parent)
        self.setCheckable(True)
        self.setChecked(True)  # 默认展开
        self.toggled.connect(self.on_toggled)

        # 内容容器
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(5, 5, 5, 5)

        # 主布局
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.content_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)

    def on_toggled(self, checked: bool):
        """折叠/展开"""
        self.content_widget.setVisible(checked)

    def addWidget(self, widget):
        """添加控件到内容区域"""
        self.content_layout.addWidget(widget)

    def addLayout(self, layout):
        """添加布局到内容区域"""
        self.content_layout.addLayout(layout)


class OptimizedLeftPanel(QWidget):
    """优化后的左侧面板 - 带滚动条和折叠组"""

    # 信号
    file_opened = Signal(str)
    npz_key_selected = Signal(str)
    dimension_filter_applied = Signal(object)
    slice_applied = Signal(object)
    semantic_changed = Signal(str)
    plot_requested = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        # 主布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # 创建滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # 滚动内容容器
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(5, 5, 5, 5)
        scroll_layout.setSpacing(10)

        # 1. 文件操作（始终可见，不折叠）
        file_group = QGroupBox("文件操作")
        file_layout = QVBoxLayout()

        self.open_btn = QPushButton("打开文件")
        self.open_btn.clicked.connect(self.on_open_clicked)
        file_layout.addWidget(self.open_btn)

        self.open_zarr_btn = QPushButton("打开 Zarr 目录")
        self.open_zarr_btn.clicked.connect(self.on_open_zarr_clicked)
        file_layout.addWidget(self.open_zarr_btn)

        self.recent_paths = []
        self.recent_combo = QComboBox()
        self.recent_combo.addItem("最近文件")
        self.recent_combo.activated.connect(self.on_recent_file_activated)
        file_layout.addWidget(self.recent_combo)

        self.file_info_label = QLabel("未加载文件")
        self.file_info_label.setWordWrap(True)
        self.file_info_label.setStyleSheet("font-size: 11px; color: #666;")
        file_layout.addWidget(self.file_info_label)

        self.drop_tip_label = QLabel("也可以将 .npy / .npz 文件或 .zarr 目录拖到窗口中打开")
        self.drop_tip_label.setWordWrap(True)
        self.drop_tip_label.setStyleSheet("font-size: 11px; color: #888;")
        file_layout.addWidget(self.drop_tip_label)

        file_group.setLayout(file_layout)
        scroll_layout.addWidget(file_group)

        # 2. 多数组源列表（可折叠）
        self.npz_group = CollapsibleGroupBox("数组列表")
        self.npz_group.setChecked(False)  # 默认折叠

        self.npz_list = QListWidget()
        self.npz_list.setMaximumHeight(150)
        self.npz_list.itemClicked.connect(self.on_npz_item_clicked)
        self.npz_group.addWidget(self.npz_list)

        scroll_layout.addWidget(self.npz_group)

        # 3. 维度筛选（可折叠，默认展开）
        self.dimension_filter_group = CollapsibleGroupBox("维度筛选")
        self.dimension_filter_group.setChecked(True)

        self.dimension_filter_widget = DimensionFilterWidget()
        self.dimension_filter_widget.filter_applied.connect(
            self.on_dimension_filter_applied_internal
        )
        self.dimension_filter_group.content_layout.addWidget(self.dimension_filter_widget)

        scroll_layout.addWidget(self.dimension_filter_group)

        # 4. 通用切片（可折叠，默认展开）
        self.slice_group = CollapsibleGroupBox("通用切片")
        self.slice_group.setChecked(True)  # 默认展开

        self.slice_widget = UniversalSliceWidget()
        self.slice_widget.slice_applied.connect(self.on_slice_applied_internal)
        self.slice_group.content_layout.addWidget(self.slice_widget)

        scroll_layout.addWidget(self.slice_group)

        # 5. 数据解释方式（可折叠，默认展开）
        self.semantic_group = CollapsibleGroupBox("数据解释方式")
        self.semantic_group.setChecked(True)  # 默认展开

        self.semantic_control = SemanticControlWidget()
        self.semantic_control.semantic_changed.connect(self.on_semantic_changed_internal)
        self.semantic_control.plot_requested.connect(self.on_plot_requested_internal)
        self.semantic_group.content_layout.addWidget(self.semantic_control)

        scroll_layout.addWidget(self.semantic_group)

        # 添加弹性空间
        scroll_layout.addStretch()

        # 设置滚动内容
        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area)

        # 底部提示
        tip_label = QLabel("💡 点击组标题可折叠/展开")
        tip_label.setStyleSheet("font-size: 10px; color: #999; padding: 5px;")
        tip_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(tip_label)

    def on_open_clicked(self):
        """打开文件按钮点击"""
        from PySide6.QtWidgets import QFileDialog
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 NPY/NPZ 文件",
            "",
            "NumPy/Zarr 文件 (*.npy *.npz *.zarr);;所有文件 (*.*)"
        )
        if file_path:
            self.file_opened.emit(file_path)

    def on_open_zarr_clicked(self):
        """Open a local .zarr directory."""
        from PySide6.QtWidgets import QFileDialog
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "选择 Zarr 目录",
            "",
            QFileDialog.ShowDirsOnly,
        )
        if dir_path:
            self.file_opened.emit(dir_path)

    def on_npz_item_clicked(self, item):
        """NPZ 列表项点击"""
        self.npz_key_selected.emit(item.text())

    def on_recent_file_activated(self, index: int):
        """最近文件选择。"""
        if index <= 0:
            return
        path_index = index - 1
        if 0 <= path_index < len(self.recent_paths):
            self.file_opened.emit(self.recent_paths[path_index])

    def on_slice_applied_internal(self, array):
        """切片应用"""
        self.slice_applied.emit(array)

    def on_dimension_filter_applied_internal(self, result):
        """维度筛选应用"""
        self.dimension_filter_applied.emit(result)

    def on_semantic_changed_internal(self, semantic_text):
        """语义改变"""
        self.semantic_changed.emit(semantic_text)

    def on_plot_requested_internal(self, params):
        """绘图请求"""
        self.plot_requested.emit(params)

    def set_file_info(self, text: str):
        """设置文件信息"""
        self.file_info_label.setText(text)

    def set_recent_files(self, paths: list):
        """更新最近文件列表。"""
        self.recent_paths = list(paths or [])
        self.recent_combo.blockSignals(True)
        self.recent_combo.clear()
        self.recent_combo.addItem("最近文件")
        for path in self.recent_paths:
            self.recent_combo.addItem(path)
        self.recent_combo.setCurrentIndex(0)
        self.recent_combo.blockSignals(False)

    def set_npz_keys(self, keys: list):
        """设置 NPZ/Zarr 键列表"""
        self.npz_list.clear()
        if keys:
            self.npz_list.addItems(keys)
            self.npz_group.setChecked(True)  # 有数据时自动展开
            self.npz_list.setCurrentRow(0)
        else:
            self.npz_group.setChecked(False)  # 无数据时折叠

    def set_array(self, array):
        """设置数组（用于切片和语义控制）"""
        self.dimension_filter_widget.set_array(array)
        self.slice_widget.set_array(array)

    def set_slice_array(self, array):
        """设置切片基准数组。"""
        self.slice_widget.set_array(array)

    def set_array_info(self, shape: tuple, semantic_info: dict):
        """设置数组信息（用于语义控制）"""
        self.semantic_control.set_array_info(shape, semantic_info)
