"""
自定义控件模块
提供切片控制、绘图控制等组件
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QComboBox, QGroupBox
)
from PySide6.QtCore import Signal
from typing import List


class SliceControlWidget(QWidget):
    """切片控制组件"""

    slice_changed = Signal(list)  # 发送切片字符串列表

    def __init__(self, parent=None):
        super().__init__(parent)
        self.slice_inputs: List[QLineEdit] = []
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # 标题
        title = QLabel("切片控制")
        title.setStyleSheet("font-weight: bold;")
        layout.addWidget(title)

        # 切片输入容器
        self.slice_container = QVBoxLayout()
        layout.addLayout(self.slice_container)

        # 应用按钮
        self.apply_btn = QPushButton("应用切片")
        self.apply_btn.clicked.connect(self.on_apply_clicked)
        layout.addWidget(self.apply_btn)

        layout.addStretch()

    def set_array_shape(self, shape: tuple):
        """根据数组形状设置切片输入框"""
        # 清空现有输入框
        for i in reversed(range(self.slice_container.count())):
            widget = self.slice_container.itemAt(i).widget()
            if widget:
                widget.deleteLater()

        self.slice_inputs.clear()

        # 为每个维度创建输入框
        for i, size in enumerate(shape):
            dim_layout = QHBoxLayout()

            label = QLabel(f"维度 {i} (0-{size-1}):")
            dim_layout.addWidget(label)

            line_edit = QLineEdit()
            line_edit.setPlaceholderText(":")
            line_edit.setText(":")
            dim_layout.addWidget(line_edit)

            self.slice_inputs.append(line_edit)
            self.slice_container.addLayout(dim_layout)

    def on_apply_clicked(self):
        """应用按钮点击"""
        slice_specs = [inp.text().strip() or ":" for inp in self.slice_inputs]
        self.slice_changed.emit(slice_specs)

    def get_slice_specs(self) -> List[str]:
        """获取当前切片规格"""
        return [inp.text().strip() or ":" for inp in self.slice_inputs]


class PlotControlWidget(QWidget):
    """绘图控制组件"""

    plot_requested = Signal(str)  # 发送绘图类型

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # 标题
        title = QLabel("绘图控制")
        title.setStyleSheet("font-weight: bold;")
        layout.addWidget(title)

        # 绘图类型选择
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("图表类型:"))

        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(["折线图", "直方图", "热力图"])
        type_layout.addWidget(self.plot_type_combo)

        layout.addLayout(type_layout)

        # 绘图按钮
        self.plot_btn = QPushButton("生成图表")
        self.plot_btn.clicked.connect(self.on_plot_clicked)
        layout.addWidget(self.plot_btn)

        layout.addStretch()

    def on_plot_clicked(self):
        """绘图按钮点击"""
        plot_type = self.plot_type_combo.currentText()
        self.plot_requested.emit(plot_type)

    def get_plot_type(self) -> str:
        """获取当前选择的绘图类型"""
        return self.plot_type_combo.currentText()
