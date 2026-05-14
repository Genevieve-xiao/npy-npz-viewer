"""
通用切片控件
支持对任意维度数组进行切片
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QGroupBox
)
from PySide6.QtCore import Signal
from typing import List


class UniversalSliceWidget(QWidget):
    """通用切片控件"""

    slice_applied = Signal(object)  # 发送切片后的数组

    def __init__(self, parent=None):
        super().__init__(parent)
        self.slice_inputs: List[QLineEdit] = []
        self.current_array = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # 标题
        title = QLabel("通用切片")
        title.setStyleSheet("font-weight: bold;")
        layout.addWidget(title)

        # 说明
        info = QLabel("对大数据先切片可提升性能\n格式: start:stop:step 或 : 表示全部")
        info.setStyleSheet("color: #666; font-size: 11px;")
        info.setWordWrap(True)
        layout.addWidget(info)

        # 切片输入容器
        self.slice_container = QVBoxLayout()
        layout.addLayout(self.slice_container)

        # 按钮行
        btn_layout = QHBoxLayout()

        self.apply_btn = QPushButton("应用切片")
        self.apply_btn.clicked.connect(self.on_apply_clicked)
        btn_layout.addWidget(self.apply_btn)

        self.reset_btn = QPushButton("重置")
        self.reset_btn.clicked.connect(self.on_reset_clicked)
        btn_layout.addWidget(self.reset_btn)

        layout.addLayout(btn_layout)

        # 切片结果提示
        self.result_label = QLabel("")
        self.result_label.setStyleSheet("color: #0066cc; font-size: 11px;")
        self.result_label.setWordWrap(True)
        layout.addWidget(self.result_label)

        layout.addStretch()

    def set_array(self, array):
        """设置要切片的数组"""
        self.current_array = array
        self.result_label.setText("")

        # 清空现有输入框
        for i in reversed(range(self.slice_container.count())):
            item = self.slice_container.itemAt(i)
            if item.layout():
                # 清空布局中的控件
                while item.layout().count():
                    widget = item.layout().takeAt(0).widget()
                    if widget:
                        widget.deleteLater()
                item.layout().deleteLater()

        self.slice_inputs.clear()

        # 为每个维度创建输入框
        for i, size in enumerate(array.shape):
            dim_layout = QHBoxLayout()

            label = QLabel(f"维度 {i} (0-{size-1}):")
            label.setMinimumWidth(120)
            dim_layout.addWidget(label)

            line_edit = QLineEdit()
            line_edit.setPlaceholderText(":")
            line_edit.setText(":")
            dim_layout.addWidget(line_edit)

            self.slice_inputs.append(line_edit)
            self.slice_container.addLayout(dim_layout)

    def on_apply_clicked(self):
        """应用切片"""
        if self.current_array is None:
            return

        from core.slicing import ArraySlicer

        slice_specs = [inp.text().strip() or ":" for inp in self.slice_inputs]
        result = ArraySlicer.apply_slice(self.current_array, slice_specs)

        if not result['success']:
            self.result_label.setText(f"❌ {result['error']}")
            self.result_label.setStyleSheet("color: #cc0000; font-size: 11px;")
            return

        sliced_array = result['array']

        # 显示切片结果
        original_shape = self.current_array.shape
        new_shape = sliced_array.shape
        reduction = (1 - sliced_array.size / self.current_array.size) * 100

        self.result_label.setText(
            f"✓ 切片成功\n"
            f"原始: {original_shape} ({self.current_array.size:,} 个元素)\n"
            f"切片后: {new_shape} ({sliced_array.size:,} 个元素)\n"
            f"数据量减少: {reduction:.1f}%"
        )
        self.result_label.setStyleSheet("color: #0066cc; font-size: 11px;")

        # 发送切片后的数组
        self.slice_applied.emit(sliced_array)

    def on_reset_clicked(self):
        """重置切片"""
        if self.current_array is None:
            return

        # 重置所有输入框
        for inp in self.slice_inputs:
            inp.setText(":")

        self.result_label.setText("")

        # 发送原始数组
        self.slice_applied.emit(self.current_array)

    def get_quick_slice_suggestions(self, array) -> List[str]:
        """获取快速切片建议"""
        suggestions = []

        if array.ndim >= 1 and array.shape[0] > 10000:
            suggestions.append(f"前 10000 行: 0:10000")

        if array.ndim >= 2 and array.shape[0] > 1000:
            suggestions.append(f"前 1000 行: 0:1000")

        if array.ndim >= 1 and array.shape[0] > 100:
            suggestions.append(f"每 10 个采样: ::10")

        return suggestions
