"""
自定义预览控件
允许用户指定预览的行范围
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QSpinBox, QPushButton, QGroupBox
)
from PySide6.QtCore import Signal


class CustomPreviewWidget(QWidget):
    """自定义预览控件"""

    preview_changed = Signal(str, int, int)  # 发送 (mode, start_row, end_row)
    preview_range_changed = Signal(int, int)  # 兼容旧信号

    def __init__(self, parent=None):
        super().__init__(parent)
        self.max_rows = 0
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("预览模式:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("自动", "auto")
        self.mode_combo.addItem("展平", "flat")
        self.mode_combo.addItem("二维切片", "slice")
        self.mode_combo.addItem("轴摘要", "summary")
        self.mode_combo.currentIndexChanged.connect(self.on_apply_clicked)
        layout.addWidget(self.mode_combo)

        # 起始行
        layout.addWidget(QLabel("范围:"))

        self.start_spin = QSpinBox()
        self.start_spin.setMinimum(0)
        self.start_spin.setMaximum(0)
        self.start_spin.setValue(0)
        self.start_spin.setPrefix("从 ")
        layout.addWidget(self.start_spin)

        layout.addWidget(QLabel("到"))

        # 结束行
        self.end_spin = QSpinBox()
        self.end_spin.setMinimum(1)
        self.end_spin.setMaximum(1)
        self.end_spin.setValue(1000)
        layout.addWidget(self.end_spin)

        # 应用按钮
        self.apply_btn = QPushButton("应用")
        self.apply_btn.clicked.connect(self.on_apply_clicked)
        layout.addWidget(self.apply_btn)

        # 快速选择按钮
        self.quick_1k_btn = QPushButton("前1000")
        self.quick_1k_btn.clicked.connect(lambda: self.set_range(0, 1000))
        layout.addWidget(self.quick_1k_btn)

        self.quick_10k_btn = QPushButton("前10000")
        self.quick_10k_btn.clicked.connect(lambda: self.set_range(0, 10000))
        layout.addWidget(self.quick_10k_btn)

        self.quick_all_btn = QPushButton("全部")
        self.quick_all_btn.clicked.connect(self.set_all)
        layout.addWidget(self.quick_all_btn)

        layout.addStretch()

    def set_array_size(self, rows: int, reset_range: bool = False):
        """设置数组大小"""
        previous_start = self.start_spin.value()
        previous_end = self.end_spin.value()
        size_changed = rows != self.max_rows
        self.max_rows = rows

        self.start_spin.setMaximum(max(0, rows - 1))
        self.end_spin.setMaximum(rows)

        if reset_range or size_changed:
            default_end = min(1000, rows)
            self.start_spin.setValue(0)
            self.end_spin.setValue(default_end)
        else:
            self.start_spin.setValue(min(previous_start, max(0, rows - 1)))
            self.end_spin.setValue(min(max(previous_end, 1), rows))

        # 更新快速按钮状态
        self.quick_1k_btn.setEnabled(rows > 1000)
        self.quick_10k_btn.setEnabled(rows > 10000)

    def set_range(self, start: int, end: int):
        """设置范围"""
        self.start_spin.setValue(start)
        self.end_spin.setValue(min(end, self.max_rows))
        self.on_apply_clicked()

    def set_all(self):
        """显示全部"""
        self.start_spin.setValue(0)
        self.end_spin.setValue(self.max_rows)
        self.on_apply_clicked()

    def on_apply_clicked(self):
        """应用按钮点击"""
        start = self.start_spin.value()
        end = self.end_spin.value()

        if start >= end:
            end = start + 1
            self.end_spin.setValue(end)

        self.preview_range_changed.emit(start, end)
        self.preview_changed.emit(self.mode_combo.currentData(), start, end)

    def get_range(self):
        """获取当前范围"""
        return self.start_spin.value(), self.end_spin.value()

    def get_mode(self):
        """获取当前预览模式。"""
        return self.mode_combo.currentData()
