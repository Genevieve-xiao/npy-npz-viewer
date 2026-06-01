"""
Dimension filter controls.
"""
from typing import List

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from npy_npz_viewer.core.dimension_filter import apply_dimension_filter


class DimensionFilterWidget(QWidget):
    """Non-destructive axis/column filtering widget."""

    filter_applied = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_array = None
        self.axis_controls = []
        self.column_list = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        title = QLabel("维度筛选")
        title.setStyleSheet("font-weight: bold;")
        layout.addWidget(title)

        info = QLabel("先筛选维度/列，再进行切片、预览、绘图和导出。格式支持 :、0,2,5、1:10:2")
        info.setWordWrap(True)
        info.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(info)

        self.axis_container = QVBoxLayout()
        layout.addLayout(self.axis_container)

        self.column_hint = QLabel("二维表格列/特征快捷选择:")
        self.column_hint.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(self.column_hint)

        self.column_list = QListWidget()
        self.column_list.setSelectionMode(QListWidget.MultiSelection)
        self.column_list.setMaximumHeight(90)
        layout.addWidget(self.column_list)

        quick_layout = QHBoxLayout()
        self.keep_columns_btn = QPushButton("只保留选中列")
        self.keep_columns_btn.clicked.connect(lambda: self.apply_column_selection("keep"))
        quick_layout.addWidget(self.keep_columns_btn)

        self.drop_columns_btn = QPushButton("剔除选中列")
        self.drop_columns_btn.clicked.connect(lambda: self.apply_column_selection("drop"))
        quick_layout.addWidget(self.drop_columns_btn)
        layout.addLayout(quick_layout)

        btn_layout = QHBoxLayout()
        self.apply_btn = QPushButton("应用筛选")
        self.apply_btn.clicked.connect(self.on_apply_clicked)
        btn_layout.addWidget(self.apply_btn)

        self.reset_btn = QPushButton("重置筛选")
        self.reset_btn.clicked.connect(self.on_reset_clicked)
        btn_layout.addWidget(self.reset_btn)
        layout.addLayout(btn_layout)

        self.result_label = QLabel("")
        self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet("color: #0066cc; font-size: 11px;")
        layout.addWidget(self.result_label)

        layout.addStretch()
        self.set_controls_enabled(False)

    def set_controls_enabled(self, enabled: bool):
        for widget in [
            self.column_hint,
            self.column_list,
            self.keep_columns_btn,
            self.drop_columns_btn,
            self.apply_btn,
            self.reset_btn,
        ]:
            widget.setEnabled(enabled)

    def clear_axis_controls(self):
        while self.axis_container.count():
            item = self.axis_container.takeAt(0)
            layout = item.layout()
            if layout is not None:
                while layout.count():
                    child = layout.takeAt(0)
                    widget = child.widget()
                    if widget:
                        widget.deleteLater()
                layout.deleteLater()
        self.axis_controls.clear()

    def set_array(self, array):
        """Set the original array to filter."""
        self.current_array = array
        self.result_label.setText("")
        self.clear_axis_controls()
        self.column_list.clear()

        if array is None:
            self.set_controls_enabled(False)
            return

        self.set_controls_enabled(True)

        for axis, size in enumerate(array.shape):
            row = QHBoxLayout()

            label = QLabel(f"轴 {axis} (0-{size - 1}):")
            label.setMinimumWidth(88)
            row.addWidget(label)

            mode_combo = QComboBox()
            mode_combo.addItem("保留", "keep")
            mode_combo.addItem("剔除", "drop")
            row.addWidget(mode_combo)

            spec_input = QLineEdit()
            spec_input.setPlaceholderText(":")
            spec_input.setText(":")
            row.addWidget(spec_input)

            self.axis_controls.append(
                {"axis": axis, "mode": mode_combo, "spec": spec_input}
            )
            self.axis_container.addLayout(row)

        is_table = array.ndim == 2
        self.column_hint.setVisible(is_table)
        self.column_list.setVisible(is_table)
        self.keep_columns_btn.setVisible(is_table)
        self.drop_columns_btn.setVisible(is_table)

        if is_table:
            for col in range(array.shape[1]):
                self.column_list.addItem(f"列 {col}")

    def collect_axis_filters(self) -> List[dict]:
        filters = []
        for control in self.axis_controls:
            spec = control["spec"].text().strip() or ":"
            mode = control["mode"].currentData()
            if spec == ":" and mode == "keep":
                continue
            filters.append(
                {
                    "axis": control["axis"],
                    "mode": mode,
                    "spec": spec,
                }
            )
        return filters

    def on_apply_clicked(self):
        if self.current_array is None:
            return

        result = apply_dimension_filter(self.current_array, self.collect_axis_filters())
        self.show_result(result)
        if result["success"]:
            self.filter_applied.emit(result)

    def on_reset_clicked(self):
        if self.current_array is None:
            return

        for control in self.axis_controls:
            control["mode"].setCurrentIndex(0)
            control["spec"].setText(":")
        self.column_list.clearSelection()

        result = {
            "success": True,
            "array": self.current_array,
            "axis_index_maps": {},
            "summary": "筛选已重置",
        }
        self.show_result(result)
        self.filter_applied.emit(result)

    def apply_column_selection(self, mode: str):
        if self.current_array is None or self.current_array.ndim != 2:
            return

        selected = self.column_list.selectedItems()
        if not selected:
            self.result_label.setText("请先选择一个或多个列")
            self.result_label.setStyleSheet("color: #cc0000; font-size: 11px;")
            return

        cols = [self.column_list.row(item) for item in selected]
        spec = ",".join(str(col) for col in cols)
        column_control = self.axis_controls[1]
        column_control["mode"].setCurrentIndex(0 if mode == "keep" else 1)
        column_control["spec"].setText(spec)
        self.on_apply_clicked()

    def show_result(self, result: dict):
        if not result["success"]:
            self.result_label.setText(f"筛选失败: {result['error']}")
            self.result_label.setStyleSheet("color: #cc0000; font-size: 11px;")
            return

        array = result["array"]
        summary = result.get("summary") or "筛选成功"
        self.result_label.setText(f"{summary}\n当前形状: {array.shape}")
        self.result_label.setStyleSheet("color: #0066cc; font-size: 11px;")
