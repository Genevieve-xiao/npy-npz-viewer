"""
主窗口模块
"""
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QTextEdit, QTableWidget, QTableWidgetItem,
    QPushButton, QFileDialog, QMessageBox, QListWidget,
    QLabel, QGroupBox
)
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import pandas as pd

from core.loaders import ArrayLoader
from core.stats import ArrayStats
from core.slicing import ArraySlicer
from core.visualization import ArrayVisualizer
from utils.helpers import PreviewHelper
from ui.widgets import SliceControlWidget, PlotControlWidget


class MainWindow(QMainWindow):
    """主窗口"""

    def __init__(self):
        super().__init__()
        self.loader = ArrayLoader()
        self.current_sliced_array = None  # 当前切片后的数组
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("NPY/NPZ 数据查看器")
        self.setGeometry(100, 100, 1400, 900)

        # 中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # 左侧面板
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)

        # 右侧面板
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)

        # 设置分割比例
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)

    def create_left_panel(self) -> QWidget:
        """创建左侧面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # 文件操作
        file_group = QGroupBox("文件操作")
        file_layout = QVBoxLayout()

        self.open_btn = QPushButton("打开文件")
        self.open_btn.clicked.connect(self.open_file)
        file_layout.addWidget(self.open_btn)

        self.file_info_label = QLabel("未加载文件")
        self.file_info_label.setWordWrap(True)
        file_layout.addWidget(self.file_info_label)

        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        # NPZ 键列表
        npz_group = QGroupBox("NPZ 数组列表")
        npz_layout = QVBoxLayout()

        self.npz_list = QListWidget()
        self.npz_list.itemClicked.connect(self.on_npz_key_selected)
        npz_layout.addWidget(self.npz_list)

        npz_group.setLayout(npz_layout)
        layout.addWidget(npz_group)

        # 切片控制
        self.slice_control = SliceControlWidget()
        self.slice_control.slice_changed.connect(self.on_slice_changed)
        layout.addWidget(self.slice_control)

        # 绘图控制
        self.plot_control = PlotControlWidget()
        self.plot_control.plot_requested.connect(self.on_plot_requested)
        layout.addWidget(self.plot_control)

        layout.addStretch()
        return panel

    def create_right_panel(self) -> QWidget:
        """创建右侧面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # 统计信息
        stats_group = QGroupBox("数组统计信息")
        stats_layout = QVBoxLayout()

        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMaximumHeight(200)
        stats_layout.addWidget(self.stats_text)

        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)

        # 数据预览
        preview_group = QGroupBox("数据预览")
        preview_layout = QVBoxLayout()

        self.preview_info_label = QLabel("")
        preview_layout.addWidget(self.preview_info_label)

        self.preview_table = QTableWidget()
        preview_layout.addWidget(self.preview_table)

        # 导出按钮
        export_layout = QHBoxLayout()
        self.export_csv_btn = QPushButton("导出为 CSV")
        self.export_csv_btn.clicked.connect(self.export_csv)
        export_layout.addWidget(self.export_csv_btn)
        export_layout.addStretch()
        preview_layout.addLayout(export_layout)

        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)

        # 图表区域
        plot_group = QGroupBox("可视化")
        plot_layout = QVBoxLayout()

        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvasQTAgg(self.figure)
        plot_layout.addWidget(self.canvas)

        # 保存图表按钮
        save_plot_layout = QHBoxLayout()
        self.save_plot_btn = QPushButton("保存图表为 PNG")
        self.save_plot_btn.clicked.connect(self.save_plot)
        save_plot_layout.addWidget(self.save_plot_btn)
        save_plot_layout.addStretch()
        plot_layout.addLayout(save_plot_layout)

        plot_group.setLayout(plot_layout)
        layout.addWidget(plot_group)

        return panel

    def open_file(self):
        """打开文件对话框"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 NPY/NPZ 文件",
            "",
            "NumPy 文件 (*.npy *.npz);;所有文件 (*.*)"
        )

        if not file_path:
            return

        result = self.loader.load_file(file_path)

        if not result['success']:
            QMessageBox.critical(self, "错误", result['error'])
            return

        # 更新文件信息
        self.file_info_label.setText(f"文件: {file_path}\n类型: {result['file_type'].upper()}")

        # 处理不同文件类型
        if result['file_type'] == 'npy':
            self.npz_list.clear()
            self.load_array(result['array'])
        else:  # npz
            self.npz_list.clear()
            self.npz_list.addItems(result['keys'])
            self.npz_list.setCurrentRow(0)
            # 加载第一个数组
            array = self.loader.get_current_array()
            if array is not None:
                self.load_array(array)

    def on_npz_key_selected(self, item):
        """NPZ 键选择"""
        key = item.text()
        result = self.loader.switch_npz_key(key)

        if not result['success']:
            QMessageBox.critical(self, "错误", result['error'])
            return

        self.load_array(result['array'])

    def load_array(self, array):
        """加载数组并显示信息"""
        # 重置切片数组
        self.current_sliced_array = array

        # 计算并显示统计信息
        stats = ArrayStats.compute_stats(array)
        stats_text = ArrayStats.format_stats(stats)
        self.stats_text.setText(stats_text)

        # 设置切片控制
        self.slice_control.set_array_shape(array.shape)

        # 显示预览
        self.update_preview(array)

        # 清空图表
        self.figure.clear()
        self.canvas.draw()

    def update_preview(self, array):
        """更新数据预览"""
        preview_array, info = PreviewHelper.get_preview_slice(array)
        self.preview_info_label.setText(info)

        df = PreviewHelper.array_to_dataframe(preview_array)
        if df is None:
            self.preview_table.clear()
            return

        # 填充表格
        self.preview_table.setRowCount(len(df))
        self.preview_table.setColumnCount(len(df.columns))
        self.preview_table.setHorizontalHeaderLabels([str(c) for c in df.columns])

        for i in range(len(df)):
            for j in range(len(df.columns)):
                value = df.iloc[i, j]
                item = QTableWidgetItem(str(value))
                self.preview_table.setItem(i, j, item)

    def on_slice_changed(self, slice_specs):
        """切片改变"""
        array = self.loader.get_current_array()
        if array is None:
            QMessageBox.warning(self, "警告", "未加载数组")
            return

        result = ArraySlicer.apply_slice(array, slice_specs)

        if not result['success']:
            QMessageBox.critical(self, "错误", result['error'])
            return

        sliced = result['array']
        self.current_sliced_array = sliced

        # 更新预览
        self.update_preview(sliced)

        # 更新统计信息
        stats = ArrayStats.compute_stats(sliced)
        stats_text = ArrayStats.format_stats(stats)
        self.stats_text.setText(stats_text)

        QMessageBox.information(self, "成功", f"切片后形状: {sliced.shape}")

    def on_plot_requested(self, plot_type):
        """绘图请求"""
        if self.current_sliced_array is None:
            QMessageBox.warning(self, "警告", "未加载数组")
            return

        array = self.current_sliced_array

        # 检查是否可以绘图
        check = ArrayVisualizer.can_plot(array, plot_type)
        if not check['can_plot']:
            QMessageBox.warning(self, "无法绘图", check['reason'])
            return

        # 绘图
        if plot_type == "折线图":
            result = ArrayVisualizer.plot_1d_line(array, self.figure)
        elif plot_type == "直方图":
            result = ArrayVisualizer.plot_1d_histogram(array, self.figure)
        elif plot_type == "热力图":
            result = ArrayVisualizer.plot_2d_heatmap(array, self.figure)
        else:
            QMessageBox.warning(self, "错误", "未知的绘图类型")
            return

        if not result['success']:
            QMessageBox.critical(self, "绘图失败", result['error'])
            return

        self.canvas.draw()

    def export_csv(self):
        """导出为 CSV"""
        if self.current_sliced_array is None:
            QMessageBox.warning(self, "警告", "未加载数组")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存 CSV 文件",
            "",
            "CSV 文件 (*.csv)"
        )

        if not file_path:
            return

        try:
            df = PreviewHelper.array_to_dataframe(self.current_sliced_array)
            if df is None:
                QMessageBox.critical(self, "错误", "无法转换为表格格式")
                return

            df.to_csv(file_path, index=False)
            QMessageBox.information(self, "成功", f"已导出到: {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出失败: {str(e)}")

    def save_plot(self):
        """保存图表"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存图表",
            "",
            "PNG 图像 (*.png)"
        )

        if not file_path:
            return

        try:
            self.figure.savefig(file_path, dpi=300, bbox_inches='tight')
            QMessageBox.information(self, "成功", f"图表已保存到: {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存失败: {str(e)}")
