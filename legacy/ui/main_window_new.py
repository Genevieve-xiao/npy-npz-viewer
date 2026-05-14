"""
重构后的主窗口模块
集成语义驱动的可视化系统
"""
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QTextEdit, QTableWidget, QTableWidgetItem,
    QPushButton, QFileDialog, QMessageBox, QListWidget,
    QLabel, QGroupBox, QTabWidget
)
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import pandas as pd
import numpy as np

from core.loaders import ArrayLoader
from core.stats import ArrayStats
from core.semantic_inference import SemanticInference
from core.data_semantics import DataSemantics
from core.projection import ArrayProjection
from core.visualization_new import SemanticVisualizer
from utils.helpers import PreviewHelper
from ui.semantic_control import SemanticControlWidget


class MainWindowNew(QMainWindow):
    """重构后的主窗口"""

    def __init__(self):
        super().__init__()
        self.loader = ArrayLoader()
        self.current_array = None  # 原始数组
        self.current_semantic = None  # 当前语义
        self.semantic_info = None  # 语义推断信息
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("NPY/NPZ 数据查看器 - 语义驱动版")
        self.setGeometry(100, 100, 1600, 1000)

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

        # 语义控制（新增）
        self.semantic_control = SemanticControlWidget()
        self.semantic_control.semantic_changed.connect(self.on_semantic_changed)
        self.semantic_control.plot_requested.connect(self.on_plot_requested_new)
        layout.addWidget(self.semantic_control)

        layout.addStretch()
        return panel

    def create_right_panel(self) -> QWidget:
        """创建右侧面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # 使用 Tab 组织内容
        tabs = QTabWidget()

        # Tab 1: 统计信息
        stats_tab = self.create_stats_tab()
        tabs.addTab(stats_tab, "统计信息")

        # Tab 2: 数据预览
        preview_tab = self.create_preview_tab()
        tabs.addTab(preview_tab, "数据预览")

        # Tab 3: 可视化
        viz_tab = self.create_viz_tab()
        tabs.addTab(viz_tab, "可视化")

        layout.addWidget(tabs)
        return panel

    def create_stats_tab(self) -> QWidget:
        """创建统计信息标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # 语义信息
        self.semantic_info_text = QTextEdit()
        self.semantic_info_text.setReadOnly(True)
        self.semantic_info_text.setMaximumHeight(100)
        layout.addWidget(QLabel("数据语义:"))
        layout.addWidget(self.semantic_info_text)

        # 统计信息
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        layout.addWidget(QLabel("数组统计:"))
        layout.addWidget(self.stats_text)

        return tab

    def create_preview_tab(self) -> QWidget:
        """创建数据预览标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.preview_info_label = QLabel("")
        layout.addWidget(self.preview_info_label)

        self.preview_table = QTableWidget()
        layout.addWidget(self.preview_table)

        # 导出按钮
        export_layout = QHBoxLayout()
        self.export_csv_btn = QPushButton("导出为 CSV")
        self.export_csv_btn.clicked.connect(self.export_csv)
        export_layout.addWidget(self.export_csv_btn)
        export_layout.addStretch()
        layout.addLayout(export_layout)

        return tab

    def create_viz_tab(self) -> QWidget:
        """创建可视化标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # 图表区域
        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(self.canvas)

        # 保存图表按钮
        save_layout = QHBoxLayout()
        self.save_plot_btn = QPushButton("保存图表为 PNG")
        self.save_plot_btn.clicked.connect(self.save_plot)
        save_layout.addWidget(self.save_plot_btn)
        save_layout.addStretch()
        layout.addLayout(save_layout)

        return tab

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

    def load_array(self, array: np.ndarray):
        """加载数组并进行语义推断"""
        self.current_array = array

        # 语义推断
        self.semantic_info = SemanticInference.infer(array)
        self.current_semantic = self.semantic_info['semantic']

        # 显示语义信息
        semantic_text = f"推断语义: {self.current_semantic.value}\n"
        semantic_text += f"原因: {self.semantic_info['reason']}\n"
        semantic_text += f"置信度: {self.semantic_info['confidence']}\n"
        if self.semantic_info['suggestions']:
            suggestions = [s.value for s in self.semantic_info['suggestions']]
            semantic_text += f"其他可能: {', '.join(suggestions)}"
        self.semantic_info_text.setText(semantic_text)

        # 计算并显示统计信息
        stats = ArrayStats.compute_stats(array)
        stats_text = ArrayStats.format_stats(stats)
        self.stats_text.setText(stats_text)

        # 更新语义控制组件
        self.semantic_control.set_array_info(array.shape, self.semantic_info)

        # 显示预览
        self.update_preview(array)

        # 清空图表
        self.figure.clear()
        self.canvas.draw()

        # 特殊提示：检查是否有单调列（用于表格数据）
        if self.current_semantic == DataSemantics.TABULAR_2D and array.shape[1] > 0:
            first_col = array[:, 0]
            mono_info = SemanticInference.check_monotonic(first_col)
            if mono_info['is_monotonic']:
                QMessageBox.information(
                    self,
                    "提示",
                    f"检测到列 0 {mono_info['suggestion']}\n建议将其作为 X 轴使用。"
                )

    def update_preview(self, array: np.ndarray):
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

    def on_semantic_changed(self, semantic_text: str):
        """语义类型改变"""
        # 用户手动切换了语义类型
        pass

    def on_plot_requested_new(self, params: dict):
        """处理新的绘图请求"""
        if self.current_array is None:
            QMessageBox.warning(self, "警告", "未加载数组")
            return

        plot_type = params['plot_type']
        semantic = params['semantic']

        try:
            # 根据语义和图表类型调用相应的绘图方法
            if semantic == DataSemantics.SEQUENCE_1D:
                result = self.plot_sequence(plot_type)

            elif semantic == DataSemantics.TABULAR_2D:
                result = self.plot_tabular(plot_type, params)

            elif semantic == DataSemantics.IMAGE_2D:
                result = self.plot_image(plot_type)

            elif semantic == DataSemantics.VOLUME_3D:
                result = self.plot_volume(plot_type, params)

            elif semantic == DataSemantics.MULTICHANNEL_3D:
                result = self.plot_multichannel(plot_type, params)

            elif semantic == DataSemantics.VOLUME_4D:
                result = self.plot_volume_4d(plot_type, params)

            else:
                QMessageBox.warning(self, "错误", "不支持的数据语义")
                return

            if not result['success']:
                QMessageBox.critical(self, "绘图失败", result['error'])
                return

            self.canvas.draw()

        except Exception as e:
            QMessageBox.critical(self, "错误", f"绘图过程出错: {str(e)}")

    def plot_sequence(self, plot_type: str) -> dict:
        """绘制一维序列"""
        if plot_type == "折线图":
            return SemanticVisualizer.plot_sequence_line(self.current_array, self.figure)
        elif plot_type == "直方图":
            return SemanticVisualizer.plot_sequence_histogram(self.current_array, self.figure)
        else:
            return {'success': False, 'error': f'未知图表类型: {plot_type}'}

    def plot_tabular(self, plot_type: str, params: dict) -> dict:
        """绘制表格数据"""
        if plot_type == "多折线图":
            x_col = params.get('x_col')
            y_cols = params.get('y_cols')
            invert_y = params.get('invert_y', False)
            return SemanticVisualizer.plot_tabular_multiline(
                self.current_array, self.figure, x_col, y_cols, invert_y
            )

        elif plot_type == "单列直方图":
            col_idx = params.get('col_idx', 0)
            return SemanticVisualizer.plot_tabular_column_histogram(
                self.current_array, self.figure, col_idx
            )

        elif plot_type == "散点图":
            x_col = params.get('x_col', 0)
            y_col = params.get('y_col', 1)
            return SemanticVisualizer.plot_tabular_scatter(
                self.current_array, self.figure, x_col, y_col
            )

        elif plot_type == "相关性热力图":
            return SemanticVisualizer.plot_tabular_correlation(
                self.current_array, self.figure
            )

        else:
            return {'success': False, 'error': f'未知图表类型: {plot_type}'}

    def plot_image(self, plot_type: str) -> dict:
        """绘制图像数据"""
        if plot_type in ["热力图", "图像显示"]:
            return SemanticVisualizer.plot_image_heatmap(self.current_array, self.figure)
        else:
            return {'success': False, 'error': f'未知图表类型: {plot_type}'}

    def plot_volume(self, plot_type: str, params: dict) -> dict:
        """绘制三维体数据"""
        if plot_type == "切片热力图":
            axis = params.get('axis', 0)
            index = params.get('index', 0)
            return SemanticVisualizer.plot_volume_slice(
                self.current_array, self.figure, axis, index
            )

        elif plot_type == "投影图":
            axis = params.get('axis', 0)
            method = params.get('method', 'mean')
            return SemanticVisualizer.plot_volume_projection(
                self.current_array, self.figure, axis, method
            )

        else:
            return {'success': False, 'error': f'未知图表类型: {plot_type}'}

    def plot_multichannel(self, plot_type: str, params: dict) -> dict:
        """绘制多通道 3D 数据"""
        channel_axis = params.get('channel_axis', 2)
        channel_index = params.get('channel_index', 0)

        # 先选择通道
        result = ArrayProjection.select_channel(
            self.current_array, channel_axis, channel_index
        )

        if not result['success']:
            return result

        channel_data = result['array']

        # 然后绘制 2D 图像
        if plot_type in ["通道热力图", "通道图像"]:
            return SemanticVisualizer.plot_image_heatmap(channel_data, self.figure)
        else:
            return {'success': False, 'error': f'未知图表类型: {plot_type}'}

    def plot_volume_4d(self, plot_type: str, params: dict) -> dict:
        """绘制四维数据"""
        channel_axis = params.get('channel_axis', 3)
        channel_index = params.get('channel_index', 0)

        # 先选择通道
        result = ArrayProjection.select_channel(
            self.current_array, channel_axis, channel_index
        )

        if not result['success']:
            return result

        volume_data = result['array']

        # 然后按 3D 体数据处理
        if plot_type == "切片热力图":
            axis = params.get('axis', 0)
            index = params.get('index', 0)
            return SemanticVisualizer.plot_volume_slice(
                volume_data, self.figure, axis, index
            )

        elif plot_type == "投影图":
            axis = params.get('axis', 0)
            method = params.get('method', 'mean')
            return SemanticVisualizer.plot_volume_projection(
                volume_data, self.figure, axis, method
            )

        else:
            return {'success': False, 'error': f'未知图表类型: {plot_type}'}

    def export_csv(self):
        """导出为 CSV"""
        if self.current_array is None:
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
            df = PreviewHelper.array_to_dataframe(self.current_array)
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
