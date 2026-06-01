"""
NPY/NPZ/Zarr 数据查看器应用入口。
"""
import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QTextEdit, QTableWidget, QTableWidgetItem,
    QPushButton, QFileDialog, QMessageBox,
    QLabel, QProgressBar, QTabWidget
)
from PySide6.QtCore import Qt, QSettings
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import pandas as pd
import numpy as np

from npy_npz_viewer.core.loaders import ArrayLoader
from npy_npz_viewer.core.array_session import ArraySession
from npy_npz_viewer.core.array_compute import ArrayComputeService
from npy_npz_viewer.core.stats import ArrayStats
from npy_npz_viewer.core.semantic_inference import SemanticInference
from npy_npz_viewer.core.data_semantics import DataSemantics
from npy_npz_viewer.core.projection import ArrayProjection
from npy_npz_viewer.core.visualization_new import SemanticVisualizer
from npy_npz_viewer.core.visualization_3d import Visualizer3D
from npy_npz_viewer.utils.helpers import PreviewHelper
from npy_npz_viewer.utils.large_data_optimizer import LargeDataOptimizer
from npy_npz_viewer.ui.optimized_left_panel import OptimizedLeftPanel
from npy_npz_viewer.ui.custom_preview import CustomPreviewWidget
from npy_npz_viewer.ui.task_runner import TaskRunner
from npy_npz_viewer.logging_config import configure_logging


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.loader = ArrayLoader()
        self.session = ArraySession()
        self.settings = QSettings("DataViewer", "NPYNPZViewer")
        self.task_runner = TaskRunner(self)
        self.task_runner.task_started.connect(self.on_task_started)
        self.task_runner.task_progress.connect(self.on_task_progress)
        self.task_runner.task_finished.connect(self.on_task_finished)
        self.original_array = None
        self.filtered_array = None
        self.current_array = None
        self.axis_index_maps = {}
        self.current_semantic = None
        self.semantic_info = None
        self.preview_start = 0
        self.preview_end = 1000
        self.preview_mode = "auto"
        self.view_refresh_id = 0
        self.file_load_id = 0
        self.pending_payloads = {}
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("NPY/NPZ/Zarr 数据查看器")
        self.setGeometry(100, 100, 1600, 1000)
        self.setAcceptDrops(True)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # 使用优化后的左侧面板
        self.left_panel = OptimizedLeftPanel()
        self.left_panel.file_opened.connect(self.on_file_opened)
        self.left_panel.npz_key_selected.connect(self.on_npz_key_selected)
        self.left_panel.dimension_filter_applied.connect(self.on_dimension_filter_applied)
        self.left_panel.slice_applied.connect(self.on_slice_applied)
        self.left_panel.semantic_changed.connect(self.on_semantic_changed)
        self.left_panel.plot_requested.connect(self.on_plot_requested)
        self.left_panel.set_recent_files(self.settings.value("recent_files", [], type=list))
        splitter.addWidget(self.left_panel)

        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)

        # 设置左右面板的初始宽度比例（左侧更宽）
        splitter.setSizes([300, 1300])  # 左侧 500px，右侧 1100px

    def get_supported_drop_file(self, event):
        """从拖拽事件中提取第一个支持的本地文件路径。"""
        if not event.mimeData().hasUrls():
            return None

        for url in event.mimeData().urls():
            if not url.isLocalFile():
                continue
            file_path = url.toLocalFile()
            if file_path.lower().endswith(('.npy', '.npz', '.zarr')):
                return file_path
        return None

    def dragEnterEvent(self, event):
        """接受 NPY/NPZ 文件拖入。"""
        if self.get_supported_drop_file(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        """拖动过程中保持可投放状态。"""
        if self.get_supported_drop_file(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        """拖放文件后加载第一个支持的 NPY/NPZ 文件。"""
        file_path = self.get_supported_drop_file(event)
        if not file_path:
            QMessageBox.warning(self, "不支持的文件", "请拖入 .npy、.npz 文件或 .zarr 目录")
            event.ignore()
            return

        supported_count = sum(
            1
            for url in event.mimeData().urls()
            if url.isLocalFile() and url.toLocalFile().lower().endswith(('.npy', '.npz', '.zarr'))
        )
        if supported_count > 1:
            QMessageBox.information(self, "拖拽文件", "一次只加载一个文件，已加载第一个支持的文件。")

        event.acceptProposedAction()
        self.on_file_opened(file_path)

    def closeEvent(self, event):
        """关闭窗口前先让后台线程安全收尾。"""
        self.task_runner.cancel_all()
        if not self.task_runner.wait_for_all(timeout_ms=5000):
            QMessageBox.warning(
                self,
                "后台任务仍在运行",
                "当前仍有后台任务没有结束。请稍等任务完成或取消后再关闭窗口。",
            )
            event.ignore()
            return
        event.accept()

    def create_right_panel(self) -> QWidget:
        """创建右侧面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        tabs = QTabWidget()

        stats_tab = self.create_stats_tab()
        tabs.addTab(stats_tab, "统计信息")

        preview_tab = self.create_preview_tab()
        tabs.addTab(preview_tab, "数据预览")

        viz_tab = self.create_viz_tab()
        tabs.addTab(viz_tab, "可视化")

        layout.addWidget(tabs)
        return panel

    def create_stats_tab(self) -> QWidget:
        """创建统计信息标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.data_size_warning = QLabel("")
        self.data_size_warning.setWordWrap(True)
        self.data_size_warning.setStyleSheet(
            "background-color: #fff3cd; padding: 10px; border-radius: 5px;"
        )
        self.data_size_warning.hide()
        layout.addWidget(self.data_size_warning)

        self.semantic_info_text = QTextEdit()
        self.semantic_info_text.setReadOnly(True)
        self.semantic_info_text.setMaximumHeight(100)
        layout.addWidget(QLabel("数据语义:"))
        layout.addWidget(self.semantic_info_text)

        task_layout = QHBoxLayout()
        self.task_status_label = QLabel("就绪")
        task_layout.addWidget(self.task_status_label)
        self.task_progress = QProgressBar()
        self.task_progress.setRange(0, 100)
        self.task_progress.setValue(0)
        task_layout.addWidget(self.task_progress)
        self.cancel_task_btn = QPushButton("取消任务")
        self.cancel_task_btn.clicked.connect(self.task_runner.cancel_current)
        self.cancel_task_btn.setEnabled(False)
        task_layout.addWidget(self.cancel_task_btn)
        layout.addLayout(task_layout)

        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        layout.addWidget(QLabel("数组统计:"))
        layout.addWidget(self.stats_text)

        return tab

    def create_preview_tab(self) -> QWidget:
        """创建数据预览标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # 自定义预览范围控件
        self.custom_preview = CustomPreviewWidget()
        self.custom_preview.preview_changed.connect(self.on_preview_changed)
        layout.addWidget(self.custom_preview)

        self.preview_info_label = QLabel("")
        layout.addWidget(self.preview_info_label)

        self.preview_table = QTableWidget()
        layout.addWidget(self.preview_table)

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

        self.downsample_info = QLabel("")
        self.downsample_info.setWordWrap(True)
        self.downsample_info.setStyleSheet(
            "background-color: #d1ecf1; padding: 8px; border-radius: 5px;"
        )
        self.downsample_info.hide()
        layout.addWidget(self.downsample_info)

        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(self.canvas)

        save_layout = QHBoxLayout()
        self.save_plot_btn = QPushButton("保存图表为 PNG")
        self.save_plot_btn.clicked.connect(self.save_plot)
        save_layout.addWidget(self.save_plot_btn)
        save_layout.addStretch()
        layout.addLayout(save_layout)

        return tab

    def remember_recent_file(self, file_path: str):
        """保存最近打开文件。"""
        recent = self.settings.value("recent_files", [], type=list)
        recent = [p for p in recent if p != file_path]
        recent.insert(0, file_path)
        self.settings.setValue("recent_files", recent[:10])
        if hasattr(self, "left_panel"):
            self.left_panel.set_recent_files(recent[:10])

    def on_task_started(self, name: str):
        """后台任务开始。"""
        self.task_status_label.setText(f"{name}...")
        self.task_progress.setValue(0)
        self.cancel_task_btn.setEnabled(True)

    def on_task_progress(self, name: str, value: int, message: str):
        """后台任务进度。"""
        self.task_status_label.setText(f"{name}: {message}")
        self.task_progress.setValue(value)

    def on_task_finished(self, name: str, result):
        """后台任务结束。"""
        self.cancel_task_btn.setEnabled(False)
        self.task_progress.setValue(100 if result.success else 0)
        if not result.success:
            if result.metadata.get("cancelled"):
                self.task_status_label.setText(f"{name}已取消")
                return
            self.task_status_label.setText(f"{name}失败")
            QMessageBox.warning(self, "后台任务失败", result.error or "未知错误")
            return

        self.task_status_label.setText(f"{name}完成 ({result.elapsed_ms:.0f} ms)")
        if name == "刷新视图":
            payload = result.data
            metadata = payload.get("metadata", {})
            if metadata.get("refresh_id") != self.view_refresh_id:
                return
            if payload.get("stats_text") is not None:
                self.stats_text.setText(payload["stats_text"])
            self.apply_preview_dataframe(
                payload.get("df"),
                payload.get("preview_info", ""),
                metadata.get("array_ndim", 1),
            )
        elif name == "加载文件":
            payload = result.data
            if payload.get('load_id') != self.file_load_id:
                return
            if not payload.get('success'):
                QMessageBox.critical(self, "错误", payload.get('error', '文件加载失败'))
                return

            self.loader = payload['loader']
            file_path = payload.get('file_path', '')
            self.left_panel.set_file_info(f"文件: {file_path}\n类型: {payload['file_type'].upper()}")
            if payload['file_type'] == 'npy':
                self.left_panel.set_npz_keys([])
                self.load_array(payload['array'], payload.get('handle'))
            else:
                self.left_panel.set_npz_keys(payload['keys'])
                array = self.loader.get_current_array()
                if array is not None:
                    self.load_array(array, self.loader.get_current_handle())
        elif name == "切换数组":
            payload = result.data
            if payload.get('load_id') != self.file_load_id:
                return
            if not payload.get('success'):
                QMessageBox.critical(self, "错误", payload.get('error', '切换数组失败'))
                return
            self.loader = payload['loader']
            self.load_array(payload['array'], payload.get('handle'))

    def on_file_opened(self, file_path: str):
        """文件打开回调"""
        self.remember_recent_file(file_path)
        self.file_load_id += 1
        load_id = self.file_load_id
        self.left_panel.set_file_info(f"文件: {file_path}\n状态: 正在后台加载...")

        def load_payload(worker):
            worker.progress.emit("加载文件", 10, "打开文件")
            loader = ArrayLoader()
            payload = loader.load_file(file_path)
            payload['file_path'] = file_path
            payload['load_id'] = load_id
            payload['loader'] = loader
            worker.progress.emit("加载文件", 90, "准备数组视图")
            return payload

        self.task_runner.run_task("加载文件", load_payload)

    def on_npz_key_selected(self, key: str):
        """NPZ 键选择回调"""
        self.file_load_id += 1
        load_id = self.file_load_id
        loader = self.loader

        def switch_payload(worker):
            worker.progress.emit("切换数组", 20, f"读取 {key}")
            payload = loader.switch_npz_key(key)
            payload['load_id'] = load_id
            payload['loader'] = loader
            worker.progress.emit("切换数组", 90, "准备数组视图")
            return payload

        self.task_runner.run_task("切换数组", switch_payload)

    def load_array(self, array: np.ndarray, handle=None):
        """加载数组"""
        self.session.load_array(
            array,
            source_path=self.loader.current_file,
            npz_key=self.loader.current_key,
            handle=handle,
        )
        self.original_array = array
        self.filtered_array = array
        self.current_array = array
        self.axis_index_maps = {}
        self.preview_start = 0
        self.preview_end = 1000

        # 数据大小警告
        warning = LargeDataOptimizer.format_data_size_warning(array)
        if warning:
            self.data_size_warning.setText(warning)
            self.data_size_warning.show()
        else:
            self.data_size_warning.hide()

        # 设置左侧面板
        self.left_panel.set_array(array)

        self.refresh_current_array(array)

        # 单调列检测
        if (
            self.current_semantic == DataSemantics.TABULAR_2D
            and array.shape[1] > 0
            and not ArrayComputeService.is_lazy_array(array)
        ):
            first_col = array[:, 0]
            mono_info = SemanticInference.check_monotonic(first_col)
            if mono_info['is_monotonic']:
                QMessageBox.information(
                    self, "提示",
                    f"检测到列 0 {mono_info['suggestion']}\n建议将其作为 X 轴使用。"
                )

    def refresh_current_array(self, array: np.ndarray):
        """刷新统计、语义、预览和图表状态。"""
        self.current_array = array

        # 语义推断
        self.semantic_info = SemanticInference.infer(array)
        self.current_semantic = self.semantic_info['semantic']

        semantic_text = f"推断语义: {self.current_semantic.value}\n"
        semantic_text += f"原因: {self.semantic_info['reason']}\n"
        semantic_text += f"置信度: {self.semantic_info['confidence']}\n"
        if self.semantic_info['suggestions']:
            suggestions = [s.value for s in self.semantic_info['suggestions']]
            semantic_text += f"其他可能: {', '.join(suggestions)}"
        self.semantic_info_text.setText(semantic_text)

        # 更新左侧面板的语义控制
        self.left_panel.set_array_info(array.shape, self.semantic_info)

        # 清空图表
        self.figure.clear()
        self.canvas.draw()
        self.downsample_info.hide()

        self.schedule_view_refresh(array, include_stats=True)

    def on_dimension_filter_applied(self, result: dict):
        """维度筛选应用回调。"""
        if not result['success']:
            QMessageBox.warning(self, "维度筛选失败", result['error'])
            return

        self.filtered_array = result['array']
        self.axis_index_maps = result.get('axis_index_maps', {})
        self.session.filtered_array = self.filtered_array
        self.session.current_array = self.filtered_array
        self.session.spec.axis_maps = self.axis_index_maps
        self.left_panel.set_slice_array(self.filtered_array)
        self.refresh_current_array(self.filtered_array)

    def on_slice_applied(self, sliced_array: np.ndarray):
        """切片应用回调"""
        self.session.set_current_array(sliced_array)
        self.refresh_current_array(sliced_array)

    def get_axis_index_map(self, axis: int):
        """获取筛选后轴索引到原始轴索引的映射。"""
        if axis not in self.axis_index_maps:
            return None
        return self.axis_index_maps[axis]

    def format_column_label(self, col_idx: int) -> str:
        """格式化二维表格列标签，尽量保留原始列索引。"""
        axis_map = self.get_axis_index_map(1)
        if axis_map is None or col_idx >= len(axis_map):
            return f"列 {col_idx}"
        original_idx = axis_map[col_idx]
        if original_idx == col_idx:
            return f"列 {col_idx}"
        return f"列 {col_idx} (原列 {original_idx})"

    def get_column_labels(self, cols: int):
        return [self.format_column_label(i) for i in range(cols)]

    def schedule_view_refresh(self, array: np.ndarray, include_stats: bool):
        """后台生成统计和预览。"""
        self.view_refresh_id += 1
        refresh_id = self.view_refresh_id
        rows = array.size if array.ndim > 2 else (array.shape[0] if array.ndim >= 1 else 1)
        self.custom_preview.set_array_size(rows, reset_range=include_stats)
        if include_stats:
            self.preview_start, self.preview_end = self.custom_preview.get_range()
        start = min(self.preview_start, max(0, rows - 1))
        end = min(max(self.preview_end, start + 1), rows)
        mode = self.preview_mode

        if include_stats:
            self.stats_text.setText("正在计算统计信息...")
        self.preview_info_label.setText("正在生成预览...")
        self.preview_table.clear()

        def build_payload(worker):
            worker.progress.emit("刷新视图", 15, "生成预览")
            df, info = ArrayComputeService.build_preview(array, mode, start, end)
            metadata = {
                "refresh_id": refresh_id,
                "preview_info": info,
                "preview_ndim": 2 if df is not None and len(df.columns) > 2 else 1,
                "array_ndim": array.ndim,
                "array_shape": array.shape,
            }
            stats_text = None
            sampled = False
            if include_stats:
                worker.progress.emit("刷新视图", 45, "计算统计")
                stats = ArrayComputeService.compute_stats(array)
                stats_text = ArrayStats.format_stats(stats)
                sampled = stats.get("sampled", False)
            worker.progress.emit("刷新视图", 90, "准备界面数据")
            return {
                "df": df,
                "preview_info": info,
                "stats_text": stats_text,
                "sampled": sampled,
                "metadata": metadata,
            }

        self.task_runner.run_task("刷新视图", build_payload)

    def apply_preview_dataframe(self, df, info: str, array_ndim: int):
        """把后台生成的 DataFrame 显示到预览表。"""
        self.preview_info_label.setText(info)
        if df is None:
            self.preview_table.clear()
            return
        self.preview_table.setRowCount(len(df))
        self.preview_table.setColumnCount(len(df.columns))
        if array_ndim == 2 and len(df.columns) <= self.current_array.shape[1]:
            headers = self.get_column_labels(len(df.columns))
        else:
            headers = [str(c) for c in df.columns]
        self.preview_table.setHorizontalHeaderLabels(headers)

        for i in range(len(df)):
            for j in range(len(df.columns)):
                value = df.iloc[i, j]
                item = QTableWidgetItem(str(value))
                self.preview_table.setItem(i, j, item)

    def on_semantic_changed(self, semantic_text: str):
        """语义改变回调"""
        pass

    def on_preview_range_changed(self, start: int, end: int):
        """预览范围改变回调"""
        self.on_preview_changed(self.preview_mode, start, end)

    def on_preview_changed(self, mode: str, start: int, end: int):
        """预览模式或范围改变回调。"""
        self.preview_mode = mode
        self.preview_start = start
        self.preview_end = end
        if self.current_array is not None:
            self.schedule_view_refresh(self.current_array, include_stats=False)

    def on_plot_requested(self, params: dict):
        """绘图请求回调"""
        if self.current_array is None:
            QMessageBox.warning(self, "警告", "未加载数组")
            return

        plot_type = params['plot_type']
        if plot_type == "__quick_2d__":
            self.make_quick_2d_view()
            return

        semantic = params['semantic']

        self.downsample_info.hide()

        try:
            if semantic == DataSemantics.SEQUENCE_1D:
                result = self.plot_sequence(plot_type)
            elif semantic == DataSemantics.TABULAR_2D:
                result = self.plot_tabular(plot_type, params)
            elif semantic == DataSemantics.IMAGE_2D:
                result = self.plot_image(plot_type, params)
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
        array = self.current_array

        array, info = ArrayComputeService.downsample_1d_for_plot(array)
        if info:
            self.downsample_info.setText(info)
            self.downsample_info.show()

        if plot_type == "折线图":
            return SemanticVisualizer.plot_sequence_line(array, self.figure)
        elif plot_type == "直方图":
            return SemanticVisualizer.plot_sequence_histogram(array, self.figure)
        else:
            return {'success': False, 'error': f'未知图表类型: {plot_type}'}

    def plot_tabular(self, plot_type: str, params: dict) -> dict:
        """绘制表格数据"""
        array = self.current_array

        if plot_type == "多折线图":
            x_col = params.get('x_col')
            y_cols = params.get('y_cols')
            invert_y = params.get('invert_y', False)

            x_data = array[:, x_col] if x_col is not None else None
            downsampled, x_downsampled, info = ArrayComputeService.downsample_2d_for_plot(
                array, x_data
            )

            if info:
                self.downsample_info.setText(info)
                self.downsample_info.show()

            return SemanticVisualizer.plot_tabular_multiline(
                downsampled, self.figure, x_col, y_cols, invert_y,
                column_labels=self.get_column_labels(array.shape[1])
            )

        elif plot_type == "单列直方图":
            col_idx = params.get('col_idx', 0)
            column_data, info = ArrayComputeService.downsample_1d_for_plot(array[:, col_idx])
            if info:
                self.downsample_info.setText(info)
                self.downsample_info.show()
            plot_array = column_data.reshape(-1, 1)
            return SemanticVisualizer.plot_tabular_column_histogram(
                plot_array, self.figure, 0,
                column_label=self.format_column_label(col_idx)
            )

        elif plot_type == "散点图":
            x_col = params.get('x_col', 0)
            y_col = params.get('y_col', 1)

            x_data = array[:, x_col]
            y_data = array[:, y_col]
            x_down, y_down, info = ArrayComputeService.downsample_scatter(x_data, y_data)

            if info:
                self.downsample_info.setText(info)
                self.downsample_info.show()

            temp_array = np.column_stack([x_down, y_down])
            return SemanticVisualizer.plot_tabular_scatter(
                temp_array, self.figure, 0, 1,
                x_label=self.format_column_label(x_col),
                y_label=self.format_column_label(y_col)
            )

        elif plot_type == "相关性热力图":
            plot_array, _, info = ArrayComputeService.downsample_2d_for_plot(array)
            if info:
                self.downsample_info.setText(info)
                self.downsample_info.show()
            return SemanticVisualizer.plot_tabular_correlation(
                plot_array, self.figure, column_labels=self.get_column_labels(array.shape[1])
            )

        else:
            return {'success': False, 'error': f'未知图表类型: {plot_type}'}

    def plot_image(self, plot_type: str, params: dict = None) -> dict:
        """绘制图像数据"""
        array = self.current_array
        params = params or {}

        if array.ndim == 2:
            downsampled, info = ArrayComputeService.downsample_heatmap(array)
            if info:
                self.downsample_info.setText(info)
                self.downsample_info.show()
            array = downsampled

        if plot_type in ["热力图", "图像显示"]:
            return SemanticVisualizer.plot_image_heatmap(
                array,
                self.figure,
                swap_axes=params.get('swap_axes', False),
                x_order=params.get('x_order', 'asc'),
                y_order=params.get('y_order', 'asc'),
                cmap=params.get('cmap', 'viridis'),
                aspect=params.get('aspect', 'auto'),
            )
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
                self.current_array, self.figure, axis, method,
                quality=params.get('quality', 'fast')
            )
        elif plot_type == "3D散点图":
            return Visualizer3D.plot_3d_scatter(
                self.current_array, self.figure
            )
        elif plot_type == "3D表面图":
            axis = params.get('axis', 0)
            index = params.get('index')
            return Visualizer3D.plot_3d_surface(
                self.current_array, self.figure, axis, index
            )
        elif plot_type == "3D线框图":
            axis = params.get('axis', 0)
            index = params.get('index')
            return Visualizer3D.plot_3d_wireframe(
                self.current_array, self.figure, axis, index
            )
        elif plot_type == "3D等高线图":
            axis = params.get('axis', 0)
            index = params.get('index')
            return Visualizer3D.plot_3d_contour(
                self.current_array, self.figure, axis, index
            )
        elif plot_type == "3D体素图":
            return Visualizer3D.plot_3d_voxel(
                self.current_array, self.figure
            )
        elif plot_type == "3D切片堆叠图":
            axis = params.get('axis', 0)
            return Visualizer3D.plot_3d_slice_stack(
                self.current_array, self.figure, axis
            )
        else:
            return {'success': False, 'error': f'未知图表类型: {plot_type}'}

    def plot_multichannel(self, plot_type: str, params: dict) -> dict:
        """绘制多通道 3D 数据"""
        channel_axis = params.get('channel_axis', 2)
        channel_index = params.get('channel_index', 0)

        result = ArrayProjection.select_channel(
            self.current_array, channel_axis, channel_index
        )

        if not result['success']:
            return result

        channel_data = result['array']

        if plot_type in ["通道热力图", "通道图像"]:
            return SemanticVisualizer.plot_image_heatmap(channel_data, self.figure)
        else:
            return {'success': False, 'error': f'未知图表类型: {plot_type}'}

    def plot_volume_4d(self, plot_type: str, params: dict) -> dict:
        """绘制四维数据"""
        channel_axis = params.get('channel_axis', 3)
        channel_index = params.get('channel_index', 0)

        result = ArrayProjection.select_channel(
            self.current_array, channel_axis, channel_index
        )

        if not result['success']:
            return result

        volume_data = result['array']

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
                volume_data, self.figure, axis, method,
                quality=params.get('quality', 'fast')
            )
        else:
            return {'success': False, 'error': f'未知图表类型: {plot_type}'}

    def export_csv(self):
        """导出为 CSV"""
        if self.current_array is None:
            QMessageBox.warning(self, "警告", "未加载数组")
            return

        if self.current_array.ndim > 2:
            QMessageBox.warning(self, "导出受限", "高维数组请先通过维度筛选或切片转换为 1D/2D 视图后再导出。")
            return

        if self.current_array.size > 2_000_000:
            reply = QMessageBox.question(
                self,
                "导出大数据",
                f"当前视图包含 {self.current_array.size:,} 个元素，导出 CSV 可能较慢。是否继续？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存 CSV 文件", "", "CSV 文件 (*.csv)"
        )

        if not file_path:
            return

        try:
            export_array = ArrayComputeService.to_numpy(self.current_array)
            df = PreviewHelper.array_to_dataframe(export_array)
            if df is None:
                QMessageBox.critical(self, "错误", "无法转换为表格格式")
                return

            if export_array.ndim == 2:
                df.columns = self.get_column_labels(export_array.shape[1])

            df.to_csv(file_path, index=False)
            QMessageBox.information(self, "成功", f"已导出到: {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出失败: {str(e)}")

    def save_plot(self):
        """保存图表"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存图表", "", "PNG 图像 (*.png)"
        )

        if not file_path:
            return

        try:
            self.figure.savefig(file_path, dpi=300, bbox_inches='tight')
            QMessageBox.information(self, "成功", f"图表已保存到: {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存失败: {str(e)}")

    def make_quick_2d_view(self):
        """快速把当前高维视图变成可绘制 2D 视图。"""
        array = self.current_array
        if array is None:
            return

        view = array.squeeze() if hasattr(array, "squeeze") else np.squeeze(array)
        fixed = []
        while view.ndim > 2:
            axis = view.ndim - 1
            index = view.shape[axis] // 2
            view = view.take(index, axis=axis) if hasattr(view, "take") else np.take(view, index, axis=axis)
            fixed.append((axis, index))

        if view.ndim == 1:
            view = view.reshape(-1, 1)

        self.session.set_current_array(view)
        self.current_array = view
        self.filtered_array = view
        self.left_panel.set_slice_array(view)
        detail = "，".join(f"轴{axis}=中间索引{index}" for axis, index in fixed)
        if detail:
            self.downsample_info.setText(f"已生成 2D 快速视图：{view.shape}（{detail}）")
        else:
            self.downsample_info.setText(f"已生成 2D 快速视图：{view.shape}")
        self.downsample_info.show()
        self.refresh_current_array(view)


def main():
    """主函数"""
    configure_logging()
    app = QApplication(sys.argv)
    app.setApplicationName("NPY/NPZ/Zarr 数据查看器")
    app.setOrganizationName("DataViewer")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
