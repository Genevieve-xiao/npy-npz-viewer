"""
重构后的可视化模块
基于数据语义的可视化系统
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Dict, List, Optional
from core.data_semantics import DataSemantics

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class SemanticVisualizer:
    """基于语义的可视化器"""

    # ==================== 一维序列可视化 ====================

    @staticmethod
    def plot_sequence_line(array: np.ndarray, figure: Figure,
                          x_data: Optional[np.ndarray] = None) -> Dict:
        """绘制一维序列折线图"""
        if array.ndim != 1:
            return {'success': False, 'error': '需要 1D 数组'}

        if not np.issubdtype(array.dtype, np.number):
            return {'success': False, 'error': '需要数值类型数组'}

        try:
            figure.clear()
            ax = figure.add_subplot(111)

            if x_data is not None and len(x_data) == len(array):
                ax.plot(x_data, array)
                ax.set_xlabel('X 轴')
            else:
                ax.plot(array)
                ax.set_xlabel('索引')

            ax.set_ylabel('值')
            ax.set_title('一维序列折线图')
            ax.grid(True, alpha=0.3)
            figure.tight_layout()
            return {'success': True}
        except Exception as e:
            return {'success': False, 'error': f'绘图失败: {str(e)}'}

    @staticmethod
    def plot_sequence_histogram(array: np.ndarray, figure: Figure, bins: int = 50) -> Dict:
        """绘制一维序列直方图"""
        if array.ndim != 1:
            return {'success': False, 'error': '需要 1D 数组'}

        if not np.issubdtype(array.dtype, np.number):
            return {'success': False, 'error': '需要数值类型数组'}

        try:
            valid_data = array[np.isfinite(array)]
            if len(valid_data) == 0:
                return {'success': False, 'error': '没有有效数据（全为 NaN/Inf）'}

            figure.clear()
            ax = figure.add_subplot(111)
            ax.hist(valid_data, bins=bins, edgecolor='black', alpha=0.7)
            ax.set_xlabel('值')
            ax.set_ylabel('频数')
            ax.set_title('一维序列直方图')
            ax.grid(True, alpha=0.3, axis='y')
            figure.tight_layout()
            return {'success': True}
        except Exception as e:
            return {'success': False, 'error': f'绘图失败: {str(e)}'}

    # ==================== 二维表格可视化 ====================

    @staticmethod
    def plot_tabular_multiline(array: np.ndarray, figure: Figure,
                               x_col: Optional[int] = None,
                               y_cols: Optional[List[int]] = None,
                               invert_y: bool = False,
                               column_labels: Optional[List[str]] = None) -> Dict:
        """
        绘制二维表格多折线图

        Args:
            array: (N, C) 数组
            x_col: x 轴列索引，None 表示使用行索引
            y_cols: y 轴列索引列表，None 表示使用所有列（除 x_col）
            invert_y: 是否反转 y 轴（用于深度数据）
        """
        if array.ndim != 2:
            return {'success': False, 'error': '需要 2D 数组'}

        if not np.issubdtype(array.dtype, np.number):
            return {'success': False, 'error': '需要数值类型数组'}

        try:
            rows, cols = array.shape

            # 确定 x 轴数据
            if x_col is not None:
                if x_col < 0 or x_col >= cols:
                    return {'success': False, 'error': f'x 列索引 {x_col} 超出范围'}
                x_data = array[:, x_col]
            else:
                x_data = np.arange(rows)

            # 确定 y 轴列
            if y_cols is None:
                y_cols = [i for i in range(cols) if i != x_col]

            if len(y_cols) == 0:
                return {'success': False, 'error': '没有可绘制的 y 列'}

            # 绘图
            figure.clear()
            ax = figure.add_subplot(111)

            for col_idx in y_cols:
                if col_idx < 0 or col_idx >= cols:
                    continue
                y_data = array[:, col_idx]
                label = column_labels[col_idx] if column_labels and col_idx < len(column_labels) else f'列 {col_idx}'
                ax.plot(x_data, y_data, label=label, alpha=0.8)

            if x_col is not None and column_labels and x_col < len(column_labels):
                ax.set_xlabel(column_labels[x_col])
            else:
                ax.set_xlabel('X 轴' if x_col is not None else '行索引')
            ax.set_ylabel('值')
            ax.set_title('多属性折线图')
            ax.legend()
            ax.grid(True, alpha=0.3)

            if invert_y:
                ax.invert_yaxis()

            figure.tight_layout()
            return {'success': True}
        except Exception as e:
            return {'success': False, 'error': f'绘图失败: {str(e)}'}

    @staticmethod
    def plot_tabular_column_histogram(array: np.ndarray, figure: Figure,
                                     col_idx: int, bins: int = 50,
                                     column_label: Optional[str] = None) -> Dict:
        """绘制二维表格单列直方图"""
        if array.ndim != 2:
            return {'success': False, 'error': '需要 2D 数组'}

        rows, cols = array.shape
        if col_idx < 0 or col_idx >= cols:
            return {'success': False, 'error': f'列索引 {col_idx} 超出范围 [0, {cols-1}]'}

        column_data = array[:, col_idx]
        valid_data = column_data[np.isfinite(column_data)]

        if len(valid_data) == 0:
            return {'success': False, 'error': '该列没有有效数据'}

        try:
            figure.clear()
            ax = figure.add_subplot(111)
            ax.hist(valid_data, bins=bins, edgecolor='black', alpha=0.7)
            label = column_label or f'列 {col_idx}'
            ax.set_xlabel(label)
            ax.set_ylabel('频数')
            ax.set_title(f'{label} 直方图')
            ax.grid(True, alpha=0.3, axis='y')
            figure.tight_layout()
            return {'success': True}
        except Exception as e:
            return {'success': False, 'error': f'绘图失败: {str(e)}'}

    @staticmethod
    def plot_tabular_scatter(array: np.ndarray, figure: Figure,
                            x_col: int, y_col: int,
                            x_label: Optional[str] = None,
                            y_label: Optional[str] = None) -> Dict:
        """绘制二维表格散点图"""
        if array.ndim != 2:
            return {'success': False, 'error': '需要 2D 数组'}

        rows, cols = array.shape
        if x_col < 0 or x_col >= cols:
            return {'success': False, 'error': f'x 列索引 {x_col} 超出范围'}
        if y_col < 0 or y_col >= cols:
            return {'success': False, 'error': f'y 列索引 {y_col} 超出范围'}

        try:
            x_data = array[:, x_col]
            y_data = array[:, y_col]

            # 过滤无效数据
            valid_mask = np.isfinite(x_data) & np.isfinite(y_data)
            x_valid = x_data[valid_mask]
            y_valid = y_data[valid_mask]

            if len(x_valid) == 0:
                return {'success': False, 'error': '没有有效数据点'}

            figure.clear()
            ax = figure.add_subplot(111)
            ax.scatter(x_valid, y_valid, alpha=0.6, s=20)
            x_name = x_label or f'列 {x_col}'
            y_name = y_label or f'列 {y_col}'
            ax.set_xlabel(x_name)
            ax.set_ylabel(y_name)
            ax.set_title(f'散点图: {x_name} vs {y_name}')
            ax.grid(True, alpha=0.3)
            figure.tight_layout()
            return {'success': True}
        except Exception as e:
            return {'success': False, 'error': f'绘图失败: {str(e)}'}

    @staticmethod
    def plot_tabular_correlation(array: np.ndarray, figure: Figure,
                                 column_labels: Optional[List[str]] = None,
                                 max_cols: int = 200) -> Dict:
        """绘制二维表格相关性热力图"""
        if array.ndim != 2:
            return {'success': False, 'error': '需要 2D 数组'}

        try:
            rows, cols = array.shape
            if cols < 2:
                return {'success': False, 'error': '相关性热力图至少需要 2 列'}
            if cols > max_cols:
                return {
                    'success': False,
                    'error': f'列数过多（{cols} 列），请先用维度筛选减少到 {max_cols} 列以内'
                }
            if not np.issubdtype(array.dtype, np.number):
                return {'success': False, 'error': '需要数值类型数组'}

            finite_rows = np.all(np.isfinite(array), axis=1)
            valid_array = array[finite_rows]
            if valid_array.shape[0] < 2:
                return {'success': False, 'error': '有效行不足，无法计算相关性'}

            # 计算相关系数矩阵
            corr_matrix = np.corrcoef(valid_array.T)

            figure.clear()
            ax = figure.add_subplot(111)
            im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')

            # 设置刻度
            ax.set_xticks(range(cols))
            ax.set_yticks(range(cols))
            labels = column_labels if column_labels and len(column_labels) == cols else [f'列 {i}' for i in range(cols)]
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_yticklabels(labels)

            ax.set_title('列相关性热力图')
            figure.colorbar(im, ax=ax)
            figure.tight_layout()
            return {'success': True}
        except Exception as e:
            return {'success': False, 'error': f'绘图失败: {str(e)}'}

    # ==================== 二维图像可视化 ====================

    @staticmethod
    def plot_image_heatmap(array: np.ndarray, figure: Figure,
                           swap_axes: bool = False,
                           x_order: str = 'asc',
                           y_order: str = 'asc',
                           cmap: str = 'viridis',
                           aspect: str = 'auto') -> Dict:
        """绘制二维图像热力图"""
        if array.ndim != 2:
            return {'success': False, 'error': '需要 2D 数组'}

        if not np.issubdtype(array.dtype, np.number):
            return {'success': False, 'error': '需要数值类型数组'}

        if x_order not in {'asc', 'desc'}:
            return {'success': False, 'error': f'未知 X 轴方向: {x_order}'}
        if y_order not in {'asc', 'desc'}:
            return {'success': False, 'error': f'未知 Y 轴方向: {y_order}'}

        try:
            plot_data = array.T if swap_axes else array
            x_source_axis = 0 if swap_axes else 1
            y_source_axis = 1 if swap_axes else 0

            if x_order == 'desc':
                plot_data = plot_data[:, ::-1]
            if y_order == 'desc':
                plot_data = plot_data[::-1, :]

            x_size = plot_data.shape[1]
            y_size = plot_data.shape[0]
            x_min, x_max = -0.5, x_size - 0.5
            y_min, y_max = -0.5, y_size - 0.5
            extent = [
                x_min if x_order == 'asc' else x_max,
                x_max if x_order == 'asc' else x_min,
                y_min if y_order == 'asc' else y_max,
                y_max if y_order == 'asc' else y_min,
            ]

            figure.clear()
            ax = figure.add_subplot(111)
            im = ax.imshow(
                plot_data,
                aspect=aspect,
                cmap=cmap,
                interpolation='nearest',
                origin='lower',
                extent=extent,
            )
            ax.set_xlabel(f'轴 {x_source_axis} 索引（{"升序" if x_order == "asc" else "降序"}）')
            ax.set_ylabel(f'轴 {y_source_axis} 索引（{"升序" if y_order == "asc" else "降序"}）')
            title_suffix = '，X/Y 已交换' if swap_axes else ''
            ax.set_title(f'二维热力图{title_suffix}')
            figure.colorbar(im, ax=ax)
            figure.tight_layout()
            return {'success': True}
        except Exception as e:
            return {'success': False, 'error': f'绘图失败: {str(e)}'}

    # ==================== 三维体数据可视化 ====================

    @staticmethod
    def plot_volume_slice(array: np.ndarray, figure: Figure,
                         axis: int, index: int) -> Dict:
        """
        绘制三维体数据切片

        Args:
            array: 3D 数组
            axis: 切片轴 (0, 1, 2)
            index: 切片索引
        """
        if array.ndim != 3:
            return {'success': False, 'error': '需要 3D 数组'}

        if axis < 0 or axis >= 3:
            return {'success': False, 'error': f'轴 {axis} 超出范围 [0, 2]'}

        if index < 0 or index >= array.shape[axis]:
            return {'success': False, 'error': f'索引 {index} 超出轴 {axis} 的范围'}

        try:
            # 提取切片
            if axis == 0:
                slice_data = array[index, :, :]
                xlabel, ylabel = '轴 1', '轴 2'
            elif axis == 1:
                slice_data = array[:, index, :]
                xlabel, ylabel = '轴 0', '轴 2'
            else:  # axis == 2
                slice_data = array[:, :, index]
                xlabel, ylabel = '轴 0', '轴 1'

            figure.clear()
            ax = figure.add_subplot(111)
            im = ax.imshow(slice_data, aspect='auto', cmap='viridis', interpolation='nearest')
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(f'三维切片: 轴 {axis}, 索引 {index}/{array.shape[axis]-1}')
            figure.colorbar(im, ax=ax)
            figure.tight_layout()
            return {'success': True}
        except Exception as e:
            return {'success': False, 'error': f'绘图失败: {str(e)}'}

    @staticmethod
    def plot_volume_projection(array: np.ndarray, figure: Figure,
                              axis: int, method: str = 'mean',
                              quality: str = 'fast') -> Dict:
        """
        绘制三维体数据投影

        Args:
            array: 3D 数组
            axis: 投影轴
            method: 投影方法 'mean', 'max', 'min'
        """
        if array.ndim != 3:
            return {'success': False, 'error': '需要 3D 数组'}

        if axis < 0 or axis >= 3:
            return {'success': False, 'error': f'轴 {axis} 超出范围 [0, 2]'}

        try:
            limits = {
                'fast': 1_000_000,
                'medium': 4_000_000,
                'high': 12_000_000,
            }
            max_elements = limits.get(quality, limits['fast'])
            work_array = array
            sampled = False
            if array.size > max_elements:
                scale = (array.size / max_elements) ** (1 / array.ndim)
                steps = tuple(max(1, int(np.ceil(scale))) for _ in range(array.ndim))
                work_array = array[tuple(slice(None, None, step) for step in steps)]
                sampled = True

            # 投影
            if method == 'mean':
                proj_data = np.mean(work_array, axis=axis)
            elif method == 'max':
                proj_data = np.max(work_array, axis=axis)
            elif method == 'min':
                proj_data = np.min(work_array, axis=axis)
            else:
                return {'success': False, 'error': f'未知投影方法: {method}'}

            # 确定轴标签
            remaining_axes = [i for i in range(3) if i != axis]
            xlabel = f'轴 {remaining_axes[1]}'
            ylabel = f'轴 {remaining_axes[0]}'

            figure.clear()
            ax = figure.add_subplot(111)
            im = ax.imshow(proj_data, aspect='auto', cmap='viridis', interpolation='nearest')
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            suffix = "（降采样预览）" if sampled else ""
            ax.set_title(f'三维投影: 沿轴 {axis} {method}{suffix}')
            figure.colorbar(im, ax=ax)
            figure.tight_layout()
            return {'success': True}
        except Exception as e:
            return {'success': False, 'error': f'绘图失败: {str(e)}'}


# 向后兼容的旧接口
class ArrayVisualizer:
    """保持向后兼容的可视化器"""

    @staticmethod
    def plot_1d_line(array: np.ndarray, figure: Figure) -> Dict:
        return SemanticVisualizer.plot_sequence_line(array, figure)

    @staticmethod
    def plot_1d_histogram(array: np.ndarray, figure: Figure, bins: int = 50) -> Dict:
        return SemanticVisualizer.plot_sequence_histogram(array, figure, bins)

    @staticmethod
    def plot_2d_heatmap(array: np.ndarray, figure: Figure) -> Dict:
        return SemanticVisualizer.plot_image_heatmap(array, figure)

    @staticmethod
    def can_plot(array: np.ndarray, plot_type: str) -> Dict:
        """向后兼容的检查方法"""
        if not np.issubdtype(array.dtype, np.number):
            return {'can_plot': False, 'reason': '非数值类型数组无法绘图'}

        if plot_type in ['折线图', '直方图']:
            if array.ndim != 1:
                return {'can_plot': False, 'reason': f'{plot_type}需要 1D 数组'}
        elif plot_type == '热力图':
            if array.ndim != 2:
                return {'can_plot': False, 'reason': '热力图需要 2D 数组'}
        else:
            return {'can_plot': False, 'reason': '未知的绘图类型'}

        return {'can_plot': True, 'reason': ''}
