"""
可视化模块
负责使用 Matplotlib 绘制数组图表
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Optional, Dict

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ArrayVisualizer:
    """数组可视化器"""

    @staticmethod
    def plot_1d_line(array: np.ndarray, figure: Figure) -> Dict:
        """绘制 1D 折线图"""
        if array.ndim != 1:
            return {'success': False, 'error': '需要 1D 数组'}

        if not np.issubdtype(array.dtype, np.number):
            return {'success': False, 'error': '需要数值类型数组'}

        try:
            figure.clear()
            ax = figure.add_subplot(111)
            ax.plot(array)
            ax.set_xlabel('索引')
            ax.set_ylabel('值')
            ax.set_title('1D 折线图')
            ax.grid(True, alpha=0.3)
            figure.tight_layout()
            return {'success': True}
        except Exception as e:
            return {'success': False, 'error': f'绘图失败: {str(e)}'}

    @staticmethod
    def plot_1d_histogram(array: np.ndarray, figure: Figure, bins: int = 50) -> Dict:
        """绘制 1D 直方图"""
        if array.ndim != 1:
            return {'success': False, 'error': '需要 1D 数组'}

        if not np.issubdtype(array.dtype, np.number):
            return {'success': False, 'error': '需要数值类型数组'}

        try:
            # 过滤 NaN 和 Inf
            valid_data = array[np.isfinite(array)]
            if len(valid_data) == 0:
                return {'success': False, 'error': '没有有效数据（全为 NaN/Inf）'}

            figure.clear()
            ax = figure.add_subplot(111)
            ax.hist(valid_data, bins=bins, edgecolor='black', alpha=0.7)
            ax.set_xlabel('值')
            ax.set_ylabel('频数')
            ax.set_title('1D 直方图')
            ax.grid(True, alpha=0.3, axis='y')
            figure.tight_layout()
            return {'success': True}
        except Exception as e:
            return {'success': False, 'error': f'绘图失败: {str(e)}'}

    @staticmethod
    def plot_2d_heatmap(array: np.ndarray, figure: Figure) -> Dict:
        """绘制 2D 热力图"""
        if array.ndim != 2:
            return {'success': False, 'error': '需要 2D 数组'}

        if not np.issubdtype(array.dtype, np.number):
            return {'success': False, 'error': '需要数值类型数组'}

        try:
            figure.clear()
            ax = figure.add_subplot(111)
            im = ax.imshow(array, aspect='auto', cmap='viridis', interpolation='nearest')
            ax.set_xlabel('列索引')
            ax.set_ylabel('行索引')
            ax.set_title('2D 热力图')
            figure.colorbar(im, ax=ax)
            figure.tight_layout()
            return {'success': True}
        except Exception as e:
            return {'success': False, 'error': f'绘图失败: {str(e)}'}

    @staticmethod
    def can_plot(array: np.ndarray, plot_type: str) -> Dict:
        """
        检查数组是否可以绘制指定类型的图

        Args:
            array: 数组
            plot_type: '折线图', '直方图', '热力图'

        Returns:
            dict: {'can_plot': bool, 'reason': str}
        """
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
