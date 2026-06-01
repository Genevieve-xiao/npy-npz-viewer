"""
大数据优化工具
提供绘图降采样等优化功能
"""
import numpy as np
from typing import Tuple, Optional


class LargeDataOptimizer:
    """大数据优化器"""

    # 绘图降采样阈值
    PLOT_MAX_POINTS_1D = 10000      # 1D 折线图最多显示点数
    PLOT_MAX_POINTS_2D = 100000     # 2D 散点图最多显示点数
    PLOT_MAX_ROWS_HEATMAP = 2000    # 热力图最多显示行数
    PLOT_MAX_COLS_HEATMAP = 2000    # 热力图最多显示列数

    @staticmethod
    def should_downsample_1d(array: np.ndarray) -> bool:
        """判断 1D 数组是否需要降采样"""
        return array.size > LargeDataOptimizer.PLOT_MAX_POINTS_1D

    @staticmethod
    def downsample_1d(array: np.ndarray, max_points: int = None) -> Tuple[np.ndarray, str]:
        """
        对 1D 数组降采样

        Returns:
            (降采样后的数组, 说明信息)
        """
        if max_points is None:
            max_points = LargeDataOptimizer.PLOT_MAX_POINTS_1D

        if array.size <= max_points:
            return array, ""

        # 均匀采样
        indices = np.linspace(0, array.size - 1, max_points, dtype=int)
        downsampled = array[indices]

        info = f"⚠️ 数据量过大，已降采样至 {max_points:,} 个点（原始: {array.size:,} 个点）"
        return downsampled, info

    @staticmethod
    def downsample_2d_for_plot(array: np.ndarray, x_col: Optional[np.ndarray] = None,
                               max_points: int = None) -> Tuple[np.ndarray, Optional[np.ndarray], str]:
        """
        对 2D 表格数据降采样（用于折线图）

        Args:
            array: (N, C) 数组
            x_col: X 轴数据（如果有）
            max_points: 最大点数

        Returns:
            (降采样后的数组, 降采样后的 x_col, 说明信息)
        """
        if max_points is None:
            max_points = LargeDataOptimizer.PLOT_MAX_POINTS_1D

        rows = array.shape[0]
        if rows <= max_points:
            return array, x_col, ""

        # 均匀采样行
        indices = np.linspace(0, rows - 1, max_points, dtype=int)
        downsampled = array[indices, :]

        if x_col is not None:
            x_col_downsampled = x_col[indices]
        else:
            x_col_downsampled = None

        info = f"⚠️ 数据量过大，已降采样至 {max_points:,} 行（原始: {rows:,} 行）"
        return downsampled, x_col_downsampled, info

    @staticmethod
    def downsample_scatter(x_data: np.ndarray, y_data: np.ndarray,
                          max_points: int = None) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        对散点图数据降采样

        Returns:
            (降采样后的 x, 降采样后的 y, 说明信息)
        """
        if max_points is None:
            max_points = LargeDataOptimizer.PLOT_MAX_POINTS_2D

        if len(x_data) <= max_points:
            return x_data, y_data, ""

        # 随机采样（保持数据分布）
        indices = np.random.choice(len(x_data), max_points, replace=False)
        indices = np.sort(indices)  # 排序保持顺序

        x_downsampled = x_data[indices]
        y_downsampled = y_data[indices]

        info = f"⚠️ 数据量过大，已随机采样 {max_points:,} 个点（原始: {len(x_data):,} 个点）"
        return x_downsampled, y_downsampled, info

    @staticmethod
    def downsample_heatmap(array: np.ndarray) -> Tuple[np.ndarray, str]:
        """
        对热力图数据降采样

        Returns:
            (降采样后的数组, 说明信息)
        """
        rows, cols = array.shape
        max_rows = LargeDataOptimizer.PLOT_MAX_ROWS_HEATMAP
        max_cols = LargeDataOptimizer.PLOT_MAX_COLS_HEATMAP

        if rows <= max_rows and cols <= max_cols:
            return array, ""

        # 计算采样步长
        row_step = max(1, rows // max_rows)
        col_step = max(1, cols // max_cols)

        # 均匀采样
        downsampled = array[::row_step, ::col_step]

        info = f"⚠️ 数据量过大，已降采样\n原始: {rows} × {cols}\n显示: {downsampled.shape[0]} × {downsampled.shape[1]}"
        return downsampled, info

    @staticmethod
    def get_quick_slice_suggestion(array: np.ndarray) -> Optional[str]:
        """
        获取快速切片建议

        Returns:
            切片建议字符串，如 "0:10000" 或 None
        """
        if array.ndim == 0:
            return None

        # 对于第一维度
        first_dim_size = array.shape[0]

        if first_dim_size > 100000:
            return "0:10000"  # 前 1 万行
        elif first_dim_size > 50000:
            return "0:5000"   # 前 5000 行
        elif first_dim_size > 10000:
            return "0:1000"   # 前 1000 行
        else:
            return None

    @staticmethod
    def format_data_size_warning(array: np.ndarray) -> Optional[str]:
        """
        生成数据大小警告信息

        Returns:
            警告信息或 None
        """
        size = array.size
        shape = array.shape

        if size > 10_000_000:  # 1000 万
            return (
                f"⚠️ 数据量很大: {shape} ({size:,} 个元素)\n"
                f"已使用内存映射、限量预览和采样统计；绘图前建议先用「维度筛选」或「通用切片」减少数据量。"
            )
        elif size > 1_000_000:  # 100 万
            return (
                f"ℹ️ 数据量较大: {shape} ({size:,} 个元素)\n"
                f"如遇性能问题，可先使用「维度筛选」或「通用切片」功能。"
            )
        else:
            return None
