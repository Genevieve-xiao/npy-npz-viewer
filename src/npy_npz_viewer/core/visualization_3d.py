"""
3D 可视化模块
使用 Matplotlib 的 3D 绘图功能
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, Optional
from npy_npz_viewer.core.array_compute import ArrayComputeService


class Visualizer3D:
    """3D 可视化器"""

    @staticmethod
    def _regular_indices(length: int, max_count: int) -> np.ndarray:
        count = min(length, max_count)
        return np.linspace(0, length - 1, count, dtype=int)

    @staticmethod
    def _downsample_2d(data: np.ndarray, max_rows: int = 240,
                       max_cols: int = 240) -> np.ndarray:
        row_step = max(1, int(np.ceil(data.shape[0] / max_rows)))
        col_step = max(1, int(np.ceil(data.shape[1] / max_cols)))
        return ArrayComputeService.to_numpy(data[::row_step, ::col_step])

    @staticmethod
    def _sample_volume_points(array: np.ndarray, sample_size: int):
        d, h, w = array.shape
        per_axis = max(2, int(np.ceil(sample_size ** (1 / 3))))
        z_idx = Visualizer3D._regular_indices(d, per_axis)
        y_idx = Visualizer3D._regular_indices(h, per_axis)
        x_idx = Visualizer3D._regular_indices(w, per_axis)
        zz, yy, xx = np.meshgrid(z_idx, y_idx, x_idx, indexing='ij')
        x = xx.ravel()
        y = yy.ravel()
        z = zz.ravel()
        values = array[z, y, x]
        if values.size > sample_size:
            indices = np.linspace(0, values.size - 1, sample_size, dtype=int)
            x = x[indices]
            y = y[indices]
            z = z[indices]
            values = values[indices]
        return x, y, z, ArrayComputeService.to_numpy(values)

    @staticmethod
    def plot_3d_scatter(array: np.ndarray, figure: Figure,
                       sample_size: int = 5000) -> Dict:
        """
        绘制 3D 散点图

        Args:
            array: (N, 3) 或 (D, H, W) 数组
            sample_size: 采样点数（避免过多点导致卡顿）
        """
        if array.ndim == 2 and array.shape[1] == 3:
            # (N, 3) 格式
            if array.shape[0] > sample_size:
                indices = np.linspace(0, array.shape[0] - 1, sample_size, dtype=int)
                points = array[indices]
            else:
                points = array
            x, y, z = points[:, 0], points[:, 1], points[:, 2]
        elif array.ndim == 3:
            x, y, z, values = Visualizer3D._sample_volume_points(array, sample_size)
        else:
            return {'success': False, 'error': '需要 (N, 3) 或 (D, H, W) 数组'}

        try:
            figure.clear()
            ax = figure.add_subplot(111, projection='3d')

            if array.ndim == 3:
                # 使用值作为颜色
                scatter = ax.scatter(x, y, z, c=values, cmap='viridis',
                                   alpha=0.6, s=20)
                figure.colorbar(scatter, ax=ax, label='值')
            else:
                ax.scatter(x, y, z, alpha=0.6, s=20)

            ax.set_xlabel('X 轴')
            ax.set_ylabel('Y 轴')
            ax.set_zlabel('Z 轴')
            ax.set_title('3D 散点图')

            figure.tight_layout()
            return {'success': True}
        except Exception as e:
            return {'success': False, 'error': f'绘图失败: {str(e)}'}

    @staticmethod
    def plot_3d_surface(array: np.ndarray, figure: Figure,
                       slice_axis: int = 0, slice_index: int = None) -> Dict:
        """
        绘制 3D 表面图

        Args:
            array: (D, H, W) 数组
            slice_axis: 切片轴
            slice_index: 切片索引（None 表示使用中间切片）
        """
        if array.ndim != 3:
            return {'success': False, 'error': '需要 3D 数组'}

        try:
            # 提取切片
            if slice_index is None:
                slice_index = array.shape[slice_axis] // 2

            if slice_axis == 0:
                data = array[slice_index, :, :]
                xlabel, ylabel = 'Y 轴', 'Z 轴'
            elif slice_axis == 1:
                data = array[:, slice_index, :]
                xlabel, ylabel = 'X 轴', 'Z 轴'
            else:
                data = array[:, :, slice_index]
                xlabel, ylabel = 'X 轴', 'Y 轴'
            data = Visualizer3D._downsample_2d(data)

            # 创建网格
            h, w = data.shape
            x = np.arange(w)
            y = np.arange(h)
            X, Y = np.meshgrid(x, y)

            figure.clear()
            ax = figure.add_subplot(111, projection='3d')

            surf = ax.plot_surface(X, Y, data, cmap='viridis',
                                  alpha=0.8, antialiased=True)

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_zlabel('值')
            ax.set_title(f'3D 表面图 (轴 {slice_axis}, 索引 {slice_index})')

            figure.colorbar(surf, ax=ax, shrink=0.5)
            figure.tight_layout()

            return {'success': True}
        except Exception as e:
            return {'success': False, 'error': f'绘图失败: {str(e)}'}

    @staticmethod
    def plot_3d_wireframe(array: np.ndarray, figure: Figure,
                         slice_axis: int = 0, slice_index: int = None,
                         stride: int = 2) -> Dict:
        """
        绘制 3D 线框图

        Args:
            array: (D, H, W) 数组
            stride: 采样步长（减少线条数量）
        """
        if array.ndim != 3:
            return {'success': False, 'error': '需要 3D 数组'}

        try:
            # 提取切片
            if slice_index is None:
                slice_index = array.shape[slice_axis] // 2

            if slice_axis == 0:
                data = array[slice_index, :, :]
                xlabel, ylabel = 'Y 轴', 'Z 轴'
            elif slice_axis == 1:
                data = array[:, slice_index, :]
                xlabel, ylabel = 'X 轴', 'Z 轴'
            else:
                data = array[:, :, slice_index]
                xlabel, ylabel = 'X 轴', 'Y 轴'
            data = Visualizer3D._downsample_2d(data)

            # 创建网格
            h, w = data.shape
            x = np.arange(w)
            y = np.arange(h)
            X, Y = np.meshgrid(x, y)

            figure.clear()
            ax = figure.add_subplot(111, projection='3d')

            ax.plot_wireframe(X, Y, data, rstride=stride, cstride=stride,
                            color='blue', alpha=0.7)

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_zlabel('值')
            ax.set_title(f'3D 线框图 (轴 {slice_axis}, 索引 {slice_index})')

            figure.tight_layout()

            return {'success': True}
        except Exception as e:
            return {'success': False, 'error': f'绘图失败: {str(e)}'}

    @staticmethod
    def plot_3d_contour(array: np.ndarray, figure: Figure,
                       slice_axis: int = 0, slice_index: int = None,
                       levels: int = 15) -> Dict:
        """
        绘制 3D 等高线图

        Args:
            array: (D, H, W) 数组
            levels: 等高线层数
        """
        if array.ndim != 3:
            return {'success': False, 'error': '需要 3D 数组'}

        try:
            # 提取切片
            if slice_index is None:
                slice_index = array.shape[slice_axis] // 2

            if slice_axis == 0:
                data = array[slice_index, :, :]
                xlabel, ylabel = 'Y 轴', 'Z 轴'
            elif slice_axis == 1:
                data = array[:, slice_index, :]
                xlabel, ylabel = 'X 轴', 'Z 轴'
            else:
                data = array[:, :, slice_index]
                xlabel, ylabel = 'X 轴', 'Y 轴'
            data = Visualizer3D._downsample_2d(data)

            # 创建网格
            h, w = data.shape
            x = np.arange(w)
            y = np.arange(h)
            X, Y = np.meshgrid(x, y)

            figure.clear()
            ax = figure.add_subplot(111, projection='3d')

            contour = ax.contour3D(X, Y, data, levels=levels, cmap='viridis')

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_zlabel('值')
            ax.set_title(f'3D 等高线图 (轴 {slice_axis}, 索引 {slice_index})')

            figure.colorbar(contour, ax=ax, shrink=0.5)
            figure.tight_layout()

            return {'success': True}
        except Exception as e:
            return {'success': False, 'error': f'绘图失败: {str(e)}'}

    @staticmethod
    def plot_3d_voxel(array: np.ndarray, figure: Figure,
                     threshold: float = None, max_voxels: int = 10000) -> Dict:
        """
        绘制 3D 体素图

        Args:
            array: (D, H, W) 数组
            threshold: 阈值（只显示大于阈值的体素）
            max_voxels: 最大体素数（避免过多导致卡顿）
        """
        if array.ndim != 3:
            return {'success': False, 'error': '需要 3D 数组'}

        try:
            d_step = max(1, int(np.ceil(array.shape[0] / 60)))
            h_step = max(1, int(np.ceil(array.shape[1] / 60)))
            w_step = max(1, int(np.ceil(array.shape[2] / 60)))
            sampled = ArrayComputeService.to_numpy(array[::d_step, ::h_step, ::w_step])

            # 自动计算阈值（如果未指定）
            if threshold is None:
                threshold = np.percentile(sampled, 75)  # 只显示前 25% 的值

            # 创建布尔掩码
            mask = sampled > threshold

            # 限制体素数量
            if np.sum(mask) > max_voxels:
                # 随机采样
                indices = np.where(mask)
                sample_indices = np.random.choice(
                    len(indices[0]), max_voxels, replace=False
                )
                new_mask = np.zeros_like(mask)
                new_mask[
                    indices[0][sample_indices],
                    indices[1][sample_indices],
                    indices[2][sample_indices]
                ] = True
                mask = new_mask

            figure.clear()
            ax = figure.add_subplot(111, projection='3d')

            # 绘制体素
            max_value = np.nanmax(sampled)
            if max_value == 0 or not np.isfinite(max_value):
                max_value = 1
            colors = plt.cm.viridis(sampled / max_value)
            ax.voxels(mask, facecolors=colors, edgecolor='k', alpha=0.7)

            ax.set_xlabel('X 轴')
            ax.set_ylabel('Y 轴')
            ax.set_zlabel('Z 轴')
            ax.set_title(f'3D 体素图 (阈值: {threshold:.2f})')

            figure.tight_layout()

            return {'success': True}
        except Exception as e:
            return {'success': False, 'error': f'绘图失败: {str(e)}'}

    @staticmethod
    def plot_3d_slice_stack(array: np.ndarray, figure: Figure,
                           axis: int = 0, num_slices: int = 5) -> Dict:
        """
        绘制 3D 切片堆叠图

        Args:
            array: (D, H, W) 数组
            axis: 切片轴
            num_slices: 显示的切片数量
        """
        if array.ndim != 3:
            return {'success': False, 'error': '需要 3D 数组'}

        try:
            figure.clear()
            ax = figure.add_subplot(111, projection='3d')

            # 选择切片索引
            total_slices = array.shape[axis]
            indices = np.linspace(0, total_slices - 1, num_slices, dtype=int)

            for i, idx in enumerate(indices):
                if axis == 0:
                    data = array[idx, :, :]
                    data = Visualizer3D._downsample_2d(data, 160, 160)
                    h, w = data.shape
                    x = np.arange(w)
                    y = np.arange(h)
                    X, Y = np.meshgrid(x, y)
                    Z = np.full_like(X, idx, dtype=float)
                elif axis == 1:
                    data = array[:, idx, :]
                    data = Visualizer3D._downsample_2d(data, 160, 160)
                    d, w = data.shape
                    x = np.arange(w)
                    z = np.arange(d)
                    X, Z = np.meshgrid(x, z)
                    Y = np.full_like(X, idx, dtype=float)
                else:
                    data = array[:, :, idx]
                    data = Visualizer3D._downsample_2d(data, 160, 160)
                    d, h = data.shape
                    y = np.arange(h)
                    z = np.arange(d)
                    Y, Z = np.meshgrid(y, z)
                    X = np.full_like(Y, idx, dtype=float)

                # 绘制切片
                max_value = np.nanmax(data)
                if max_value == 0 or not np.isfinite(max_value):
                    max_value = 1
                ax.plot_surface(X, Y, Z, facecolors=plt.cm.viridis(data / max_value),
                              alpha=0.7, antialiased=False)

            ax.set_xlabel('X 轴')
            ax.set_ylabel('Y 轴')
            ax.set_zlabel('Z 轴')
            ax.set_title(f'3D 切片堆叠图 (沿轴 {axis})')

            figure.tight_layout()

            return {'success': True}
        except Exception as e:
            return {'success': False, 'error': f'绘图失败: {str(e)}'}
