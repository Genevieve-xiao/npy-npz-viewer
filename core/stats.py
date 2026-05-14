"""
统计计算模块
负责计算数组的各种统计信息
"""
import numpy as np
from typing import Dict, Optional


class ArrayStats:
    """数组统计计算器"""

    # 大数组阈值：超过此元素数量时使用采样
    LARGE_ARRAY_THRESHOLD = 10_000_000
    LARGE_SAMPLE_SIZE = 200_000
    SAMPLE_CHUNKS = 10

    @staticmethod
    def _sample_large_array(array: np.ndarray) -> np.ndarray:
        """用少量连续块采样大数组，避免 memmap 上的大量随机读。"""
        flat = array.ravel()
        sample_size = min(ArrayStats.LARGE_SAMPLE_SIZE, flat.size)
        if sample_size <= 0:
            return flat[:0]

        chunk_count = min(ArrayStats.SAMPLE_CHUNKS, sample_size)
        chunk_size = max(1, sample_size // chunk_count)

        if flat.size <= sample_size:
            return flat

        max_start = max(0, flat.size - chunk_size)
        starts = np.linspace(0, max_start, chunk_count, dtype=np.int64)
        chunks = [flat[start:start + chunk_size] for start in starts]
        sample = np.concatenate(chunks)
        return sample[:sample_size]

    @staticmethod
    def compute_stats(array: np.ndarray) -> Dict:
        """
        计算数组统计信息

        Returns:
            dict: {
                'shape': tuple,
                'dtype': str,
                'ndim': int,
                'size': int,
                'memory_mb': float,
                'has_nan': bool,
                'has_inf': bool,
                'min': float/None,
                'max': float/None,
                'mean': float/None,
                'std': float/None,
                'is_numeric': bool,
                'sampled': bool
            }
        """
        stats = {
            'shape': array.shape,
            'dtype': str(array.dtype),
            'ndim': array.ndim,
            'size': array.size,
            'memory_mb': array.nbytes / (1024 * 1024),
            'is_numeric': np.issubdtype(array.dtype, np.number),
            'sampled': False
        }

        # 非数值类型数组无法计算统计量
        if not stats['is_numeric']:
            stats.update({
                'has_nan': False,
                'has_inf': False,
                'min': None,
                'max': None,
                'mean': None,
                'std': None
            })
            return stats

        # 对于大数组使用采样
        if array.size > ArrayStats.LARGE_ARRAY_THRESHOLD:
            sample = ArrayStats._sample_large_array(array)
            stats['sampled'] = True
        else:
            sample = array

        # 计算统计量
        try:
            stats['has_nan'] = bool(np.isnan(sample).any())
            stats['has_inf'] = bool(np.isinf(sample).any())

            # 过滤 NaN 和 Inf 后计算统计量
            valid_data = sample[np.isfinite(sample)]
            if len(valid_data) > 0:
                stats['min'] = float(valid_data.min())
                stats['max'] = float(valid_data.max())
                stats['mean'] = float(valid_data.mean())
                stats['std'] = float(valid_data.std())
            else:
                stats['min'] = None
                stats['max'] = None
                stats['mean'] = None
                stats['std'] = None
        except Exception:
            # 某些特殊 dtype 可能无法计算
            stats.update({
                'has_nan': False,
                'has_inf': False,
                'min': None,
                'max': None,
                'mean': None,
                'std': None
            })

        return stats

    @staticmethod
    def format_stats(stats: Dict) -> str:
        """格式化统计信息为可读文本"""
        lines = [
            f"形状: {stats['shape']}",
            f"数据类型: {stats['dtype']}",
            f"维度: {stats['ndim']}",
            f"元素数量: {stats['size']:,}",
            f"内存占用: {stats['memory_mb']:.2f} MB",
        ]

        if stats['sampled']:
            lines.append("⚠️ 数组过大，统计量基于采样计算")

        if stats['is_numeric']:
            lines.append(f"包含 NaN: {'是' if stats['has_nan'] else '否'}")
            lines.append(f"包含 Inf: {'是' if stats['has_inf'] else '否'}")

            if stats['min'] is not None:
                lines.append(f"最小值: {stats['min']:.6g}")
                lines.append(f"最大值: {stats['max']:.6g}")
                lines.append(f"均值: {stats['mean']:.6g}")
                lines.append(f"标准差: {stats['std']:.6g}")
            else:
                lines.append("统计量: 无有效数据（全为 NaN/Inf）")
        else:
            lines.append("非数值类型，无统计量")

        return "\n".join(lines)
