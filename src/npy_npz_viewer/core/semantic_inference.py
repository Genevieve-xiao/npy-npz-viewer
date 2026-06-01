"""
语义推断模块
根据数组形状和类型推断数据语义
"""
import numpy as np
from typing import Dict, Tuple
from npy_npz_viewer.core.data_semantics import DataSemantics


class SemanticInference:
    """语义推断器"""

    # 阈值配置
    SMALL_CHANNEL_THRESHOLD = 16  # 小于此值认为是通道/属性维度
    IMAGE_MIN_SIZE = 50           # 图像最小尺寸

    @staticmethod
    def infer(array: np.ndarray) -> Dict:
        """
        推断数组的数据语义

        Returns:
            dict: {
                'semantic': DataSemantics,
                'confidence': str,  # 'high', 'medium', 'low'
                'reason': str,
                'suggestions': list  # 其他可能的解释
            }
        """
        shape = array.shape
        ndim = array.ndim
        dtype = array.dtype

        # 非数值类型
        if not np.issubdtype(dtype, np.number):
            return {
                'semantic': DataSemantics.UNKNOWN,
                'confidence': 'high',
                'reason': '非数值类型数组',
                'suggestions': []
            }

        # 1D 数组
        if ndim == 1:
            return {
                'semantic': DataSemantics.SEQUENCE_1D,
                'confidence': 'high',
                'reason': f'一维数组，长度 {shape[0]}',
                'suggestions': []
            }

        # 2D 数组
        elif ndim == 2:
            return SemanticInference._infer_2d(shape)

        # 3D 数组
        elif ndim == 3:
            return SemanticInference._infer_3d(shape)

        # 4D 数组
        elif ndim == 4:
            return SemanticInference._infer_4d(shape)

        # 更高维度
        else:
            return {
                'semantic': DataSemantics.UNKNOWN,
                'confidence': 'high',
                'reason': f'{ndim}维数组，维度过高',
                'suggestions': []
            }

    @staticmethod
    def _infer_2d(shape: Tuple[int, int]) -> Dict:
        """推断 2D 数组语义"""
        rows, cols = shape

        # 列数很小 → 表格数据（样本 × 特征）
        if cols <= SemanticInference.SMALL_CHANNEL_THRESHOLD:
            return {
                'semantic': DataSemantics.TABULAR_2D,
                'confidence': 'high',
                'reason': f'{rows} 个样本 × {cols} 个属性/特征',
                'suggestions': [DataSemantics.IMAGE_2D]
            }

        # 两维都较大 → 图像/矩阵
        elif rows >= SemanticInference.IMAGE_MIN_SIZE and cols >= SemanticInference.IMAGE_MIN_SIZE:
            return {
                'semantic': DataSemantics.IMAGE_2D,
                'confidence': 'high',
                'reason': f'{rows} × {cols} 图像/矩阵',
                'suggestions': [DataSemantics.TABULAR_2D]
            }

        # 行数很小 → 可能是转置的表格
        elif rows <= SemanticInference.SMALL_CHANNEL_THRESHOLD:
            return {
                'semantic': DataSemantics.TABULAR_2D,
                'confidence': 'medium',
                'reason': f'{rows} 个属性 × {cols} 个样本（可能需要转置）',
                'suggestions': [DataSemantics.IMAGE_2D]
            }

        # 默认表格
        else:
            return {
                'semantic': DataSemantics.TABULAR_2D,
                'confidence': 'medium',
                'reason': f'{rows} × {cols}，默认为表格数据',
                'suggestions': [DataSemantics.IMAGE_2D]
            }

    @staticmethod
    def _infer_3d(shape: Tuple[int, int, int]) -> Dict:
        """推断 3D 数组语义"""
        d0, d1, d2 = shape

        # 最后一维很小 → 多通道图像
        if d2 <= SemanticInference.SMALL_CHANNEL_THRESHOLD:
            return {
                'semantic': DataSemantics.MULTICHANNEL_3D,
                'confidence': 'high',
                'reason': f'{d0} × {d1} 图像，{d2} 个通道',
                'suggestions': [DataSemantics.VOLUME_3D]
            }

        # 第一维很小 → 可能是多通道
        elif d0 <= SemanticInference.SMALL_CHANNEL_THRESHOLD:
            return {
                'semantic': DataSemantics.MULTICHANNEL_3D,
                'confidence': 'medium',
                'reason': f'{d0} 个通道，{d1} × {d2} 图像（轴 0 为通道）',
                'suggestions': [DataSemantics.VOLUME_3D]
            }

        # 默认体数据
        else:
            return {
                'semantic': DataSemantics.VOLUME_3D,
                'confidence': 'high',
                'reason': f'{d0} × {d1} × {d2} 体数据',
                'suggestions': [DataSemantics.MULTICHANNEL_3D]
            }

    @staticmethod
    def _infer_4d(shape: Tuple[int, int, int, int]) -> Dict:
        """推断 4D 数组语义"""
        d0, d1, d2, d3 = shape

        # 最后一维很小 → 体数据 + 通道
        if d3 <= SemanticInference.SMALL_CHANNEL_THRESHOLD:
            return {
                'semantic': DataSemantics.VOLUME_4D,
                'confidence': 'high',
                'reason': f'{d0} × {d1} × {d2} 体数据，{d3} 个通道',
                'suggestions': []
            }

        # 第一维很小 → 可能是通道在前
        elif d0 <= SemanticInference.SMALL_CHANNEL_THRESHOLD:
            return {
                'semantic': DataSemantics.VOLUME_4D,
                'confidence': 'medium',
                'reason': f'{d0} 个通道，{d1} × {d2} × {d3} 体数据（轴 0 为通道）',
                'suggestions': []
            }

        # 默认
        else:
            return {
                'semantic': DataSemantics.VOLUME_4D,
                'confidence': 'medium',
                'reason': f'{d0} × {d1} × {d2} × {d3} 四维数据',
                'suggestions': []
            }

    @staticmethod
    def check_monotonic(array: np.ndarray) -> Dict:
        """
        检查 1D 数组是否单调

        Returns:
            dict: {
                'is_monotonic': bool,
                'direction': 'increasing' | 'decreasing' | None,
                'suggestion': str
            }
        """
        if array.ndim != 1 or len(array) < 2:
            return {'is_monotonic': False, 'direction': None, 'suggestion': ''}

        try:
            diff = np.diff(array)
            if np.all(diff > 0):
                return {
                    'is_monotonic': True,
                    'direction': 'increasing',
                    'suggestion': '此列单调递增，可能是深度/时间/坐标轴'
                }
            elif np.all(diff < 0):
                return {
                    'is_monotonic': True,
                    'direction': 'decreasing',
                    'suggestion': '此列单调递减，可能是深度/时间/坐标轴'
                }
            elif np.all(diff >= 0):
                return {
                    'is_monotonic': True,
                    'direction': 'increasing',
                    'suggestion': '此列单调非递减，可能是深度/时间/坐标轴'
                }
            elif np.all(diff <= 0):
                return {
                    'is_monotonic': True,
                    'direction': 'decreasing',
                    'suggestion': '此列单调非递增，可能是深度/时间/坐标轴'
                }
        except Exception:
            pass

        return {'is_monotonic': False, 'direction': None, 'suggestion': ''}
