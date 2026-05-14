"""
切片逻辑模块
负责处理多维数组的切片操作
"""
import numpy as np
from typing import List, Tuple, Optional, Dict


class ArraySlicer:
    """数组切片器"""

    @staticmethod
    def parse_slice_string(slice_str: str) -> Optional[slice]:
        """
        解析切片字符串，格式: start:stop:step
        支持省略形式: :10, 5:, ::2, :
        """
        if not slice_str or slice_str.strip() == ':':
            return slice(None)

        parts = slice_str.split(':')
        if len(parts) > 3:
            return None

        try:
            start = int(parts[0]) if parts[0].strip() else None
            stop = int(parts[1]) if len(parts) > 1 and parts[1].strip() else None
            step = int(parts[2]) if len(parts) > 2 and parts[2].strip() else None
            return slice(start, stop, step)
        except ValueError:
            return None

    @staticmethod
    def apply_slice(array: np.ndarray, slice_specs: List[str]) -> Dict:
        """
        对数组应用切片

        Args:
            array: 原始数组
            slice_specs: 每个维度的切片字符串列表

        Returns:
            dict: {
                'success': bool,
                'array': np.ndarray (如果成功),
                'error': str (如果失败)
            }
        """
        if len(slice_specs) != array.ndim:
            return {
                'success': False,
                'error': f'切片维度数 ({len(slice_specs)}) 与数组维度 ({array.ndim}) 不匹配'
            }

        # 解析所有切片
        slices = []
        for i, spec in enumerate(slice_specs):
            s = ArraySlicer.parse_slice_string(spec)
            if s is None:
                return {
                    'success': False,
                    'error': f'第 {i} 维切片格式错误: "{spec}"'
                }
            slices.append(s)

        # 应用切片
        try:
            sliced = array[tuple(slices)]
            return {'success': True, 'array': sliced}
        except Exception as e:
            return {'success': False, 'error': f'切片失败: {str(e)}'}

    @staticmethod
    def get_slice_for_dimension(array: np.ndarray, dim: int, index: int) -> np.ndarray:
        """
        获取指定维度的单个索引切片
        用于降维操作
        """
        slices = [slice(None)] * array.ndim
        slices[dim] = index
        return array[tuple(slices)]
