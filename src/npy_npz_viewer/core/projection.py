"""
投影模块
负责对多维数组进行投影操作
"""
import numpy as np
from typing import Dict, Optional


class ArrayProjection:
    """数组投影器"""

    @staticmethod
    def project(array: np.ndarray, axis: int, method: str = 'mean') -> Dict:
        """
        沿指定轴投影数组

        Args:
            array: 输入数组
            axis: 投影轴
            method: 投影方法 'mean', 'max', 'min', 'sum'

        Returns:
            dict: {
                'success': bool,
                'array': np.ndarray (如果成功),
                'error': str (如果失败)
            }
        """
        if axis < 0 or axis >= array.ndim:
            return {
                'success': False,
                'error': f'轴 {axis} 超出范围 [0, {array.ndim-1}]'
            }

        if not np.issubdtype(array.dtype, np.number):
            return {
                'success': False,
                'error': '投影需要数值类型数组'
            }

        try:
            if method == 'mean':
                result = np.mean(array, axis=axis)
            elif method == 'max':
                result = np.max(array, axis=axis)
            elif method == 'min':
                result = np.min(array, axis=axis)
            elif method == 'sum':
                result = np.sum(array, axis=axis)
            else:
                return {
                    'success': False,
                    'error': f'未知的投影方法: {method}'
                }

            return {
                'success': True,
                'array': result,
                'method': method,
                'axis': axis
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'投影失败: {str(e)}'
            }

    @staticmethod
    def slice_along_axis(array: np.ndarray, axis: int, index: int) -> Dict:
        """
        沿指定轴切片

        Args:
            array: 输入数组
            axis: 切片轴
            index: 切片索引

        Returns:
            dict: {
                'success': bool,
                'array': np.ndarray (如果成功),
                'error': str (如果失败)
            }
        """
        if axis < 0 or axis >= array.ndim:
            return {
                'success': False,
                'error': f'轴 {axis} 超出范围 [0, {array.ndim-1}]'
            }

        if index < 0 or index >= array.shape[axis]:
            return {
                'success': False,
                'error': f'索引 {index} 超出轴 {axis} 的范围 [0, {array.shape[axis]-1}]'
            }

        try:
            # 构建切片元组
            slices = [slice(None)] * array.ndim
            slices[axis] = index
            result = array[tuple(slices)]

            return {
                'success': True,
                'array': result,
                'axis': axis,
                'index': index
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'切片失败: {str(e)}'
            }

    @staticmethod
    def select_channel(array: np.ndarray, channel_axis: int, channel_index: int) -> Dict:
        """
        选择指定通道

        Args:
            array: 输入数组
            channel_axis: 通道轴
            channel_index: 通道索引

        Returns:
            dict: 同 slice_along_axis
        """
        return ArrayProjection.slice_along_axis(array, channel_axis, channel_index)
