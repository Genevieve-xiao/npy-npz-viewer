"""
数据语义类型定义
定义不同的数据解释方式
"""
from enum import Enum
from typing import Tuple


class DataSemantics(Enum):
    """数据语义类型"""
    SEQUENCE_1D = "一维序列"
    TABULAR_2D = "二维表格"
    IMAGE_2D = "二维图像/矩阵"
    VOLUME_3D = "三维体数据"
    MULTICHANNEL_3D = "三维多通道"
    VOLUME_4D = "四维体数据"
    UNKNOWN = "未知类型"

    @classmethod
    def get_all_types(cls):
        """获取所有语义类型"""
        return [s.value for s in cls if s != cls.UNKNOWN]

    @classmethod
    def from_string(cls, s: str):
        """从字符串获取枚举"""
        for semantic in cls:
            if semantic.value == s:
                return semantic
        return cls.UNKNOWN


# 每种语义支持的可视化类型
SEMANTIC_PLOT_OPTIONS = {
    DataSemantics.SEQUENCE_1D: [
        "折线图",
        "直方图"
    ],
    DataSemantics.TABULAR_2D: [
        "多折线图",
        "单列直方图",
        "散点图",
        "相关性热力图"
    ],
    DataSemantics.IMAGE_2D: [
        "热力图",
        "图像显示"
    ],
    DataSemantics.VOLUME_3D: [
        "切片热力图",
        "投影图",
        "3D散点图",
        "3D表面图",
        "3D线框图",
        "3D等高线图",
        "3D体素图",
        "3D切片堆叠图"
    ],
    DataSemantics.MULTICHANNEL_3D: [
        "通道热力图",
        "通道图像"
    ],
    DataSemantics.VOLUME_4D: [
        "切片热力图",
        "投影图"
    ],
    DataSemantics.UNKNOWN: []
}


def get_plot_options(semantic: DataSemantics) -> list:
    """获取指定语义的可视化选项"""
    return SEMANTIC_PLOT_OPTIONS.get(semantic, [])
