"""
辅助函数模块
提供格式化、预览限制等工具函数
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional


class PreviewHelper:
    """数据预览辅助类"""

    # 预览限制
    MAX_PREVIEW_ROWS = 1000
    MAX_PREVIEW_COLS = 20

    @staticmethod
    def build_preview(array: np.ndarray, mode: str = "auto",
                      start: int = 0, end: int = 1000) -> Tuple[Optional[pd.DataFrame], str]:
        """Build a bounded preview DataFrame and human-readable info."""
        try:
            if array.ndim == 0:
                return pd.DataFrame({"值": [array.item()]}), "标量数据"

            if mode == "auto":
                mode = "slice" if array.ndim >= 3 else "table"

            if mode == "summary":
                rows = []
                for axis, size in enumerate(array.shape):
                    rows.append({
                        "轴": axis,
                        "长度": size,
                        "中间索引": size // 2,
                        "建议切片": f"{size // 2}",
                    })
                return pd.DataFrame(rows), f"轴摘要: {array.shape}"

            if mode == "slice" and array.ndim >= 3:
                indexer = []
                fixed_axes = []
                open_axes = []
                for axis, size in enumerate(array.shape):
                    if len(open_axes) < 2:
                        indexer.append(slice(None))
                        open_axes.append(axis)
                    else:
                        mid = size // 2
                        indexer.append(mid)
                        fixed_axes.append((axis, mid))
                data = array[tuple(indexer)]
                if data.ndim > 2:
                    data = np.squeeze(data)
                if data.ndim != 2:
                    return PreviewHelper.build_preview(array, "flat", start, end)
                rows = min(data.shape[0], PreviewHelper.MAX_PREVIEW_ROWS)
                cols = min(data.shape[1], PreviewHelper.MAX_PREVIEW_COLS)
                info = f"切片预览: 轴 {open_axes[0]} × 轴 {open_axes[1]}"
                if fixed_axes:
                    info += "，固定 " + ", ".join(f"轴{axis}={idx}" for axis, idx in fixed_axes)
                info += f"，显示 {rows} × {cols}（原始切片 {data.shape}）"
                return pd.DataFrame(data[:rows, :cols]), info

            if mode == "flat" or array.ndim > 2:
                flat = array.ravel()
                start = max(0, min(start, flat.size))
                end = max(start, min(end, flat.size))
                preview = flat[start:end]
                return (
                    pd.DataFrame({"展平索引": range(start, end), "值": preview}),
                    f"展平预览第 {start} 到 {end - 1} 个元素（共 {flat.size} 个元素）",
                )

            if array.ndim == 1:
                start = max(0, min(start, array.size))
                end = max(start, min(end, array.size))
                return (
                    pd.DataFrame({"索引": range(start, end), "值": array[start:end]}),
                    f"显示第 {start} 到 {end - 1} 个元素（共 {array.size} 个）",
                )

            rows = array.shape[0]
            start = max(0, min(start, rows))
            end = max(start, min(end, rows))
            cols = min(array.shape[1], PreviewHelper.MAX_PREVIEW_COLS)
            return (
                pd.DataFrame(array[start:end, :cols]),
                f"显示第 {start} 到 {end - 1} 行 × 前 {cols} 列（原始: {array.shape[0]} 行 × {array.shape[1]} 列）",
            )
        except Exception:
            return None, "预览生成失败"

    @staticmethod
    def get_preview_slice(array: np.ndarray) -> Tuple[np.ndarray, str]:
        """
        获取数组的预览切片

        Returns:
            (预览数组, 说明文本)
        """
        if array.ndim == 1:
            if array.size <= PreviewHelper.MAX_PREVIEW_ROWS:
                return array, f"显示全部 {array.size} 个元素"
            else:
                preview = array[:PreviewHelper.MAX_PREVIEW_ROWS]
                return preview, f"显示前 {PreviewHelper.MAX_PREVIEW_ROWS} 个元素（共 {array.size} 个）"

        elif array.ndim == 2:
            rows, cols = array.shape
            limited_rows = min(rows, PreviewHelper.MAX_PREVIEW_ROWS)
            limited_cols = min(cols, PreviewHelper.MAX_PREVIEW_COLS)

            preview = array[:limited_rows, :limited_cols]

            if rows <= PreviewHelper.MAX_PREVIEW_ROWS and cols <= PreviewHelper.MAX_PREVIEW_COLS:
                msg = f"显示全部 {rows} 行 × {cols} 列"
            else:
                msg = f"显示前 {limited_rows} 行 × {limited_cols} 列（原始: {rows} 行 × {cols} 列）"

            return preview, msg

        else:
            # 高维数组：展平后预览
            flat = array.ravel()
            if flat.size <= PreviewHelper.MAX_PREVIEW_ROWS:
                return flat, f"展平显示全部 {flat.size} 个元素"
            else:
                preview = flat[:PreviewHelper.MAX_PREVIEW_ROWS]
                return preview, f"展平显示前 {PreviewHelper.MAX_PREVIEW_ROWS} 个元素（共 {flat.size} 个）"

    @staticmethod
    def array_to_dataframe(array: np.ndarray) -> Optional[pd.DataFrame]:
        """
        将数组转换为 DataFrame 用于表格显示

        Returns:
            DataFrame 或 None（如果无法转换）
        """
        try:
            if array.ndim == 1:
                return pd.DataFrame({'索引': range(len(array)), '值': array})
            elif array.ndim == 2:
                rows = min(array.shape[0], PreviewHelper.MAX_PREVIEW_ROWS)
                cols = min(array.shape[1], PreviewHelper.MAX_PREVIEW_COLS)
                return pd.DataFrame(array[:rows, :cols])
            else:
                # 高维数组只取有限数量的展平预览，避免大体数据打开时整块复制。
                flat = array.ravel()
                rows = min(flat.size, PreviewHelper.MAX_PREVIEW_ROWS)
                preview = flat[:rows]
                return pd.DataFrame({'展平索引': range(rows), '值': preview})
        except Exception:
            return None

    @staticmethod
    def format_shape(shape: Tuple) -> str:
        """格式化形状显示"""
        return ' × '.join(map(str, shape))

    @staticmethod
    def format_memory(bytes_size: int) -> str:
        """格式化内存大小"""
        mb = bytes_size / (1024 * 1024)
        if mb < 1:
            kb = bytes_size / 1024
            return f"{kb:.2f} KB"
        elif mb < 1024:
            return f"{mb:.2f} MB"
        else:
            gb = mb / 1024
            return f"{gb:.2f} GB"
