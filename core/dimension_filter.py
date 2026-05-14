"""
Dimension filtering utilities.
"""
from typing import Dict, Iterable, List

import numpy as np


class DimensionFilter:
    """Apply keep/drop filters to one or more array axes."""

    VALID_MODES = {"keep", "drop"}

    @staticmethod
    def _normalize_index(index: int, size: int) -> int:
        if index < 0:
            index += size
        if index < 0 or index >= size:
            raise ValueError(f"索引 {index} 超出范围 [0, {size - 1}]")
        return index

    @staticmethod
    def _parse_part(part: str, size: int) -> Iterable[int]:
        part = part.strip()
        if not part:
            raise ValueError("包含空的索引片段")

        if ":" in part:
            pieces = part.split(":")
            if len(pieces) > 3:
                raise ValueError(f"切片格式错误: {part}")
            try:
                start = int(pieces[0]) if pieces[0].strip() else None
                stop = int(pieces[1]) if len(pieces) > 1 and pieces[1].strip() else None
                step = int(pieces[2]) if len(pieces) > 2 and pieces[2].strip() else None
            except ValueError as exc:
                raise ValueError(f"切片中包含非整数: {part}") from exc
            if step == 0:
                raise ValueError("切片步长不能为 0")
            return np.arange(size)[slice(start, stop, step)].tolist()

        try:
            return [DimensionFilter._normalize_index(int(part), size)]
        except ValueError as exc:
            raise ValueError(f"索引格式错误: {part}") from exc

    @staticmethod
    def parse_selection(spec: str, size: int) -> List[int]:
        """
        Parse an axis selection.

        Supported forms:
        - ":" or empty for all indices
        - "0,2,5" for explicit indices
        - "1:10:2" for Python slice syntax
        - mixed comma parts such as "0,3:7,10"
        """
        text = (spec or ":").strip()
        if text == ":":
            return list(range(size))

        indices: List[int] = []
        for part in text.split(","):
            indices.extend(DimensionFilter._parse_part(part, size))

        if not indices:
            raise ValueError("选择结果为空")

        # Keep first occurrence order but avoid duplicated data.
        seen = set()
        unique_indices = []
        for index in indices:
            if index not in seen:
                seen.add(index)
                unique_indices.append(index)
        return unique_indices

    @staticmethod
    def apply_dimension_filter(array: np.ndarray, axis_filters: List[Dict]) -> Dict:
        """
        Apply keep/drop filters to array axes.

        Args:
            array: source ndarray
            axis_filters: dicts with axis, mode ("keep"/"drop"), spec
        """
        if array is None:
            return {"success": False, "error": "未加载数组"}

        if not axis_filters:
            return {
                "success": True,
                "array": array,
                "axis_index_maps": {},
                "summary": "未应用维度筛选",
            }

        filtered = array
        current_axes = list(range(array.ndim))
        axis_index_maps = {}
        summaries = []
        seen_axes = set()

        for raw_filter in axis_filters:
            try:
                axis = int(raw_filter.get("axis"))
            except (TypeError, ValueError):
                return {"success": False, "error": f"轴编号无效: {raw_filter.get('axis')}"}

            if axis < 0:
                axis += array.ndim
            if axis < 0 or axis >= array.ndim:
                return {"success": False, "error": f"轴 {axis} 超出范围 [0, {array.ndim - 1}]"}
            if axis in seen_axes:
                return {"success": False, "error": f"轴 {axis} 被重复筛选"}
            seen_axes.add(axis)

            mode = raw_filter.get("mode", "keep")
            if mode not in DimensionFilter.VALID_MODES:
                return {"success": False, "error": f"未知筛选模式: {mode}"}

            if axis not in current_axes:
                return {"success": False, "error": f"轴 {axis} 已被前面的筛选移除"}

            current_axis = current_axes.index(axis)
            size = filtered.shape[current_axis]
            spec = raw_filter.get("spec", ":")

            try:
                selected = DimensionFilter.parse_selection(spec, size)
            except ValueError as exc:
                return {"success": False, "error": f"轴 {axis}: {exc}"}

            if mode == "drop":
                if len(selected) == size:
                    if size != 1:
                        return {
                            "success": False,
                            "error": f"轴 {axis} 有 {size} 个索引，不能直接整轴剔除；请先保留单个索引、切片降维，或选择部分索引剔除",
                        }
                    filtered = np.squeeze(filtered, axis=current_axis)
                    current_axes.pop(current_axis)
                    axis_index_maps[axis] = []
                    summaries.append(f"轴 {axis} 剔除整轴: {size} -> 移除")
                    continue

                dropped = set(selected)
                final_indices = [i for i in range(size) if i not in dropped]
            else:
                final_indices = selected

            if not final_indices:
                if mode == "drop" and size == 1:
                    filtered = np.squeeze(filtered, axis=current_axis)
                    current_axes.pop(current_axis)
                    axis_index_maps[axis] = []
                    summaries.append(f"轴 {axis} 剔除整轴: {size} -> 移除")
                    continue
                return {"success": False, "error": f"轴 {axis} 的筛选结果为空"}

            try:
                filtered = np.take(filtered, final_indices, axis=current_axis)
            except Exception as exc:
                return {"success": False, "error": f"轴 {axis} 筛选失败: {exc}"}

            axis_index_maps[axis] = final_indices
            action = "保留" if mode == "keep" else "剔除"
            summaries.append(f"轴 {axis} {action} {spec}: {size} -> {len(final_indices)}")

        return {
            "success": True,
            "array": filtered,
            "axis_index_maps": axis_index_maps,
            "summary": "\n".join(summaries),
        }


def apply_dimension_filter(array: np.ndarray, axis_filters: List[Dict]) -> Dict:
    """Public helper matching the planned interface."""
    return DimensionFilter.apply_dimension_filter(array, axis_filters)
