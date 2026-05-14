"""
Array session and view state.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from core.data_semantics import DataSemantics
from core.dimension_filter import apply_dimension_filter
from core.semantic_inference import SemanticInference
from core.slicing import ArraySlicer
from core.task_result import TaskResult


@dataclass
class ArrayViewSpec:
    """Serializable description of the current array view."""

    source_path: Optional[str] = None
    npz_key: Optional[str] = None
    axis_filters: List[Dict[str, Any]] = field(default_factory=list)
    slice_specs: List[str] = field(default_factory=list)
    axis_maps: Dict[int, List[int]] = field(default_factory=dict)
    semantic_override: Optional[str] = None


class ArraySession:
    """Owns original/current array state for the active file/key."""

    def __init__(self):
        self.spec = ArrayViewSpec()
        self.original_array: Optional[np.ndarray] = None
        self.filtered_array: Optional[np.ndarray] = None
        self.current_array: Optional[np.ndarray] = None
        self.semantic_info: Optional[Dict[str, Any]] = None

    def load_array(self, array: np.ndarray, source_path: str = None,
                   npz_key: str = None) -> TaskResult:
        self.spec = ArrayViewSpec(source_path=source_path, npz_key=npz_key)
        self.original_array = array
        self.filtered_array = array
        self.current_array = array
        self.semantic_info = SemanticInference.infer(array)
        return TaskResult.ok(
            data=array,
            metadata={"semantic_info": self.semantic_info, "spec": self.spec},
        )

    def apply_filters(self, axis_filters: List[Dict[str, Any]]) -> TaskResult:
        if self.original_array is None:
            return TaskResult.fail("未加载数组")

        result = apply_dimension_filter(self.original_array, axis_filters)
        if not result["success"]:
            return TaskResult.fail(result["error"])

        self.spec.axis_filters = axis_filters
        self.spec.axis_maps = result.get("axis_index_maps", {})
        self.spec.slice_specs = []
        self.filtered_array = result["array"]
        self.current_array = self.filtered_array
        self.semantic_info = SemanticInference.infer(self.current_array)
        return TaskResult.ok(
            data=self.current_array,
            metadata={
                "summary": result.get("summary"),
                "axis_maps": self.spec.axis_maps,
                "semantic_info": self.semantic_info,
                "spec": self.spec,
            },
        )

    def reset_filters(self) -> TaskResult:
        return self.apply_filters([])

    def apply_slice(self, slice_specs: List[str]) -> TaskResult:
        if self.filtered_array is None:
            return TaskResult.fail("未加载数组")

        result = ArraySlicer.apply_slice(self.filtered_array, slice_specs)
        if not result["success"]:
            return TaskResult.fail(result["error"])

        self.spec.slice_specs = slice_specs
        self.current_array = result["array"]
        self.semantic_info = SemanticInference.infer(self.current_array)
        return TaskResult.ok(
            data=self.current_array,
            metadata={"semantic_info": self.semantic_info, "spec": self.spec},
        )

    def set_current_array(self, array: np.ndarray) -> TaskResult:
        self.current_array = array
        self.semantic_info = SemanticInference.infer(array)
        return TaskResult.ok(
            data=array,
            metadata={"semantic_info": self.semantic_info, "spec": self.spec},
        )

    def current_semantic(self) -> DataSemantics:
        if not self.semantic_info:
            return DataSemantics.UNKNOWN
        return self.semantic_info.get("semantic", DataSemantics.UNKNOWN)
