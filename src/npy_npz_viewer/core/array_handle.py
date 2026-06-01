"""Shared interfaces for NumPy, Dask, and Zarr-backed arrays."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np


def is_lazy_array(array: Any) -> bool:
    """Return True for Dask-like lazy arrays without importing Dask eagerly."""

    module = type(array).__module__
    return module.startswith("dask.") or (
        hasattr(array, "compute") and hasattr(array, "chunks")
    )


def estimate_nbytes(shape: Tuple[int, ...], dtype: Any) -> int:
    """Estimate array memory footprint from shape and dtype."""

    try:
        return int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize
    except Exception:
        return 0


@dataclass
class ArrayHandle:
    """Uniform descriptor for arrays regardless of storage backend."""

    source_path: Optional[str]
    source_type: str
    key: Optional[str]
    array: Any
    shape: Tuple[int, ...]
    dtype: Any
    ndim: int
    chunks: Optional[Any] = None
    is_lazy: bool = False
    nbytes_estimate: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_array(
        cls,
        array: Any,
        *,
        source_path: Optional[str],
        source_type: str,
        key: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "ArrayHandle":
        shape = tuple(int(dim) for dim in getattr(array, "shape", ()))
        dtype = getattr(array, "dtype", object)
        return cls(
            source_path=source_path,
            source_type=source_type,
            key=key,
            array=array,
            shape=shape,
            dtype=dtype,
            ndim=len(shape),
            chunks=getattr(array, "chunks", None),
            is_lazy=is_lazy_array(array),
            nbytes_estimate=estimate_nbytes(shape, dtype),
            metadata=dict(metadata or {}),
        )


@dataclass
class ArrayLoadResult:
    """Structured result for file loading and key switching."""

    success: bool
    file_type: Optional[str] = None
    handles: Dict[str, ArrayHandle] = field(default_factory=dict)
    active_key: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def keys(self) -> list[str]:
        return list(self.metadata.get("keys", self.handles.keys()))

    @property
    def current_handle(self) -> Optional[ArrayHandle]:
        if self.active_key is None:
            return None
        return self.handles.get(self.active_key)

    @classmethod
    def ok(
        cls,
        *,
        file_type: str,
        handles: Dict[str, ArrayHandle],
        active_key: Optional[str],
        **metadata: Any,
    ) -> "ArrayLoadResult":
        return cls(
            success=True,
            file_type=file_type,
            handles=handles,
            active_key=active_key,
            metadata=metadata,
        )

    @classmethod
    def fail(cls, error: str) -> "ArrayLoadResult":
        return cls(success=False, error=error)

    def to_payload(self) -> Dict[str, Any]:
        """Return the legacy dict shape used by the current GUI callbacks."""

        handle = self.current_handle
        payload: Dict[str, Any] = {
            "success": self.success,
            "file_type": self.file_type,
            "keys": self.keys,
            "handle": handle,
            "load_result": self,
            "error": self.error,
        }
        if handle is not None:
            payload["array"] = handle.array
        return payload
