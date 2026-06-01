"""Array file loading for NPY, NPZ, and Zarr sources."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from npy_npz_viewer.config import DEFAULT_CONFIG, ViewerConfig
from npy_npz_viewer.core.array_handle import (
    ArrayHandle,
    ArrayLoadResult,
    estimate_nbytes,
)

try:  # pragma: no cover - exercised when optional dependency is installed
    import dask.array as da
except Exception:  # pragma: no cover
    da = None

try:  # pragma: no cover - exercised when optional dependency is installed
    import zarr
except Exception:  # pragma: no cover
    zarr = None


logger = logging.getLogger(__name__)


class ArrayLoader:
    """Load arrays from local files/directories and keep active source state."""

    SUPPORTED_SUFFIXES = {".npy", ".npz", ".zarr"}

    def __init__(self, config: ViewerConfig = DEFAULT_CONFIG):
        self.config = config
        self.current_file: Optional[str] = None
        self.file_type: Optional[str] = None
        self.npz_data: Optional[np.lib.npyio.NpzFile] = None
        self.zarr_root: Any = None
        self.zarr_arrays: Dict[str, Any] = {}
        self.current_array: Any = None
        self.current_handle: Optional[ArrayHandle] = None
        self.current_key: Optional[str] = None
        self.keys: list[str] = []

    def load_file(self, file_path: str) -> Dict:
        """Load a .npy/.npz file or .zarr directory."""

        path = Path(file_path)
        if not path.exists():
            return ArrayLoadResult.fail("File does not exist").to_payload()

        suffix = path.suffix.lower()
        if suffix not in self.SUPPORTED_SUFFIXES:
            return ArrayLoadResult.fail("Unsupported file format. Use .npy, .npz, or .zarr.").to_payload()

        try:
            self.close()
            if suffix == ".npy":
                result = self._load_npy(path)
            elif suffix == ".npz":
                result = self._load_npz(path)
            else:
                result = self._load_zarr(path)
            return result.to_payload()
        except Exception as exc:
            logger.exception("Failed to load %s", file_path)
            return ArrayLoadResult.fail(f"File load failed: {exc}").to_payload()

    def _choose_chunks(self, shape: tuple[int, ...], dtype: Any) -> tuple[int, ...]:
        if not shape:
            return ()
        chunks = [max(1, int(dim)) for dim in shape]
        itemsize = max(1, np.dtype(dtype).itemsize)
        while np.prod(chunks, dtype=np.int64) * itemsize > self.config.dask_chunk_target_bytes:
            index = int(np.argmax(chunks))
            if chunks[index] <= 1:
                break
            chunks[index] = max(1, chunks[index] // 2)
        return tuple(chunks)

    def _wrap_array(self, array: Any, *, source_path: str, source_type: str, key: Optional[str]) -> ArrayHandle:
        shape = tuple(int(dim) for dim in getattr(array, "shape", ()))
        dtype = getattr(array, "dtype", object)
        nbytes = estimate_nbytes(shape, dtype)
        wrapped = array
        force_lazy = source_type == "zarr"
        if da is not None and (force_lazy or nbytes >= self.config.dask_threshold_bytes):
            chunks = getattr(array, "chunks", None) or self._choose_chunks(shape, dtype)
            wrapped = da.from_array(array, chunks=chunks)
            logger.info("Wrapped %s key=%s as Dask array chunks=%s", source_type, key, wrapped.chunks)
        return ArrayHandle.from_array(
            wrapped,
            source_path=source_path,
            source_type=source_type,
            key=key,
            metadata={"storage_nbytes": nbytes},
        )

    def _activate_handle(self, handle: ArrayHandle) -> None:
        self.current_handle = handle
        self.current_array = handle.array
        self.current_key = handle.key

    def _load_npy(self, path: Path) -> ArrayLoadResult:
        array = np.load(path, mmap_mode="r")
        self.current_file = str(path)
        self.file_type = "npy"
        self.keys = []
        handle = self._wrap_array(array, source_path=str(path), source_type="npy", key=None)
        self._activate_handle(handle)
        return ArrayLoadResult.ok(
            file_type="npy",
            handles={"array": handle},
            active_key="array",
            keys=[],
        )

    def _load_npz(self, path: Path) -> ArrayLoadResult:
        npz = np.load(path)
        keys = list(npz.keys())
        if not keys:
            return ArrayLoadResult.fail("NPZ file is empty.")

        self.current_file = str(path)
        self.file_type = "npz"
        self.npz_data = npz
        self.keys = keys
        return self._switch_key(keys[0])

    def _is_zarr_array(self, item: Any) -> bool:
        return hasattr(item, "shape") and hasattr(item, "dtype")

    def _collect_zarr_arrays(self, group: Any, prefix: str = "") -> Dict[str, Any]:
        arrays: Dict[str, Any] = {}
        try:
            for name, array in group.arrays():
                key = f"{prefix}/{name}" if prefix else name
                arrays[key] = array
        except Exception:
            pass

        try:
            groups = list(group.groups())
        except Exception:
            groups = []

        if groups:
            for name, subgroup in groups:
                key_prefix = f"{prefix}/{name}" if prefix else name
                arrays.update(self._collect_zarr_arrays(subgroup, key_prefix))
            return arrays

        try:
            keys = list(group.keys())
        except Exception:
            keys = []
        for name in keys:
            key = f"{prefix}/{name}" if prefix else name
            try:
                item = group[name]
            except Exception:
                continue
            if self._is_zarr_array(item):
                arrays[key] = item
            else:
                arrays.update(self._collect_zarr_arrays(item, key))
        return arrays

    def _load_zarr(self, path: Path) -> ArrayLoadResult:
        if zarr is None:
            return ArrayLoadResult.fail("Zarr support is not installed.")

        root = zarr.open(str(path), mode="r")
        self.current_file = str(path)
        self.file_type = "zarr"
        self.zarr_root = root

        if self._is_zarr_array(root):
            key = "array"
            self.zarr_arrays = {key: root}
        else:
            self.zarr_arrays = self._collect_zarr_arrays(root)

        keys = list(self.zarr_arrays.keys())
        if not keys:
            return ArrayLoadResult.fail("Zarr store does not contain arrays.")
        self.keys = keys
        return self._switch_key(keys[0])

    def _switch_key(self, key: str) -> ArrayLoadResult:
        if self.file_type == "npz":
            if self.npz_data is None:
                return ArrayLoadResult.fail("Current source is not an NPZ file.")
            if key not in self.npz_data.keys():
                return ArrayLoadResult.fail(f'Key "{key}" does not exist.')
            array = self.npz_data[key]
            handle = self._wrap_array(array, source_path=self.current_file or "", source_type="npz", key=key)
        elif self.file_type == "zarr":
            if key not in self.zarr_arrays:
                return ArrayLoadResult.fail(f'Zarr array "{key}" does not exist.')
            array = self.zarr_arrays[key]
            handle = self._wrap_array(array, source_path=self.current_file or "", source_type="zarr", key=key)
        else:
            return ArrayLoadResult.fail("Current source does not support key switching.")

        self._activate_handle(handle)
        return ArrayLoadResult.ok(
            file_type=self.file_type or "",
            handles={key: handle},
            active_key=key,
            keys=self.keys,
        )

    def switch_npz_key(self, key: str) -> Dict:
        """Switch the active NPZ/Zarr array key."""

        return self._switch_key(key).to_payload()

    def get_current_array(self) -> Any:
        return self.current_array

    def get_current_handle(self) -> Optional[ArrayHandle]:
        return self.current_handle

    def close(self) -> None:
        if self.npz_data is not None:
            self.npz_data.close()
        self.current_file = None
        self.file_type = None
        self.npz_data = None
        self.zarr_root = None
        self.zarr_arrays = {}
        self.current_array = None
        self.current_handle = None
        self.current_key = None
        self.keys = []
