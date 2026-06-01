"""Bounded compute helpers for eager and lazy arrays."""

from __future__ import annotations

import logging
import time
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd

from npy_npz_viewer.config import DEFAULT_CONFIG, ViewerConfig
from npy_npz_viewer.core.array_handle import is_lazy_array

try:  # pragma: no cover - exercised when optional dependency is installed
    import dask.array as da
except Exception:  # pragma: no cover
    da = None


logger = logging.getLogger(__name__)


class ArrayComputeService:
    """Compute bounded NumPy results from NumPy, Dask, or Zarr-backed arrays."""

    @staticmethod
    def is_lazy_array(array: Any) -> bool:
        return is_lazy_array(array)

    @staticmethod
    def to_numpy(array: Any) -> np.ndarray:
        """Materialize an array-like object as a NumPy array."""

        if is_lazy_array(array):
            start = time.perf_counter()
            result = array.compute()
            logger.info("Computed lazy array slice in %.1f ms", (time.perf_counter() - start) * 1000)
            return np.asarray(result)
        return np.asarray(array)

    @staticmethod
    def shape(array: Any) -> Tuple[int, ...]:
        return tuple(int(dim) for dim in getattr(array, "shape", ()))

    @staticmethod
    def size(array: Any) -> int:
        shape = ArrayComputeService.shape(array)
        if not shape:
            return 1
        return int(np.prod(shape, dtype=np.int64))

    @staticmethod
    def build_preview(
        array: Any,
        mode: str = "auto",
        start: int = 0,
        end: int = 1000,
        config: ViewerConfig = DEFAULT_CONFIG,
    ) -> Tuple[Optional[pd.DataFrame], str]:
        """Build a preview without materializing more than the configured bounds."""

        try:
            shape = ArrayComputeService.shape(array)
            ndim = len(shape)
            if ndim == 0:
                value = ArrayComputeService.to_numpy(array).item()
                return pd.DataFrame({"value": [value]}), "Scalar data"

            if mode == "auto":
                mode = "slice" if ndim >= 3 else "table"

            if mode == "summary":
                rows = [
                    {
                        "axis": axis,
                        "length": size,
                        "middle_index": size // 2,
                        "suggested_slice": f"{size // 2}",
                    }
                    for axis, size in enumerate(shape)
                ]
                return pd.DataFrame(rows), f"Axis summary: {shape}"

            if mode == "slice" and ndim >= 3:
                indexer = []
                fixed_axes = []
                open_axes = []
                for axis, size in enumerate(shape):
                    if len(open_axes) < 2:
                        limit = config.preview_max_rows if len(open_axes) == 0 else config.preview_max_cols
                        indexer.append(slice(0, min(size, limit)))
                        open_axes.append(axis)
                    else:
                        mid = size // 2
                        indexer.append(mid)
                        fixed_axes.append((axis, mid))
                data = ArrayComputeService.to_numpy(array[tuple(indexer)])
                data = np.squeeze(data)
                if data.ndim != 2:
                    return ArrayComputeService.build_preview(array, "flat", start, end, config)
                info = f"Slice preview: axis {open_axes[0]} x axis {open_axes[1]}"
                if fixed_axes:
                    info += "; fixed " + ", ".join(f"axis {axis}={idx}" for axis, idx in fixed_axes)
                info += f"; showing {data.shape[0]} x {data.shape[1]} from original {shape}"
                return pd.DataFrame(data), info

            if mode == "flat" or ndim > 2:
                total = ArrayComputeService.size(array)
                start = max(0, min(start, total))
                end = max(start, min(end, total, start + config.preview_max_rows))
                flat = array.reshape((total,)) if hasattr(array, "reshape") else np.ravel(array)
                data = ArrayComputeService.to_numpy(flat[start:end])
                return (
                    pd.DataFrame({"flat_index": range(start, end), "value": data}),
                    f"Flat preview {start}..{end - 1} of {total} elements",
                )

            if ndim == 1:
                total = shape[0]
                start = max(0, min(start, total))
                end = max(start, min(end, total, start + config.preview_max_rows))
                data = ArrayComputeService.to_numpy(array[start:end])
                return (
                    pd.DataFrame({"index": range(start, end), "value": data}),
                    f"Showing elements {start}..{end - 1} of {total}",
                )

            rows = shape[0]
            cols = min(shape[1], config.preview_max_cols)
            start = max(0, min(start, rows))
            end = max(start, min(end, rows, start + config.preview_max_rows))
            data = ArrayComputeService.to_numpy(array[start:end, :cols])
            return (
                pd.DataFrame(data),
                f"Showing rows {start}..{end - 1} x first {cols} columns from {shape[0]} x {shape[1]}",
            )
        except Exception as exc:
            logger.exception("Preview generation failed")
            return None, f"Preview generation failed: {exc}"

    @staticmethod
    def _sample_large_array(array: Any, config: ViewerConfig = DEFAULT_CONFIG) -> np.ndarray:
        total = ArrayComputeService.size(array)
        sample_size = min(config.stats_sample_size, total)
        if sample_size <= 0:
            return np.asarray([], dtype=getattr(array, "dtype", float))

        if total <= sample_size:
            return ArrayComputeService.to_numpy(array)

        chunk_count = min(config.stats_sample_chunks, sample_size)
        chunk_size = max(1, sample_size // chunk_count)
        max_start = max(0, total - chunk_size)
        starts = np.linspace(0, max_start, chunk_count, dtype=np.int64)
        flat = array.reshape((total,)) if hasattr(array, "reshape") else np.ravel(array)
        chunks = [flat[int(start):int(start) + chunk_size] for start in starts]

        if is_lazy_array(array) and da is not None:
            sample = da.concatenate(chunks)
            return ArrayComputeService.to_numpy(sample[:sample_size])

        sample = np.concatenate([ArrayComputeService.to_numpy(chunk) for chunk in chunks])
        return sample[:sample_size]

    @staticmethod
    def compute_stats(array: Any, config: ViewerConfig = DEFAULT_CONFIG) -> dict:
        """Compute numeric stats, sampling large arrays to keep UI responsive."""

        shape = ArrayComputeService.shape(array)
        dtype = getattr(array, "dtype", object)
        size = ArrayComputeService.size(array)
        stats = {
            "shape": shape,
            "dtype": str(dtype),
            "ndim": len(shape),
            "size": size,
            "memory_mb": (size * np.dtype(dtype).itemsize / (1024 * 1024))
            if np.dtype(dtype) != np.dtype(object)
            else 0.0,
            "is_numeric": np.issubdtype(dtype, np.number),
            "sampled": False,
            "lazy": is_lazy_array(array),
        }

        if not stats["is_numeric"]:
            stats.update({"has_nan": False, "has_inf": False, "min": None, "max": None, "mean": None, "std": None})
            return stats

        start = time.perf_counter()
        sample = (
            ArrayComputeService._sample_large_array(array, config)
            if size > config.stats_large_array_threshold
            else ArrayComputeService.to_numpy(array)
        )
        stats["sampled"] = size > config.stats_large_array_threshold
        logger.info(
            "Computed stats input sample in %.1f ms lazy=%s sampled=%s shape=%s",
            (time.perf_counter() - start) * 1000,
            stats["lazy"],
            stats["sampled"],
            shape,
        )

        try:
            stats["has_nan"] = bool(np.isnan(sample).any())
            stats["has_inf"] = bool(np.isinf(sample).any())
            valid_data = sample[np.isfinite(sample)]
            if valid_data.size:
                stats["min"] = float(valid_data.min())
                stats["max"] = float(valid_data.max())
                stats["mean"] = float(valid_data.mean())
                stats["std"] = float(valid_data.std())
            else:
                stats.update({"min": None, "max": None, "mean": None, "std": None})
        except Exception:
            logger.exception("Stats calculation failed")
            stats.update({"has_nan": False, "has_inf": False, "min": None, "max": None, "mean": None, "std": None})
        return stats

    @staticmethod
    def downsample_1d_for_plot(
        array: Any, config: ViewerConfig = DEFAULT_CONFIG
    ) -> Tuple[np.ndarray, str]:
        size = ArrayComputeService.size(array)
        if size <= config.plot_max_points_1d:
            return ArrayComputeService.to_numpy(array), ""
        indices = np.linspace(0, size - 1, config.plot_max_points_1d, dtype=np.int64)
        data = ArrayComputeService.to_numpy(array[indices])
        return data, f"Data downsampled to {config.plot_max_points_1d:,} points from {size:,}."

    @staticmethod
    def downsample_2d_for_plot(
        array: Any,
        x_col: Optional[Any] = None,
        config: ViewerConfig = DEFAULT_CONFIG,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], str]:
        rows = int(array.shape[0])
        if rows <= config.plot_max_points_1d:
            data = ArrayComputeService.to_numpy(array)
            x_data = ArrayComputeService.to_numpy(x_col) if x_col is not None else None
            return data, x_data, ""
        indices = np.linspace(0, rows - 1, config.plot_max_points_1d, dtype=np.int64)
        data = ArrayComputeService.to_numpy(array[indices, :])
        x_data = ArrayComputeService.to_numpy(x_col[indices]) if x_col is not None else None
        return data, x_data, f"Rows downsampled to {config.plot_max_points_1d:,} from {rows:,}."

    @staticmethod
    def downsample_scatter(
        x_data: Any, y_data: Any, config: ViewerConfig = DEFAULT_CONFIG
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        rows = len(x_data)
        if rows <= config.plot_max_points_2d:
            return ArrayComputeService.to_numpy(x_data), ArrayComputeService.to_numpy(y_data), ""
        indices = np.random.choice(rows, config.plot_max_points_2d, replace=False)
        indices = np.sort(indices)
        return (
            ArrayComputeService.to_numpy(x_data[indices]),
            ArrayComputeService.to_numpy(y_data[indices]),
            f"Scatter data sampled to {config.plot_max_points_2d:,} from {rows:,} points.",
        )

    @staticmethod
    def downsample_heatmap(
        array: Any, config: ViewerConfig = DEFAULT_CONFIG
    ) -> Tuple[np.ndarray, str]:
        rows, cols = array.shape
        if rows <= config.plot_max_rows_heatmap and cols <= config.plot_max_cols_heatmap:
            return ArrayComputeService.to_numpy(array), ""
        row_step = max(1, int(np.ceil(rows / config.plot_max_rows_heatmap)))
        col_step = max(1, int(np.ceil(cols / config.plot_max_cols_heatmap)))
        data = ArrayComputeService.to_numpy(array[::row_step, ::col_step])
        return data, f"Heatmap downsampled from {rows} x {cols} to {data.shape[0]} x {data.shape[1]}."

    @staticmethod
    def slice_to_numpy(array: Any, indexer: Any) -> np.ndarray:
        return ArrayComputeService.to_numpy(array[indexer])
