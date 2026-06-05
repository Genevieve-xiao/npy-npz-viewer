"""Runtime configuration for array loading, preview, and plotting."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ViewerConfig:
    """Central tuning knobs for large array behavior."""

    dask_threshold_bytes: int = 128 * 1024 * 1024
    dask_chunk_target_bytes: int = 64 * 1024 * 1024
    preview_max_rows: int = 1000
    preview_max_cols: int = 20
    stats_large_array_threshold: int = 10_000_000
    stats_sample_size: int = 200_000
    stats_sample_chunks: int = 10
    plot_max_points_1d: int = 10_000
    plot_max_points_2d: int = 100_000
    plot_max_rows_heatmap: int = 2000
    plot_max_cols_heatmap: int = 2000
    voxel_max_axis_samples: int = 32
    voxel_max_voxels: int = 1800
    voxel_percentile_threshold: float = 85.0


DEFAULT_CONFIG = ViewerConfig()
