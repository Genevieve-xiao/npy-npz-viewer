"""
Lightweight verification for core non-GUI functionality.
"""
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.array_session import ArraySession
from core.dimension_filter import apply_dimension_filter
from core.loaders import ArrayLoader
from core.slicing import ArraySlicer
from core.stats import ArrayStats
from core.task_result import TaskResult
from utils.helpers import PreviewHelper


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    print(f"OK: {message}")


def main():
    loader = ArrayLoader()

    result = loader.load_file(str(DATA_DIR / "test_1d.npy"))
    check(result["success"], "NPY smoke fixture loads")

    result = loader.load_file(str(DATA_DIR / "test_data.npz"))
    check(result["success"] and len(result["keys"]) >= 1, "NPZ smoke fixture loads")

    array = np.arange(5000, dtype=np.float32).reshape(100, 50)
    stats = ArrayStats.compute_stats(array)
    check(stats["shape"] == (100, 50), "statistics report shape")
    check(stats["min"] == 0.0 and stats["max"] == 4999.0, "statistics report range")

    sliced = ArraySlicer.apply_slice(array, [":10", ":20"])
    check(sliced["success"] and sliced["array"].shape == (10, 20), "slicing works")

    filtered = apply_dimension_filter(
        array,
        [
            {"axis": 0, "mode": "keep", "spec": "0:10:2"},
            {"axis": 1, "mode": "drop", "spec": "0,2,-1"},
        ],
    )
    check(
        filtered["success"]
        and filtered["array"].shape == (5, 47)
        and filtered["axis_index_maps"][0] == [0, 2, 4, 6, 8],
        "dimension filter keep/drop works",
    )

    empty = apply_dimension_filter(array, [{"axis": 1, "mode": "drop", "spec": ":"}])
    check(not empty["success"], "empty dimension filter result is rejected")

    seismic_like = np.zeros((571551, 1, 288, 1), dtype=np.float32)
    singleton = apply_dimension_filter(
        seismic_like,
        [
            {"axis": 0, "mode": "keep", "spec": "0:2000:2"},
            {"axis": 1, "mode": "drop", "spec": ":"},
            {"axis": 3, "mode": "drop", "spec": ":"},
        ],
    )
    check(
        singleton["success"] and singleton["array"].shape == (1000, 288),
        "singleton axes can be removed",
    )

    preview, message = PreviewHelper.get_preview_slice(array)
    check(preview is not None and "100" in message, "basic preview works")

    cube = np.arange(20 * 12 * 8, dtype=np.float32).reshape(20, 12, 8)
    df, _ = PreviewHelper.build_preview(cube, mode="slice")
    check(df is not None and df.shape == (20, 12), "3D slice preview works")

    df, _ = PreviewHelper.build_preview(cube, mode="summary")
    check(df is not None and len(df) == 3, "axis summary preview works")

    session = ArraySession()
    loaded = session.load_array(cube, source_path="demo.npy")
    filtered = session.apply_filters(
        [
            {"axis": 0, "mode": "keep", "spec": "0:10:2"},
            {"axis": 2, "mode": "drop", "spec": "0,-1"},
        ]
    )
    check(
        loaded.success and filtered.success and session.current_array.shape == (5, 12, 6),
        "ArraySession state flow works",
    )

    task_result = TaskResult.ok(data={"value": 1}, sampled=True)
    check(task_result.success and task_result.sampled and task_result.data["value"] == 1, "TaskResult works")

    print("\nAll core checks passed.")


if __name__ == "__main__":
    main()
