from __future__ import annotations

import importlib
import runpy

import numpy as np
import pytest

from npy_npz_viewer.config import ViewerConfig
from npy_npz_viewer.core.array_compute import ArrayComputeService
from npy_npz_viewer.core.dimension_filter import apply_dimension_filter
from npy_npz_viewer.core.loaders import ArrayLoader
from npy_npz_viewer.core.projection import ArrayProjection
from npy_npz_viewer.core.slicing import ArraySlicer


def test_npy_loader_reports_handle_metadata(tmp_path):
    path = tmp_path / "small.npy"
    np.save(path, np.arange(12, dtype=np.float32).reshape(3, 4))

    payload = ArrayLoader().load_file(str(path))

    assert payload["success"]
    assert payload["file_type"] == "npy"
    assert payload["handle"].shape == (3, 4)
    assert payload["handle"].source_type == "npy"
    assert not payload["handle"].is_lazy


def test_large_npy_uses_dask_when_threshold_is_exceeded(tmp_path):
    path = tmp_path / "large.npy"
    np.save(path, np.arange(12, dtype=np.float32).reshape(3, 4))
    loader = ArrayLoader(ViewerConfig(dask_threshold_bytes=1))

    payload = loader.load_file(str(path))

    assert payload["success"]
    assert payload["handle"].is_lazy
    assert payload["handle"].chunks is not None
    stats = ArrayComputeService.compute_stats(payload["array"])
    assert stats["min"] == 0.0
    assert stats["max"] == 11.0


def test_npz_key_switching_preserves_lazy_interface(tmp_path):
    path = tmp_path / "suite.npz"
    np.savez(path, first=np.arange(6), second=np.arange(12).reshape(3, 4))
    loader = ArrayLoader()

    payload = loader.load_file(str(path))
    switched = loader.switch_npz_key("second")

    assert payload["success"]
    assert payload["keys"] == ["first", "second"]
    assert switched["success"]
    assert switched["handle"].key == "second"
    assert switched["array"].shape == (3, 4)


def _create_zarr_array(group, name, data, chunks):
    if hasattr(group, "create_array"):
        return group.create_array(name, data=data, chunks=chunks)
    return group.create_dataset(name, data=data, chunks=chunks)


def test_zarr_group_load_and_switch(tmp_path):
    zarr = pytest.importorskip("zarr")
    store = tmp_path / "demo.zarr"
    root = zarr.open_group(str(store), mode="w")
    _create_zarr_array(root, "image", np.arange(20).reshape(4, 5), chunks=(2, 5))
    nested = root.create_group("nested")
    _create_zarr_array(nested, "volume", np.arange(24).reshape(2, 3, 4), chunks=(1, 3, 4))

    loader = ArrayLoader()
    payload = loader.load_file(str(store))
    switched = loader.switch_npz_key("nested/volume")

    assert payload["success"]
    assert payload["file_type"] == "zarr"
    assert set(payload["keys"]) == {"image", "nested/volume"}
    assert switched["success"]
    assert switched["handle"].is_lazy
    assert switched["array"].shape == (2, 3, 4)


def test_dask_and_numpy_core_operations_match():
    da = pytest.importorskip("dask.array")
    numpy_array = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
    dask_array = da.from_array(numpy_array, chunks=(2, 5, 3))

    sliced = ArraySlicer.apply_slice(dask_array, [":2", "1:4", "::2"])
    assert sliced["success"]
    np.testing.assert_array_equal(ArrayComputeService.to_numpy(sliced["array"]), numpy_array[:2, 1:4, ::2])

    filtered = apply_dimension_filter(dask_array, [{"axis": 2, "mode": "keep", "spec": "0,2,4"}])
    assert filtered["success"]
    np.testing.assert_array_equal(ArrayComputeService.to_numpy(filtered["array"]), numpy_array[:, :, [0, 2, 4]])

    projected = ArrayProjection.project(dask_array, axis=0, method="mean")
    assert projected["success"]
    np.testing.assert_allclose(ArrayComputeService.to_numpy(projected["array"]), numpy_array.mean(axis=0))

    preview, _ = ArrayComputeService.build_preview(dask_array, mode="slice")
    assert preview is not None
    assert preview.shape == (4, 5)


def test_app_entrypoints_are_importable():
    module = importlib.import_module("npy_npz_viewer.app")
    assert hasattr(module, "main")
    namespace = runpy.run_path("main.py")
    assert "main" in namespace
