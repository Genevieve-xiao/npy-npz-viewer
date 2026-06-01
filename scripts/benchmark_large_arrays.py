"""Benchmark NumPy memmap, Dask, and Zarr preview/stat/projection paths."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from npy_npz_viewer.config import ViewerConfig
from npy_npz_viewer.core.array_compute import ArrayComputeService
from npy_npz_viewer.core.loaders import ArrayLoader


def timed(name, func):
    start = time.perf_counter()
    result = func()
    return name, (time.perf_counter() - start) * 1000, result


def create_zarr_array(group, name, data, chunks):
    if hasattr(group, "create_array"):
        return group.create_array(name, data=data, chunks=chunks)
    return group.create_dataset(name, data=data, chunks=chunks)


def ensure_inputs(output_dir: Path, shape: tuple[int, int, int]):
    output_dir.mkdir(parents=True, exist_ok=True)
    npy_path = output_dir / "benchmark_volume.npy"
    zarr_path = output_dir / "benchmark_volume.zarr"
    if not npy_path.exists():
        data = np.random.default_rng(7).normal(size=shape).astype(np.float32)
        np.save(npy_path, data)
    if not zarr_path.exists():
        zarr = __import__("zarr")
        data = np.load(npy_path, mmap_mode="r")
        root = zarr.open_group(str(zarr_path), mode="w")
        create_zarr_array(root, "volume", data, chunks=(max(1, shape[0] // 4), shape[1], max(1, shape[2] // 4)))
    return npy_path, zarr_path


def run_benchmark(shape: tuple[int, int, int], output_dir: Path):
    npy_path, zarr_path = ensure_inputs(output_dir, shape)
    config = ViewerConfig(dask_threshold_bytes=1)
    rows = []

    npy_loader = ArrayLoader(config)
    name, elapsed, payload = timed("load_npy_memmap_as_dask", lambda: npy_loader.load_file(str(npy_path)))
    rows.append((name, elapsed, payload["handle"].source_type, payload["handle"].is_lazy, payload["array"].shape))
    array = payload["array"]

    for name, func in [
        ("preview_npy", lambda: ArrayComputeService.build_preview(array, "slice")),
        ("sampled_stats_npy", lambda: ArrayComputeService.compute_stats(array)),
        ("projection_npy_mean_axis0", lambda: ArrayComputeService.to_numpy(array.mean(axis=0))),
    ]:
        step, elapsed, _ = timed(name, func)
        rows.append((step, elapsed, "npy", True, array.shape))

    zarr_loader = ArrayLoader(config)
    name, elapsed, payload = timed("load_zarr_group_as_dask", lambda: zarr_loader.load_file(str(zarr_path)))
    rows.append((name, elapsed, payload["handle"].source_type, payload["handle"].is_lazy, payload["array"].shape))
    zarr_array = payload["array"]
    for name, func in [
        ("preview_zarr", lambda: ArrayComputeService.build_preview(zarr_array, "slice")),
        ("sampled_stats_zarr", lambda: ArrayComputeService.compute_stats(zarr_array)),
        ("projection_zarr_mean_axis0", lambda: ArrayComputeService.to_numpy(zarr_array.mean(axis=0))),
    ]:
        step, elapsed, _ = timed(name, func)
        rows.append((step, elapsed, "zarr", True, zarr_array.shape))
    return rows


def write_results(rows, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "benchmark_results.csv"
    md_path = output_dir / "benchmark_results.md"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "elapsed_ms", "source_type", "lazy", "shape"])
        writer.writerows(rows)
    with md_path.open("w", encoding="utf-8") as fh:
        fh.write("| step | elapsed_ms | source_type | lazy | shape |\n")
        fh.write("|---|---:|---|---|---|\n")
        for step, elapsed, source_type, lazy, shape in rows:
            fh.write(f"| {step} | {elapsed:.2f} | {source_type} | {lazy} | {shape} |\n")
    return csv_path, md_path


def parse_shape(text: str) -> tuple[int, int, int]:
    parts = tuple(int(part) for part in text.lower().replace("x", ",").split(","))
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("shape must have three dimensions, e.g. 96x128x80")
    return parts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", type=parse_shape, default=(96, 128, 80))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "benchmark_results")
    args = parser.parse_args()
    rows = run_benchmark(args.shape, args.output_dir)
    csv_path, md_path = write_results(rows, args.output_dir)
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
