"""Reading videos (zarr), annotations and metadata (geff)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr


def list_videos(data_dir: str | Path) -> list[str]:
    return sorted(p.stem for p in Path(data_dir).glob("*.zarr"))


def open_video(data_dir: str | Path, name: str):
    """-> zarr array (T, Z, Y, X)"""
    return zarr.open(str(Path(data_dir) / f"{name}.zarr"), mode="r")["0"]


def load_gt(geff_path: str | Path) -> dict[int, np.ndarray]:
    """-> {t: annotated centres in voxels, int array (N, 3) as z, y, x}"""
    import tracksdata as td                                      # imported lazily: only needed when real annotations are read
    g = td.graph.IndexedRXGraph.from_geff(str(geff_path))
    g = g[0] if isinstance(g, tuple) else g
    df = g.node_attrs()
    if df.height == 0 or not {"z", "y", "x"} <= set(df.columns):      # a video without annotated cells: the .geff holds no positions at all
        return {}
    nd = df.select(["t", "z", "y", "x"]).to_numpy().astype(np.float64)
    out: dict[int, list] = {}
    for r in nd:
        out.setdefault(int(r[0]), []).append(np.round(r[1:4]))
    return {t: np.array(v, np.int32) for t, v in out.items()}


def find_estimate(geff_path: str | Path):
    """The organisers' estimated number of cells (summed over the frames of this video), stored in the .geff metadata. None if absent."""
    def walk(d):
        if isinstance(d, dict):
            for k, v in d.items():
                if "estimat" in str(k).lower() and isinstance(v, (int, float)):
                    return v
                r = walk(v)
                if r is not None:
                    return r
        return None
    return walk(dict(zarr.open_group(str(geff_path), mode="r").attrs))


def neighbour_ts(t: int, T: int, before: int, after: int) -> list[int]:
    """Frame indices t-before .. t+after, clamped to the video (the first / last frame repeats at the ends)."""
    return [min(max(t + d, 0), T - 1) for d in range(-before, after + 1)]


def read_stack_crop(arr, ts: list[int], lo, size) -> np.ndarray:
    """arr[ts, lo:lo+size] -> (F, z, y, x) float32, reflect padding where the crop leaves the volume."""
    dims = arr.shape[1:]
    a = [max(0, int(lo[i])) for i in range(3)]
    b = [min(dims[i], int(lo[i]) + size[i]) for i in range(3)]
    x = np.stack([np.asarray(arr[t, a[0]:b[0], a[1]:b[1], a[2]:b[2]], np.float32) for t in ts])
    pad = [(0, 0)] + [(a[i] - int(lo[i]), int(lo[i]) + size[i] - b[i]) for i in range(3)]
    return np.pad(x, pad, mode="reflect") if any(p != (0, 0) for p in pad) else x


def read_stack_full(arr, ts: list[int]) -> np.ndarray:
    """whole volumes of the frames ts -> (F, Z, Y, X) float32"""
    return np.stack([np.asarray(arr[t], np.float32) for t in ts])
