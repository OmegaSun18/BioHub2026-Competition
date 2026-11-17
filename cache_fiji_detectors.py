import glob
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Tuple

import imagej
import numpy as np
import pandas as pd
import scyjava
import zarr

# Ensure the repository root (containing `fiji_detectors` and `fiji_trackers`) is on sys.path
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from fiji_detectors.dog_detector import detect_spots_dog
from fiji_detectors.log_detector import detect_spots_log


# ==============================================================================
# 1. Configuration (Matches fiji_run_test.py)
# ==============================================================================
TEST_DATA_DIR = "/kaggle/working/split_dataset/test"
BASE_OUTPUT_DIR = "/kaggle/working"
FIJI_PATH = "/opt/Fiji.app"

RANDOM_SEED = 8001
SAMPLE_FRACTION = 0.20
DEFAULT_SCALE_ZYX = (1.625, 0.40625, 0.40625)  # Fallback voxel spacing (dz, dy, dx) in um

# Detector Hyperparameters
DETECTOR_RADIUS = 2.5
DETECTOR_THRESHOLD = 5.0


# ==============================================================================
# 2. Dataset Sampling & Zarr Helpers (Reused from fiji_run_test.py)
# ==============================================================================
def sample_zarr_datasets(
    data_dir: str,
    fraction: float = 0.20,
    seed: int = 8001,
) -> List[str]:
    """Deterministically sample a fraction of .zarr datasets using `seed`."""
    all_zarrs = sorted(glob.glob(os.path.join(data_dir, "*.zarr")))
    if not all_zarrs:
        raise FileNotFoundError(f"No .zarr files found in {data_dir}")

    n_sample = max(1, int(round(len(all_zarrs) * fraction)))
    rng = np.random.default_rng(seed)
    chosen_indices = rng.choice(len(all_zarrs), size=n_sample, replace=False)
    sampled_paths = sorted(all_zarrs[i] for i in chosen_indices)
    return sampled_paths


def load_zarr_volume_and_scale(
    zarr_path: str,
    default_scale_zyx: Tuple[float, float, float] = DEFAULT_SCALE_ZYX,
) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """Load full-resolution array `'0'` and read OME-Zarr ZYX voxel scale if present."""
    root = zarr.open(zarr_path, mode="r")
    vol = np.asarray(root["0"])

    scale_zyx = default_scale_zyx
    try:
        multiscales = dict(root.attrs).get("multiscales", [])
        if multiscales:
            datasets = multiscales[0].get("datasets", [])
            for ds in datasets:
                if ds.get("path") == "0":
                    for tf in ds.get("coordinateTransformations", []):
                        if tf.get("type") == "scale":
                            sc = tf.get("scale", [])
                            if len(sc) >= 4:
                                # (T, Z, Y, X) -> take (Z, Y, X)
                                scale_zyx = (float(sc[-3]), float(sc[-2]), float(sc[-1]))
    except Exception:
        scale_zyx = default_scale_zyx

    return vol, scale_zyx


# ==============================================================================
# 3. Metadata Note & Cache Persistence Helpers
# ==============================================================================
def write_detector_metadata_note(
    output_dir: str,
    detector_name: str,
    detector_params: Dict[str, Any],
    dataset_records: List[Dict[str, Any]],
    sample_df: pd.DataFrame,
    total_wall_time_sec: float,
) -> str:
    """Write a human-readable note and JSON manifest describing cached spots_df outputs."""
    note_path = os.path.join(output_dir, "detector_metadata_note.txt")
    manifest_path = os.path.join(output_dir, "detector_cache_manifest.json")

    cols = list(sample_df.columns) if not sample_df.empty else []
    dtypes = {col: str(dtype) for col, dtype in sample_df.dtypes.items()} if not sample_df.empty else {}
    total_spots = sum(r["spots_detected"] for r in dataset_records)

    lines = [
        f"DETECTOR CACHE METADATA NOTE: {detector_name}",
        "=" * 72,
        f"Generated (UTC)      : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"Source Test Dir      : {TEST_DATA_DIR}",
        f"Sampling Config      : fraction={SAMPLE_FRACTION}, seed={RANDOM_SEED} ({len(dataset_records)} datasets)",
        f"Detector Parameters  : {json.dumps(detector_params)}",
        f"Total Spots Cached   : {total_spots}",
        f"Total Wall Time (s)  : {round(total_wall_time_sec, 3)}",
        "",
        "CACHED FILE FORMAT & TRACKER COMPATIBILITY",
        "-" * 72,
        "Each dataset's `spots_df` DataFrame (exact return object expected by",
        "`fiji_trackers.lap_division_tracker.track_cell_divisions(spots_df=...)`) is saved as:",
        "  1. `<dataset_name>_spots.parquet` (preserves exact pandas dtypes & index)",
        "  2. `<dataset_name>_spots.csv`     (human-readable backup; reload with pd.read_csv)",
        "",
        "DATAFRAME SCHEMA (spots_df)",
        "-" * 72,
        f"Columns ({len(cols)}): {cols}",
        "Column Dtypes & Units:",
    ]

    unit_hints = {
        "spot_id": "int64   — Unique spot identifier within dataset",
        "frame": "int64   — 0-indexed time frame (t)",
        "z": "float64 — Physical Z coordinate in micrometers (um)",
        "y": "float64 — Physical Y coordinate in micrometers (um)",
        "x": "float64 — Physical X coordinate in micrometers (um)",
        "z_px": "float64 — Voxel Z index (z / dz)",
        "y_px": "float64 — Voxel Y index (y / dy)",
        "x_px": "float64 — Voxel X index (x / dx)",
        "quality": "float64 — TrackMate spot detection quality score",
        "radius": "float64 — Spot radius in physical units (um)",
    }

    for col in cols:
        hint = unit_hints.get(col, f"{dtypes[col]} — Detector output feature")
        lines.append(f"  - {col:<12}: {hint} (actual dtype: {dtypes[col]})")

    lines.extend([
        "",
        "PER-DATASET SUMMARY",
        "-" * 72,
    ])
    for rec in dataset_records:
        lines.append(
            f"  * {rec['dataset']:<35} | shape={rec['shape']} | "
            f"scale_zyx={rec['scale_zyx']} | spots={rec['spots_detected']:>6} | "
            f"detect_time={rec['detector_time_sec']:.2f}s"
        )

    with open(note_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    manifest = {
        "detector_name": detector_name,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "test_data_dir": TEST_DATA_DIR,
        "sample_fraction": SAMPLE_FRACTION,
        "random_seed": RANDOM_SEED,
        "detector_params": detector_params,
        "columns": cols,
        "dtypes": dtypes,
        "total_spots": total_spots,
        "total_wall_time_sec": round(total_wall_time_sec, 3),
        "datasets": dataset_records,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return note_path


def run_and_cache_detector(
    detector_name: str,
    detector_fn: Callable[..., pd.DataFrame],
    detector_params: Dict[str, Any],
    sampled_zarrs: List[str],
    output_dir: str,
    ij: Any,
) -> None:
    """Run an object detector across sampled Zarrs and cache tracker-ready DataFrames."""
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"CACHING DETECTOR: {detector_name.upper()} -> {output_dir}")
    print("=" * 80)

    dataset_records: List[Dict[str, Any]] = []
    sample_nonempty_df = pd.DataFrame()
    t0 = time.perf_counter()

    for idx, zarr_path in enumerate(sampled_zarrs, start=1):
        dataset_name = os.path.splitext(os.path.basename(zarr_path))[0]

        # 1. Load volume & voxel scale
        t_load = time.perf_counter()
        vol, scale_zyx = load_zarr_volume_and_scale(zarr_path)
        load_sec = time.perf_counter() - t_load

        # 2. Run detector -> produces spots_df expected by track_cell_divisions
        t_det = time.perf_counter()
        spots_df = detector_fn(vol, scale_zyx, ij)
        det_sec = time.perf_counter() - t_det

        if not spots_df.empty and sample_nonempty_df.empty:
            sample_nonempty_df = spots_df.head(5).copy()

        # 3. Cache spots_df in Parquet (exact dtypes) and CSV (universal)
        parquet_path = os.path.join(output_dir, f"{dataset_name}_spots.parquet")
        csv_path = os.path.join(output_dir, f"{dataset_name}_spots.csv")

        spots_df.to_parquet(parquet_path, index=False)
        spots_df.to_csv(csv_path, index=False)

        n_spots = len(spots_df)
        dataset_records.append(
            {
                "dataset": dataset_name,
                "zarr_path": zarr_path,
                "shape": list(vol.shape),
                "scale_zyx": list(scale_zyx),
                "spots_detected": n_spots,
                "load_time_sec": round(load_sec, 4),
                "detector_time_sec": round(det_sec, 4),
                "parquet_file": os.path.basename(parquet_path),
                "csv_file": os.path.basename(csv_path),
            }
        )

        print(
            f"[{idx:>2}/{len(sampled_zarrs)}] {dataset_name} | "
            f"Spots: {n_spots:>6} | Load: {load_sec:>5.2f}s | Detect: {det_sec:>6.2f}s"
        )

    wall_time = time.perf_counter() - t0

    # 4. Write metadata note & manifest into the detector's output directory
    note_path = write_detector_metadata_note(
        output_dir=output_dir,
        detector_name=detector_name,
        detector_params=detector_params,
        dataset_records=dataset_records,
        sample_df=sample_nonempty_df,
        total_wall_time_sec=wall_time,
    )

    print("-" * 80)
    print(f"Finished {detector_name} in {wall_time:.2f}s | Metadata note -> {note_path}")


# ==============================================================================
# 4. Main Entry Point
# ==============================================================================
def main() -> None:
    sampled_zarrs = sample_zarr_datasets(
        TEST_DATA_DIR,
        fraction=SAMPLE_FRACTION,
        seed=RANDOM_SEED,
    )
    print(
        f"Sampled {len(sampled_zarrs)} datasets ({SAMPLE_FRACTION * 100:.0f}%, seed={RANDOM_SEED}) "
        f"from {TEST_DATA_DIR}"
    )

    # Must be called BEFORE imagej.init() so scyjava uses the cached Java 21 JDK
    t_jvm0 = time.perf_counter()
    scyjava.config.set_java_constraints(fetch=True, vendor="zulu", version="21")
    scyjava.config.add_option("-Djava.awt.headless=true")

    ij = imagej.init(FIJI_PATH, mode="headless")
    TrackMate = scyjava.jimport("fiji.plugin.trackmate.TrackMate")
    print(
        f"Initialized headless Fiji JVM ({ij.getVersion()}) & loaded {TrackMate} "
        f"in {time.perf_counter() - t_jvm0:.2f}s"
    )

    common_params = {
        "radius": DETECTOR_RADIUS,
        "threshold": DETECTOR_THRESHOLD,
    }

    detectors = [
        # (
        #     "log_detector",
        #     lambda vol, scale, ij_gw: detect_spots_log(
        #         vol,
        #         ij=ij_gw,
        #         scale_zyx=scale,
        #         radius=DETECTOR_RADIUS,
        #         threshold=DETECTOR_THRESHOLD,
        #     ),
        #     common_params,
        #     os.path.join(BASE_OUTPUT_DIR, "cached_spots_log_detector"),
        # ),
        (
            "dog_detector",
            lambda vol, scale, ij_gw: detect_spots_dog(
                vol,
                ij=ij_gw,
                scale_zyx=scale,
                radius=DETECTOR_RADIUS,
                threshold=DETECTOR_THRESHOLD,
            ),
            common_params,
            os.path.join(BASE_OUTPUT_DIR, "cached_spots_dog_detector"),
        ),
    ]

    for det_name, det_fn, det_params, out_dir in detectors:
        run_and_cache_detector(
            detector_name=det_name,
            detector_fn=det_fn,
            detector_params=det_params,
            sampled_zarrs=sampled_zarrs,
            output_dir=out_dir,
            ij=ij,
        )


if __name__ == "__main__":
    main()