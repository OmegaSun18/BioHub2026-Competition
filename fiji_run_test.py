import glob
import os
import sys
import time
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

from fiji_detectors.log_detector import detect_spots_log
from fiji_detectors.dog_detector import detect_spots_dog
from fiji_detectors.stardist_3d_detector import detect_spots_stardist_3d
from fiji_trackers.lap_division_tracker import track_cell_divisions


# ==============================================================================
# 1. Configuration
# ==============================================================================
TEST_DATA_DIR = "/kaggle/working/split_dataset/test"
BASE_OUTPUT_DIR = "/kaggle/working"
FIJI_PATH = "/opt/Fiji.app"

RANDOM_SEED = 8001
SAMPLE_FRACTION = 0.20
DEFAULT_SCALE_ZYX = (1.625, 0.40625, 0.40625)  # Fallback voxel spacing (dz, dy, dx) in um

# Detector & Tracker Hyperparameters
DETECTOR_RADIUS = 2.5
DETECTOR_THRESHOLD = 5.0
STARDIST_MODEL_NAME = "3D_demo"

MAX_DIVISION_STEPS = 5          # Track parent & two daughters for up to t=5 steps
LINKING_MAX_DISTANCE = 15.0     # Physical units (um)
GAP_CLOSING_MAX_DISTANCE = 15.0
MAX_FRAME_GAP = 2
SPLITTING_MAX_DISTANCE = 15.0

OUTPUT_COLUMNS = [
    "id",
    "dataset",
    "row_type",
    "node_id",
    "t",
    "z",
    "y",
    "x",
    "source_id",
    "target_id",
]


# ==============================================================================
# 2. Dataset Sampling & Zarr Helpers
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
# 3. Convert Division Trajectories to Required Node/Edge CSV Schema
# ==============================================================================
def format_divisions_to_graph_rows(
    dataset_name: str,
    trajectories_df: pd.DataFrame,
    scale_zyx: Tuple[float, float, float],
    vol_shape: Tuple[int, ...],
    start_global_id: int,
) -> Tuple[List[Dict[str, Any]], int, int, int]:
    """
    Convert isolated cell-division trajectories into standardized node & edge rows:
        id,dataset,row_type,node_id,t,z,y,x,source_id,target_id
    """
    if trajectories_df.empty:
        return [], start_global_id, 0, 0

    dz, dy, dx = scale_zyx
    max_t = vol_shape[0] - 1
    max_z = vol_shape[-3] - 1
    max_y = vol_shape[-2] - 1
    max_x = vol_shape[-1] - 1

    # 1. Deduplicate unique spots participating in isolated division trajectories
    unique_spots = (
        trajectories_df.sort_values(["frame", "spot_id"])
        .drop_duplicates(subset=["spot_id"], keep="first")
        .reset_index(drop=True)
    )

    spot_to_node_id: Dict[int, int] = {}
    rows: List[Dict[str, Any]] = []
    global_id = start_global_id

    for local_node_id, row in enumerate(unique_spots.itertuples(index=False), start=1):
        sid = int(row.spot_id)
        spot_to_node_id[sid] = local_node_id

        z_vox = row.z_px if pd.notna(row.z_px) else (row.z / dz)
        y_vox = row.y_px if pd.notna(row.y_px) else (row.y / dy)
        x_vox = row.x_px if pd.notna(row.x_px) else (row.x / dx)

        t_int = int(np.clip(round(float(row.frame)), 0, max_t))
        z_int = int(np.clip(round(float(z_vox)), 0, max_z))
        y_int = int(np.clip(round(float(y_vox)), 0, max_y))
        x_int = int(np.clip(round(float(x_vox)), 0, max_x))

        rows.append(
            {
                "id": global_id,
                "dataset": dataset_name,
                "row_type": "node",
                "node_id": local_node_id,
                "t": t_int,
                "z": z_int,
                "y": y_int,
                "x": x_int,
                "source_id": -1,
                "target_id": -1,
            }
        )
        global_id += 1

    n_nodes = len(rows)

    # 2. Build directed lineage edges within each division event
    edge_pairs: List[Tuple[int, int]] = []
    seen_edges = set()

    for _, div_grp in trajectories_df.groupby("division_id", sort=True):
        parent_grp = div_grp[div_grp["role"] == "parent"].sort_values("step_offset")
        parent_spots = [int(s) for s in parent_grp["spot_id"].tolist()]

        # Sequential parent edges leading up to division (step_offset: -k -> ... -> 0)
        for u_spot, v_spot in zip(parent_spots[:-1], parent_spots[1:]):
            pair = (spot_to_node_id[u_spot], spot_to_node_id[v_spot])
            if pair[0] != pair[1] and pair not in seen_edges:
                seen_edges.add(pair)
                edge_pairs.append(pair)

        div_parent_spot = parent_spots[-1] if parent_spots else None

        # Daughter 1 & Daughter 2 edges (parent at step 0 -> daughter step +1 -> ... -> +t)
        for daughter_role in ("daughter_1", "daughter_2"):
            d_grp = div_grp[div_grp["role"] == daughter_role].sort_values("step_offset")
            d_spots = [int(s) for s in d_grp["spot_id"].tolist()]
            if not d_spots:
                continue

            if div_parent_spot is not None:
                split_pair = (spot_to_node_id[div_parent_spot], spot_to_node_id[d_spots[0]])
                if split_pair[0] != split_pair[1] and split_pair not in seen_edges:
                    seen_edges.add(split_pair)
                    edge_pairs.append(split_pair)

            for u_spot, v_spot in zip(d_spots[:-1], d_spots[1:]):
                pair = (spot_to_node_id[u_spot], spot_to_node_id[v_spot])
                if pair[0] != pair[1] and pair not in seen_edges:
                    seen_edges.add(pair)
                    edge_pairs.append(pair)

    for src_nid, tgt_nid in edge_pairs:
        rows.append(
            {
                "id": global_id,
                "dataset": dataset_name,
                "row_type": "edge",
                "node_id": -1,
                "t": -1,
                "z": -1,
                "y": -1,
                "x": -1,
                "source_id": src_nid,
                "target_id": tgt_nid,
            }
        )
        global_id += 1

    return rows, global_id, n_nodes, len(edge_pairs)


# ==============================================================================
# 4. Sequential Pipeline Execution & Timing Profiler
# ==============================================================================
def run_single_pipeline(
    pipeline_name: str,
    detector_fn: Callable[..., pd.DataFrame],
    sampled_zarrs: List[str],
    output_dir: str,
    ij: Any,
) -> Dict[str, Any]:
    """Run detection -> division tracking -> CSV export sequentially for one pipeline."""
    os.makedirs(output_dir, exist_ok=True)
    tracks_csv_path = os.path.join(output_dir, "tracks.csv")
    profile_csv_path = os.path.join(output_dir, "profiling_per_dataset.csv")

    print("\n" + "=" * 80)
    print(f"STARTING {pipeline_name.upper()} -> Output: {output_dir}")
    print("=" * 80)

    all_graph_rows: List[Dict[str, Any]] = []
    profile_rows: List[Dict[str, Any]] = []
    global_row_id = 0

    pipeline_t0 = time.perf_counter()

    for idx, zarr_path in enumerate(sampled_zarrs, start=1):
        dataset_name = os.path.splitext(os.path.basename(zarr_path))[0]

        # 1. Load raw 4D Zarr volume
        t_load_start = time.perf_counter()
        vol, scale_zyx = load_zarr_volume_and_scale(zarr_path)
        load_time = time.perf_counter() - t_load_start

        # 2. Run Object Detector
        t_det_start = time.perf_counter()
        spots_df = detector_fn(vol, scale_zyx, ij)
        detect_time = time.perf_counter() - t_det_start

        # 3. Run Cell-Division Tracker
        t_trk_start = time.perf_counter()
        divisions = track_cell_divisions(
            spots_df=spots_df,
            max_steps=MAX_DIVISION_STEPS,
            ij=ij,
            linking_max_distance=LINKING_MAX_DISTANCE,
            gap_closing_max_distance=GAP_CLOSING_MAX_DISTANCE,
            max_frame_gap=MAX_FRAME_GAP,
            splitting_max_distance=SPLITTING_MAX_DISTANCE,
        )
        track_time = time.perf_counter() - t_trk_start

        # 4. Format isolated divisions into required node/edge rows
        t_fmt_start = time.perf_counter()
        ds_rows, global_row_id, n_nodes, n_edges = format_divisions_to_graph_rows(
            dataset_name=dataset_name,
            trajectories_df=divisions["trajectories"],
            scale_zyx=scale_zyx,
            vol_shape=vol.shape,
            start_global_id=global_row_id,
        )
        all_graph_rows.extend(ds_rows)
        format_time = time.perf_counter() - t_fmt_start

        n_spots = len(spots_df)
        n_divisions = len(divisions["events"])

        profile_rows.append(
            {
                "pipeline": pipeline_name,
                "dataset": dataset_name,
                "shape": str(vol.shape),
                "raw_spots_detected": n_spots,
                "division_events_isolated": n_divisions,
                "exported_nodes": n_nodes,
                "exported_edges": n_edges,
                "load_time_sec": round(load_time, 4),
                "detector_time_sec": round(detect_time, 4),
                "tracker_time_sec": round(track_time, 4),
                "format_time_sec": round(format_time, 4),
                "total_dataset_time_sec": round(load_time + detect_time + track_time + format_time, 4),
            }
        )

        print(
            f"[{idx:>2}/{len(sampled_zarrs)}] {dataset_name} | "
            f"Spots: {n_spots:>5} | Divisions: {n_divisions:>3} | "
            f"Nodes: {n_nodes:>4}, Edges: {n_edges:>4} | "
            f"Detect: {detect_time:>6.2f}s | Track: {track_time:>6.2f}s"
        )

    # 5. Persist pipeline outputs
    out_df = (
        pd.DataFrame(all_graph_rows, columns=OUTPUT_COLUMNS)
        if all_graph_rows
        else pd.DataFrame(columns=OUTPUT_COLUMNS)
    )
    for col in ["id", "node_id", "t", "z", "y", "x", "source_id", "target_id"]:
        out_df[col] = out_df[col].astype(np.int64)
    out_df.to_csv(tracks_csv_path, index=False)

    prof_df = pd.DataFrame(profile_rows)
    prof_df.to_csv(profile_csv_path, index=False)

    pipeline_total_time = time.perf_counter() - pipeline_t0
    summary = {
        "pipeline": pipeline_name,
        "output_dir": output_dir,
        "datasets_processed": len(sampled_zarrs),
        "total_raw_spots": int(prof_df["raw_spots_detected"].sum()),
        "total_division_events": int(prof_df["division_events_isolated"].sum()),
        "total_exported_nodes": int(prof_df["exported_nodes"].sum()),
        "total_exported_edges": int(prof_df["exported_edges"].sum()),
        "total_detector_time_sec": round(float(prof_df["detector_time_sec"].sum()), 3),
        "mean_detector_time_sec": round(float(prof_df["detector_time_sec"].mean()), 3),
        "total_tracker_time_sec": round(float(prof_df["tracker_time_sec"].sum()), 3),
        "mean_tracker_time_sec": round(float(prof_df["tracker_time_sec"].mean()), 3),
        "total_pipeline_wall_time_sec": round(pipeline_total_time, 3),
    }

    print("-" * 80)
    print(
        f"COMPLETED {pipeline_name}: "
        f"Total Detect = {summary['total_detector_time_sec']:.2f}s "
        f"(avg {summary['mean_detector_time_sec']:.2f}s/ds) | "
        f"Total Track = {summary['total_tracker_time_sec']:.2f}s "
        f"(avg {summary['mean_tracker_time_sec']:.2f}s/ds) | "
        f"Total Wall = {summary['total_pipeline_wall_time_sec']:.2f}s"
    )
    print(f"Saved tracks -> {tracks_csv_path}")
    return summary


# ==============================================================================
# 5. Main Entry Point
# ==============================================================================
def main() -> None:
    sampled_zarrs = sample_zarr_datasets(
        TEST_DATA_DIR,
        fraction=SAMPLE_FRACTION,
        seed=RANDOM_SEED,
    )
    print(
        f"Sampled {len(sampled_zarrs)} datasets ({SAMPLE_FRACTION * 100:.0f}%, seed={RANDOM_SEED}) "
        f"from {TEST_DATA_DIR}:"
    )
    print("  First 5:", [os.path.basename(p) for p in sampled_zarrs[:5]])

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

    # Define the 3 detector + tracker pipeline combinations
    pipelines = [
        (
            "pipeline1_log_lap",
            lambda vol, scale, ij_gw: detect_spots_log(
                vol,
                ij=ij_gw,
                scale_zyx=scale,
                radius=DETECTOR_RADIUS,
                threshold=DETECTOR_THRESHOLD,
            ),
            os.path.join(BASE_OUTPUT_DIR, "output_pipeline1"),
        ),
        (
            "pipeline2_dog_lap",
            lambda vol, scale, ij_gw: detect_spots_dog(
                vol,
                ij=ij_gw,
                scale_zyx=scale,
                radius=DETECTOR_RADIUS,
                threshold=DETECTOR_THRESHOLD,
            ),
            os.path.join(BASE_OUTPUT_DIR, "output_pipeline2"),
        ),
        (
            "pipeline3_stardist3d_lap",
            lambda vol, scale, ij_gw: detect_spots_stardist_3d(
                vol,
                scale_zyx=scale,
                model_name=STARDIST_MODEL_NAME,
            ),
            os.path.join(BASE_OUTPUT_DIR, "output_pipeline3"),
        ),
    ]

    all_summaries = []
    for pipeline_name, detector_fn, out_dir in pipelines:
        summary = run_single_pipeline(
            pipeline_name=pipeline_name,
            detector_fn=detector_fn,
            sampled_zarrs=sampled_zarrs,
            output_dir=out_dir,
            ij=ij,
        )
        all_summaries.append(summary)

    summary_df = pd.DataFrame(all_summaries)
    summary_path = os.path.join(BASE_OUTPUT_DIR, "pipeline_profiling_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 80)
    print("OVERALL PROFILING COMPARISON ACROSS ALL 3 PIPELINES")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print(f"\nSaved overarching profiling table -> {summary_path}")


if __name__ == "__main__":
    main()