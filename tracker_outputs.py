import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import imagej
import numpy as np
import pandas as pd
import scyjava

# Ensure the repository root (containing `fiji_trackers`) is on sys.path
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from fiji_trackers.lap_division_tracker import track_cell_divisions


# ==============================================================================
# 1. Configuration
# ==============================================================================
FIJI_PATH = os.environ.get("FIJI_PATH", "/opt/Fiji.app")

# Mounted via: -v /mnt/data:/mnt/data
CACHE_ROOT = os.environ.get("CACHE_ROOT", "/mnt/data/cache")
OUTPUT_BASE_DIR = os.environ.get("OUTPUT_BASE_DIR", "/mnt/data/fiji_tracks")

# Explicit ordered list of the 20 detector folders in /mnt/data/cache
EXPECTED_MODELS = [
    "dog_detector",
    "log_detector",
    "raw_only+dog+edge+erode@small",
    "raw_only+dog+edge+erode@medium",
    "raw_only+dog+edge+erode@medium-4000",
    "raw_only+dog+edge+erode@large",
    "raw_only+dog+edge+erode@large-6000",
    "temporal_diff+dog@small",
    "temporal_diff+dog@medium",
    "temporal_diff+dog@medium-4000",
    "temporal_diff+dog@large",
    "temporal_diff+dog@large-6000",
    "temporal_diff+erode@small",
    "temporal_diff+erode@medium",
    "temporal_diff+erode@medium-4000",
    "temporal_stack+erode@small",
    "temporal_stack+erode@medium",
    "temporal_stack+erode@medium-4000",
    "temporal_stack+erode@large",
    "temporal_stack+erode@large-6000",
]

# Default volume geometry & scale from DETECTIONS_AND_FEATURES_FORMAT.md
DEFAULT_SCALE_ZYX = (1.625, 0.40625, 0.40625)  # (dz, dy, dx) in um
DEFAULT_VOL_SHAPE = (50, 64, 256, 256)         # (T, Z, Y, X)
DEFAULT_RADIUS_UM = 2.5

# Fiji LAP Division Tracker Hyperparameters
MAX_DIVISION_STEPS = 5
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
# 2. Score Thresholds & Repeated-Frame Loading (DETECTIONS_AND_FEATURES_FORMAT.md)
# ==============================================================================
def get_score_threshold(model_name: str) -> float:
    """Return the detector score cut-off specified in Section 7."""
    name_lower = model_name.lower()
    if name_lower in ("dog_detector", "log_detector", "dog", "log"):
        return -np.inf  # Detector's own threshold is already applied
    if "temporal_diff+dog" in name_lower or "temporal_diff+erode" in name_lower:
        return 0.05
    return 0.1


def load_cached_detections(
    det_path: str,
    thr: float = 0.0,
    radius: float = DEFAULT_RADIUS_UM,
    drop_dup_frames: bool = True,
) -> Tuple[pd.DataFrame, int]:
    """
    Load `<video>.parquet` detections, apply the repeated-frames (`dup_frame`) fix,
    and convert into the `spots_df` schema expected by `track_cell_divisions`.
    """
    d = pd.read_parquet(det_path)

    # Support legacy Fiji spots parquet if encountered directly
    if "t" not in d.columns and "frame" in d.columns:
        d = d.rename(
            columns={
                "frame": "t",
                "z": "z_um",
                "y": "y_um",
                "x": "x_um",
                "z_px": "z",
                "y_px": "y",
                "x_px": "x",
                "quality": "score",
            }
        )

    # Repeated frames fix: drop rows where `dup_frame` is True while preserving `t`
    n_dup_dropped = 0
    if drop_dup_frames and "dup_frame" in d.columns:
        dup_mask = d["dup_frame"].fillna(False).astype(bool)
        n_dup_dropped = int(dup_mask.sum())
        d = d[~dup_mask].reset_index(drop=True)

    # Apply detector score threshold
    if "score" in d.columns and np.isfinite(thr):
        d = d[d["score"] >= thr].reset_index(drop=True)
    else:
        d = d.reset_index(drop=True)

    if d.empty:
        return pd.DataFrame(
            columns=["spot_id", "frame", "z", "y", "x", "z_px", "y_px", "x_px", "radius", "quality"]
        ), n_dup_dropped

    spot_radius = d["radius"].astype(float) if "radius" in d.columns else float(radius)
    spot_quality = d["score"].astype(float) if "score" in d.columns else 1.0

    spots_df = pd.DataFrame(
        {
            "spot_id": np.arange(len(d), dtype=np.int64),
            "frame": d["t"].astype(np.int64),
            "z": d["z_um"].astype(float),
            "y": d["y_um"].astype(float),
            "x": d["x_um"].astype(float),
            "z_px": d["z"].astype(float),
            "y_px": d["y"].astype(float),
            "x_px": d["x"].astype(float),
            "radius": spot_radius,
            "quality": spot_quality,
        }
    )
    return spots_df, n_dup_dropped


# ==============================================================================
# 3. Convert Division Trajectories to Required Node/Edge CSV Schema
# ==============================================================================
def format_divisions_to_graph_rows(
    dataset_name: str,
    trajectories_df: pd.DataFrame,
    scale_zyx: Tuple[float, float, float] = DEFAULT_SCALE_ZYX,
    vol_shape: Tuple[int, ...] = DEFAULT_VOL_SHAPE,
    start_global_id: int = 0,
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

    edge_pairs: List[Tuple[int, int]] = []
    seen_edges = set()

    for _, div_grp in trajectories_df.groupby("division_id", sort=True):
        parent_grp = div_grp[div_grp["role"] == "parent"].sort_values("step_offset")
        parent_spots = [int(s) for s in parent_grp["spot_id"].tolist()]

        for u_spot, v_spot in zip(parent_spots[:-1], parent_spots[1:]):
            pair = (spot_to_node_id[u_spot], spot_to_node_id[v_spot])
            if pair[0] != pair[1] and pair not in seen_edges:
                seen_edges.add(pair)
                edge_pairs.append(pair)

        div_parent_spot = parent_spots[-1] if parent_spots else None

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
# 4. Resolve `/mnt/data/cache/<model>` Detection Directories & Run Tracker
# ==============================================================================
def resolve_cache_det_dirs(cache_root: str) -> List[Tuple[str, Path]]:
    """
    Resolve the `detections/` directory inside each `/mnt/data/cache/<model>/` folder.
    Supports both:
      - /mnt/data/cache/<model>/detections/*.parquet
      - /mnt/data/cache/<model>/test_0.2/detections/*.parquet
      - /mnt/data/cache/<model>/*.parquet
    """
    root_path = Path(cache_root)
    if not root_path.exists():
        raise FileNotFoundError(f"Cache root directory not found: {cache_root}")

    # Include all 20 expected models plus any extra directories present in /mnt/data/cache
    present_subdirs = {p.name: p for p in root_path.iterdir() if p.is_dir()}
    ordered_names = list(EXPECTED_MODELS)
    for extra in sorted(present_subdirs.keys()):
        if extra not in ordered_names:
            ordered_names.append(extra)

    resolved: List[Tuple[str, Path]] = []
    for model_name in ordered_names:
        model_dir = root_path / model_name
        if not model_dir.is_dir():
            print(f"[WARNING] Expected model folder missing: {model_dir}")
            continue

        candidates = [
            model_dir / "detections",
            model_dir / "test_0.2" / "detections",
            model_dir,
        ]
        found_dir = None
        for cand in candidates:
            if cand.is_dir() and any(cand.glob("*.parquet")):
                found_dir = cand
                break

        if found_dir is None:
            # Fallback recursive search for a `detections` folder inside model_dir
            nested = [d for d in sorted(model_dir.rglob("detections")) if d.is_dir() and any(d.glob("*.parquet"))]
            if nested:
                found_dir = nested[0]

        if found_dir is None:
            print(f"[WARNING] No detection .parquet files found under: {model_dir}")
            continue

        resolved.append((model_name, found_dir))

    return resolved


def run_tracker_for_detector(
    model_name: str,
    det_dir: Path,
    output_dir: Path,
    ij: Any,
) -> Dict[str, Any]:
    """Run Fiji LAP division tracker on every detection parquet file in `det_dir`."""
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_files = sorted(det_dir.glob("*.parquet"))
    score_thr = get_score_threshold(model_name)

    print("\n" + "=" * 80)
    print(f"TRACKING DETECTOR: {model_name} (score_thr={score_thr})")
    print(f"  Detections Dir : {det_dir} ({len(parquet_files)} files)")
    print(f"  Output Dir     : {output_dir}")
    print("=" * 80)

    log_records: List[Dict[str, Any]] = []
    detector_t0 = time.perf_counter()

    for idx, pq_path in enumerate(parquet_files, start=1):
        raw_stem = pq_path.stem
        dataset_name = raw_stem[:-6] if raw_stem.endswith("_spots") else raw_stem
        tracks_csv_path = output_dir / f"{dataset_name}_tracks.csv"

        # 1. Load cached detections + repeated-frames fix
        t_load0 = time.perf_counter()
        spots_df, n_dup_dropped = load_cached_detections(
            det_path=str(pq_path),
            thr=score_thr,
            radius=DEFAULT_RADIUS_UM,
            drop_dup_frames=True,
        )
        load_sec = time.perf_counter() - t_load0

        # 2. Run Fiji division tracker
        t_trk0 = time.perf_counter()
        if spots_df.empty:
            trajectories_df = pd.DataFrame()
            n_divisions = 0
        else:
            divisions = track_cell_divisions(
                spots_df=spots_df,
                max_steps=MAX_DIVISION_STEPS,
                ij=ij,
                linking_max_distance=LINKING_MAX_DISTANCE,
                gap_closing_max_distance=GAP_CLOSING_MAX_DISTANCE,
                max_frame_gap=MAX_FRAME_GAP,
                splitting_max_distance=SPLITTING_MAX_DISTANCE,
            )
            trajectories_df = divisions["trajectories"]
            n_divisions = len(divisions["events"])
        track_sec = time.perf_counter() - t_trk0

        # 3. Format into required node/edge CSV schema
        ds_rows, _, n_nodes, n_edges = format_divisions_to_graph_rows(
            dataset_name=dataset_name,
            trajectories_df=trajectories_df,
            scale_zyx=DEFAULT_SCALE_ZYX,
            vol_shape=DEFAULT_VOL_SHAPE,
            start_global_id=0,
        )

        out_df = (
            pd.DataFrame(ds_rows, columns=OUTPUT_COLUMNS)
            if ds_rows
            else pd.DataFrame(columns=OUTPUT_COLUMNS)
        )
        for col in ["id", "node_id", "t", "z", "y", "x", "source_id", "target_id"]:
            out_df[col] = out_df[col].astype(np.int64)
        out_df.to_csv(tracks_csv_path, index=False)

        log_records.append(
            {
                "detector": model_name,
                "dataset": dataset_name,
                "tracks_csv": tracks_csv_path.name,
                "spots_kept": len(spots_df),
                "dup_frame_rows_dropped": n_dup_dropped,
                "divisions_isolated": n_divisions,
                "exported_nodes": n_nodes,
                "exported_edges": n_edges,
                "load_time_sec": round(load_sec, 4),
                "tracker_time_sec": round(track_sec, 4),
            }
        )

        print(
            f"[{idx:>2}/{len(parquet_files)}] {dataset_name:<18} -> {tracks_csv_path.name:<29} | "
            f"Spots: {len(spots_df):>5} (dup dropped: {n_dup_dropped:>3}) | "
            f"Nodes: {n_nodes:>4}, Edges: {n_edges:>4} | Track: {track_sec:>6.3f}s"
        )

    total_wall_sec = time.perf_counter() - detector_t0
    log_df = pd.DataFrame(log_records)
    total_tracker_sec = float(log_df["tracker_time_sec"].sum()) if not log_df.empty else 0.0
    mean_tracker_sec = float(log_df["tracker_time_sec"].mean()) if not log_df.empty else 0.0

    # 4. Write timing logs inside the same output directory
    log_csv_path = output_dir / "tracker_timing_log.csv"
    log_txt_path = output_dir / "tracker_timing_log.txt"
    log_df.to_csv(log_csv_path, index=False)

    log_lines = [
        f"FIJI TRACKER TIMING LOG — DETECTOR: {model_name}",
        "=" * 80,
        f"Timestamp (UTC)          : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"Source Detections Dir    : {det_dir}",
        f"Output Directory         : {output_dir}",
        f"Score Threshold Applied  : {score_thr}",
        f"Repeated Frames Fix      : drop rows with dup_frame == True (preserve frame `t`)",
        f"Tracker Params           : max_steps={MAX_DIVISION_STEPS}, link_dist={LINKING_MAX_DISTANCE}, "
        f"gap_dist={GAP_CLOSING_MAX_DISTANCE}, max_gap={MAX_FRAME_GAP}, split_dist={SPLITTING_MAX_DISTANCE}",
        f"Files Processed          : {len(log_records)}",
        f"Total Tracker Time (s)   : {total_tracker_sec:.4f}",
        f"Mean Tracker Time/File(s): {mean_tracker_sec:.4f}",
        f"Total Wall Time (s)      : {total_wall_sec:.4f}",
        "",
        "PER-FILE TRACKER TIMINGS",
        "-" * 80,
    ]
    for rec in log_records:
        log_lines.append(
            f"{rec['tracks_csv']:<32} | spots={rec['spots_kept']:>5} | "
            f"dup_dropped={rec['dup_frame_rows_dropped']:>4} | "
            f"nodes={rec['exported_nodes']:>4} | edges={rec['exported_edges']:>4} | "
            f"tracker_time={rec['tracker_time_sec']:.4f}s"
        )

    with open(log_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")

    print("-" * 80)
    print(
        f"COMPLETED {model_name}: Total Tracker Time = {total_tracker_sec:.2f}s "
        f"(avg {mean_tracker_sec:.3f}s/file) | Log -> {log_txt_path}"
    )

    return {
        "detector": model_name,
        "output_dir": str(output_dir),
        "files_processed": len(log_records),
        "total_tracker_time_sec": round(total_tracker_sec, 4),
        "mean_tracker_time_sec": round(mean_tracker_sec, 4),
        "total_wall_time_sec": round(total_wall_sec, 4),
    }


# ==============================================================================
# 5. Main Entry Point
# ==============================================================================
def main() -> None:
    detection_dirs = resolve_cache_det_dirs(CACHE_ROOT)
    if not detection_dirs:
        raise FileNotFoundError(f"No detection parquet folders found under: {CACHE_ROOT}")

    print(f"Resolved {len(detection_dirs)} detector folders under {CACHE_ROOT}:")
    for model_name, det_dir in detection_dirs:
        print(f"  - {model_name:<35} -> {det_dir}")

    # Initialize headless Fiji JVM once before looping across all 20 detectors
    t_jvm0 = time.perf_counter()
    scyjava.config.set_java_constraints(fetch=True, vendor="zulu", version="21")
    scyjava.config.add_option("-Djava.awt.headless=true")

    ij = imagej.init(FIJI_PATH, mode="headless")
    TrackMate = scyjava.jimport("fiji.plugin.trackmate.TrackMate")
    print(
        f"\nInitialized headless Fiji JVM ({ij.getVersion()}) & loaded {TrackMate} "
        f"in {time.perf_counter() - t_jvm0:.2f}s"
    )

    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    overall_summaries: List[Dict[str, Any]] = []

    for model_name, det_dir in detection_dirs:
        model_out_dir = Path(OUTPUT_BASE_DIR) / model_name
        summary = run_tracker_for_detector(
            model_name=model_name,
            det_dir=det_dir,
            output_dir=model_out_dir,
            ij=ij,
        )
        overall_summaries.append(summary)

    master_summary_path = Path(OUTPUT_BASE_DIR) / "all_detectors_tracker_timing_summary.csv"
    pd.DataFrame(overall_summaries).to_csv(master_summary_path, index=False)
    print(f"\nSaved overall timing summary across all {len(overall_summaries)} detectors -> {master_summary_path}")


if __name__ == "__main__":
    main()