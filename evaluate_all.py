import os

# Pin internal BLAS/OpenMP thread pools to 1 thread per worker so 12 worker
# processes use ~12 cores cleanly rather than spawning 12 x 16 = 192 threads.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import concurrent.futures
import importlib.util
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import zarr


# ==============================================================================
# 1. Environment & Repository Setup (Adapted from ground_truth_evaluation.py)
# ==============================================================================
INTERNET_AVAILABLE = True
WORK_DIR = Path("/kaggle/working")
REPO_DIR = WORK_DIR / "kaggle-cell-tracking-competition"
REPO_SRC = REPO_DIR / "src"


def is_package_installed(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


packages_ready = is_package_installed("tracking_cellmot") and is_package_installed("zarr")
scripts_ready = (
    (REPO_DIR / "scripts" / "csv_to_geffs.py").exists()
    and (REPO_DIR / "scripts" / "evaluate.py").exists()
)

if packages_ready and scripts_ready:
    print("Libraries (tracking_cellmot, zarr) and evaluation scripts ready.")
else:
    if INTERNET_AVAILABLE:
        if not REPO_DIR.exists():
            subprocess.run(
                [
                    "git",
                    "clone",
                    "https://github.com/royerlab/kaggle-cell-tracking-competition.git",
                    str(REPO_DIR),
                ],
                check=True,
            )
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--break-system-packages", "-q", str(REPO_DIR), "zarr"],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "--force-reinstall",
                "--break-system-packages",
                "--no-deps",
                "numpy>=1.26.4,<2.1.0",
            ],
            check=True,
        )

if REPO_SRC.exists() and str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))


# ==============================================================================
# 2. Paths, Concurrency & Evaluation Constants
# ==============================================================================
GT_DIR = Path("/kaggle/working/split_dataset/test")
ZARR_DIR = Path("/kaggle/working/split_dataset/test")

FIJI_TRACKS_ROOT = Path("/mnt/data/fiji_tracks")
SUBMISSIONS_ROOT = Path("/mnt/data/submissions")

OUTPUT_MD_PATH = Path("/mnt/data/evaluation_results_summary.md")
OUTPUT_SUMMARY_CSV = Path("/mnt/data/evaluation_summary_table.csv")
OUTPUT_PER_VIDEO_CSV = Path("/mnt/data/evaluation_per_video_metrics.csv")

CSV_TO_GEFFS_SCRIPT = REPO_DIR / "scripts" / "csv_to_geffs.py"
EVALUATE_SCRIPT = REPO_DIR / "scripts" / "evaluate.py"

DEFAULT_SCALE_ZYX = (1.625, 0.40625, 0.40625)  # Fallback (dz, dy, dx) in um
MAX_PARENT_CHILD_DIST_UM = 14.0

# Fixed threshold t = 0.15 for Assertion 1 (node count tolerance)
FIXED_THRESHOLD_T = 0.15

# Use 12 worker processes comfortably out of 16 total CPU cores
MAX_WORKERS = min(12, max(1, os.cpu_count() or 12))

# --- COMMENTED OUT: Threshold Fine-Tuning Constants ---
# SCORE_CORRECT_THRESHOLD = 0.66
# THRESHOLD_GRID = [round(x, 2) for x in np.arange(0.05, 1.00, 0.05)]

REQUIRED_COLUMNS = [
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
# 3. Ground-Truth .geff & .zarr Metadata Loaders
# ==============================================================================
def read_estimated_nodes_from_geff(geff_path: Path) -> int:
    """Extract `estimated_number_of_nodes` from a ground-truth .geff Zarr group."""
    root = zarr.open(str(geff_path), mode="r")
    attrs = dict(root.attrs)

    if "estimated_number_of_nodes" in attrs:
        return int(attrs["estimated_number_of_nodes"])

    geff_meta = attrs.get("geff", {})
    if isinstance(geff_meta, dict) and "estimated_number_of_nodes" in geff_meta:
        return int(geff_meta["estimated_number_of_nodes"])

    if "nodes" in root:
        node_attrs = dict(root["nodes"].attrs)
        if "estimated_number_of_nodes" in node_attrs:
            return int(node_attrs["estimated_number_of_nodes"])
        if "ids" in root["nodes"]:
            return int(root["nodes"]["ids"].shape[0])

    raise KeyError(f"`estimated_number_of_nodes` not found in {geff_path}")


def read_scale_from_zarr(
    zarr_path: Path,
    default_scale_zyx: Tuple[float, float, float] = DEFAULT_SCALE_ZYX,
) -> Tuple[float, float, float]:
    """Read (dz, dy, dx) physical voxel spacing in um from an OME-Zarr volume."""
    if not zarr_path.exists():
        return default_scale_zyx
    try:
        root = zarr.open(str(zarr_path), mode="r")
        multiscales = dict(root.attrs).get("multiscales", [])
        if multiscales:
            for ds in multiscales[0].get("datasets", []):
                if ds.get("path") == "0":
                    for tf in ds.get("coordinateTransformations", []):
                        if tf.get("type") == "scale":
                            sc = tf.get("scale", [])
                            if len(sc) >= 4:
                                return (float(sc[-3]), float(sc[-2]), float(sc[-1]))
    except Exception:
        pass
    return default_scale_zyx


def load_ground_truth_metadata(
    gt_dir: Path, zarr_dir: Path
) -> Tuple[List[str], Dict[str, int], Dict[str, Tuple[float, float, float]]]:
    """Load dataset names, estimated node counts, and ZYX voxel scales for all GT videos."""
    assert gt_dir.exists(), f"Ground-truth directory not found: {gt_dir}"
    gt_geffs = sorted(gt_dir.glob("*.geff"))
    assert len(gt_geffs) > 0, f"No .geff files found in {gt_dir}"

    datasets: List[str] = []
    est_nodes_map: Dict[str, int] = {}
    scale_map: Dict[str, Tuple[float, float, float]] = {}

    for geff_path in gt_geffs:
        ds_name = geff_path.stem
        datasets.append(ds_name)
        est_nodes_map[ds_name] = read_estimated_nodes_from_geff(geff_path)
        scale_map[ds_name] = read_scale_from_zarr(zarr_dir / f"{ds_name}.zarr")

    return datasets, est_nodes_map, scale_map


# ==============================================================================
# 4. Variant Discovery & Unified CSV Loading
# ==============================================================================
def discover_variants(
    fiji_root: Path, submissions_root: Path
) -> List[Dict[str, Any]]:
    """
    Discover all (detector, tracker) variants across:
      1. /mnt/data/fiji_tracks/{detector}/*_tracks.csv (tracker='fiji_lap')
      2. /mnt/data/submissions/{detector}/{tracker}/submission.csv
    """
    variants: List[Dict[str, Any]] = []

    # 1. Fiji tracker outputs
    if fiji_root.exists():
        for det_dir in sorted(p for p in fiji_root.iterdir() if p.is_dir()):
            single_csv = det_dir / "submission.csv"
            if not single_csv.exists():
                single_csv = det_dir / "tracks.csv"
            per_video_csvs = sorted(det_dir.glob("*_tracks.csv"))

            if single_csv.is_file():
                variants.append(
                    {
                        "source_group": "fiji_tracks",
                        "detector": det_dir.name,
                        "tracker": "fiji_lap",
                        "csv_paths": [single_csv],
                    }
                )
            elif per_video_csvs:
                variants.append(
                    {
                        "source_group": "fiji_tracks",
                        "detector": det_dir.name,
                        "tracker": "fiji_lap",
                        "csv_paths": per_video_csvs,
                    }
                )

    # 2. Submissions outputs: /mnt/data/submissions/{detector}/{tracker}/submission.csv
    if submissions_root.exists():
        for sub_csv in sorted(submissions_root.glob("*/*/submission.csv")):
            tracker_name = sub_csv.parent.name
            detector_name = sub_csv.parent.parent.name
            variants.append(
                {
                    "source_group": "submissions",
                    "detector": detector_name,
                    "tracker": tracker_name,
                    "csv_paths": [sub_csv],
                }
            )

    return variants


def load_variant_dataframe(csv_paths: List[Path], tmp_combined_csv: Path) -> pd.DataFrame:
    """Load one or more CSV files for a variant, normalize schema, and write a combined CSV."""
    dfs = []
    for p in csv_paths:
        if p.stat().st_size == 0:
            continue
        df_part = pd.read_csv(p)
        if not df_part.empty:
            dfs.append(df_part)

    if not dfs:
        empty_df = pd.DataFrame(columns=REQUIRED_COLUMNS)
        empty_df.to_csv(tmp_combined_csv, index=False)
        return empty_df

    df = pd.concat(dfs, ignore_index=True)
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing required column `{col}` in {csv_paths[0]}")

    df = df[REQUIRED_COLUMNS].copy()
    df["id"] = np.arange(len(df), dtype=np.int64)
    df["dataset"] = df["dataset"].astype(str)
    df["row_type"] = df["row_type"].astype(str).str.strip()
    for col in ["node_id", "t", "z", "y", "x", "source_id", "target_id"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype(np.int64)

    df.to_csv(tmp_combined_csv, index=False)
    return df


# ==============================================================================
# 5. Official Score Evaluation
# ==============================================================================
def parse_evaluate_stdout(stdout_text: str) -> Dict[str, Any]:
    """Parse aggregate score and per-dataset scores (if printed) from evaluate.py output."""
    agg_score = np.nan
    edge_jaccard = np.nan
    div_jaccard = np.nan
    per_ds_scores: Dict[str, float] = {}

    for line in stdout_text.splitlines():
        s = line.strip()
        if not s:
            continue

        m_score = re.search(r"(?:^|\b)(?:combined\s+)?score\s*[:=]\s*([0-9]*\.?[0-9]+)", s, re.IGNORECASE)
        if m_score:
            agg_score = float(m_score.group(1))

        m_edge = re.search(r"edge\s+jaccard\s*[:=]\s*([0-9]*\.?[0-9]+)", s, re.IGNORECASE)
        if m_edge:
            edge_jaccard = float(m_edge.group(1))

        m_div = re.search(r"division\s+jaccard\s*[:=]\s*([0-9]*\.?[0-9]+)", s, re.IGNORECASE)
        if m_div:
            div_jaccard = float(m_div.group(1))

        m_ds = re.match(r"^([A-Za-z0-9_-]+)\s*[:|]\s*.*?(?:score)\s*[:=]\s*([0-9]*\.?[0-9]+)", s, re.IGNORECASE)
        if m_ds and m_ds.group(1).lower() not in {"overall", "total", "combined", "mean", "average", "final"}:
            per_ds_scores[m_ds.group(1)] = float(m_ds.group(2))

    return {
        "aggregate_score": agg_score,
        "edge_jaccard": edge_jaccard,
        "division_jaccard": div_jaccard,
        "per_ds_scores": per_ds_scores,
    }


def compute_variant_scores(
    combined_csv_path: Path,
    gt_dir: Path,
    variant_tmp_dir: Path,
) -> Tuple[float, float, float, Dict[str, float]]:
    """
    Convert combined CSV -> .geff files and run official evaluate.py for aggregate score.
    """
    pred_geff_dir = variant_tmp_dir / "pred_geffs"
    pred_geff_dir.mkdir(parents=True, exist_ok=True)

    # 1. Convert CSV -> .geff
    subprocess.run(
        [
            sys.executable,
            str(CSV_TO_GEFFS_SCRIPT),
            "--csv",
            str(combined_csv_path),
            "--out-dir",
            str(pred_geff_dir),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    # 2. Run official aggregate evaluation across all videos
    eval_proc = subprocess.run(
        [
            sys.executable,
            str(EVALUATE_SCRIPT),
            "--pred-dir",
            str(pred_geff_dir),
            "--gt-dir",
            str(gt_dir),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    parsed = parse_evaluate_stdout(eval_proc.stdout)
    return (
        parsed["aggregate_score"],
        parsed["edge_jaccard"],
        parsed["division_jaccard"],
        dict(parsed["per_ds_scores"]),
    )


# ==============================================================================
# 6. Evaluation of Assertions 1 through 6 on Each Video
# ==============================================================================
def evaluate_video_assertions(
    ds_df: pd.DataFrame,
    est_nodes: int,
    scale_zyx: Tuple[float, float, float],
    threshold_t: float = FIXED_THRESHOLD_T,
) -> Dict[str, Any]:
    """
    Evaluate Assertions 1-6 on a single video's rows from `tracks.csv` / `submission.csv`.
    """
    nodes_df = ds_df[ds_df["row_type"] == "node"].copy()
    edges_df = ds_df[ds_df["row_type"] == "edge"].copy()

    if not edges_df.empty:
        edges_df = edges_df.drop_duplicates(subset=["source_id", "target_id"]).reset_index(drop=True)

    # Assertion 1: Predicted node count within t% of estimated_number_of_nodes
    unique_node_ids = nodes_df["node_id"].unique() if not nodes_df.empty else np.array([], dtype=np.int64)
    n_pred_nodes = int(len(unique_node_ids))
    rel_node_err = (
        abs(n_pred_nodes - est_nodes) / float(est_nodes)
        if est_nodes > 0
        else (0.0 if n_pred_nodes == 0 else np.inf)
    )
    a1_pass = bool(rel_node_err <= threshold_t)

    # Build adjacency maps
    parents_of: Dict[int, List[int]] = defaultdict(list)
    children_of: Dict[int, List[int]] = defaultdict(list)
    for row in edges_df.itertuples(index=False):
        u = int(row.source_id)
        v = int(row.target_id)
        parents_of[v].append(u)
        children_of[u].append(v)

    # Assertion 2: No cell merges (in-degree <= 1 for every target node)
    a2_merge_violations = sum(1 for v, plist in parents_of.items() if len(set(plist)) > 1)
    a2_pass = a2_merge_violations == 0

    # Assertion 3: No duplicate node_ids unless consecutive in the same track
    a3_dup_violations = 0
    if not nodes_df.empty:
        node_groups = nodes_df.groupby("node_id", sort=False)
        for nid, grp in node_groups:
            if len(grp) <= 1:
                continue
            ts = sorted(grp["t"].tolist())
            expected_ts = list(range(ts[0], ts[0] + len(ts)))
            is_consecutive_frames = ts == expected_ts
            is_single_track = (
                len(set(parents_of.get(int(nid), []))) <= 1
                and len(set(children_of.get(int(nid), []))) <= 1
            )
            if not (is_consecutive_frames and is_single_track):
                a3_dup_violations += 1
    a3_pass = a3_dup_violations == 0

    node_min_t: Dict[int, int] = {}
    node_max_t: Dict[int, int] = {}
    node_coords: Dict[int, Tuple[float, float, float]] = {}
    if not nodes_df.empty:
        for nid, grp in nodes_df.groupby("node_id", sort=False):
            nid_int = int(nid)
            node_min_t[nid_int] = int(grp["t"].min())
            node_max_t[nid_int] = int(grp["t"].max())
            first_row = grp.sort_values("t").iloc[0]
            node_coords[nid_int] = (float(first_row["z"]), float(first_row["y"]), float(first_row["x"]))

    # Assertion 4: Binary division (2 children) & no immediate child re-split
    a4_non_binary_splits = 0
    a4_immediate_resplits = 0
    for parent_id, ch_list in children_of.items():
        unique_children = list(dict.fromkeys(ch_list))
        if len(unique_children) > 1:
            if len(unique_children) != 2:
                a4_non_binary_splits += 1
            for cid in unique_children:
                grand_children = set(children_of.get(cid, []))
                if len(grand_children) > 1:
                    a4_immediate_resplits += 1
    a4_violations = a4_non_binary_splits + a4_immediate_resplits
    a4_pass = a4_violations == 0

    # Assertion 5: Parent strictly before child (t_parent < t_child) & no reappearance
    a5_temporal_violations = 0
    for row in edges_df.itertuples(index=False):
        u = int(row.source_id)
        v = int(row.target_id)
        if u not in node_max_t or v not in node_min_t:
            a5_temporal_violations += 1
            continue
        if not (node_max_t[u] < node_min_t[v]):
            a5_temporal_violations += 1

    for parent_id, ch_list in children_of.items():
        if not ch_list or parent_id not in node_max_t:
            continue
        valid_child_ts = [node_min_t[c] for c in ch_list if c in node_min_t]
        if valid_child_ts and node_max_t[parent_id] >= min(valid_child_ts):
            a5_temporal_violations += 1

    a5_pass = a5_temporal_violations == 0

    # Assertion 6: Parent-to-child Euclidean distance <= 14.0 um
    dz, dy, dx = scale_zyx
    a6_distance_violations = 0
    max_observed_dist_um = 0.0

    for row in edges_df.itertuples(index=False):
        u = int(row.source_id)
        v = int(row.target_id)
        if u not in node_coords or v not in node_coords:
            a6_distance_violations += 1
            continue
        zu, yu, xu = node_coords[u]
        zv, yv, xv = node_coords[v]
        dist_um = float(
            np.sqrt(
                ((zv - zu) * dz) ** 2
                + ((yv - yu) * dy) ** 2
                + ((xv - xu) * dx) ** 2
            )
        )
        if dist_um > max_observed_dist_um:
            max_observed_dist_um = dist_um
        if dist_um > MAX_PARENT_CHILD_DIST_UM:
            a6_distance_violations += 1

    a6_pass = a6_distance_violations == 0

    all_assertions_pass = bool(
        a1_pass and a2_pass and a3_pass and a4_pass and a5_pass and a6_pass
    )

    return {
        "n_pred_nodes": n_pred_nodes,
        "n_est_nodes": est_nodes,
        "rel_node_error": float(rel_node_err),
        "n_edges": int(len(edges_df)),
        "threshold_t": float(threshold_t),
        "a1_pass": a1_pass,
        "a2_pass": bool(a2_pass),
        "a2_violations": int(a2_merge_violations),
        "a3_pass": bool(a3_pass),
        "a3_violations": int(a3_dup_violations),
        "a4_pass": bool(a4_pass),
        "a4_violations": int(a4_violations),
        "a5_pass": bool(a5_pass),
        "a5_violations": int(a5_temporal_violations),
        "a6_pass": bool(a6_pass),
        "a6_violations": int(a6_distance_violations),
        "max_parent_child_dist_um": round(max_observed_dist_um, 4),
        "all_assertions_pass": all_assertions_pass,
    }


# ==============================================================================
# 7. Parallel Worker Function (Evaluates One (detector, tracker) Variant)
# ==============================================================================
def evaluate_single_variant_worker(
    var: Dict[str, Any],
    gt_datasets: List[str],
    est_nodes_map: Dict[str, int],
    scale_map: Dict[str, Tuple[float, float, float]],
    tmp_root_str: str,
    chosen_t: float,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], float]:
    """Worker task executed in parallel across `MAX_WORKERS` processes."""
    v_t0 = time.perf_counter()
    source_group = var["source_group"]
    detector = var["detector"]
    tracker = var["tracker"]
    csv_paths = [Path(p) for p in var["csv_paths"]]

    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{source_group}__{detector}__{tracker}")
    variant_tmp_dir = Path(tmp_root_str) / safe_name
    variant_tmp_dir.mkdir(parents=True, exist_ok=True)
    combined_csv_path = variant_tmp_dir / "combined_tracks.csv"

    try:
        # 1. Load & normalize variant CSVs
        df = load_variant_dataframe(csv_paths, combined_csv_path)

        # 2. Compute official ground-truth accuracy (score)
        agg_score, edge_jaccard, div_jaccard, per_video_scores = compute_variant_scores(
            combined_csv_path=combined_csv_path,
            gt_dir=GT_DIR,
            variant_tmp_dir=variant_tmp_dir,
        )

        # 3. Evaluate Assertions 1-6 across all 40 videos using fixed t = 0.15
        grouped_by_ds = {ds: grp for ds, grp in df.groupby("dataset", sort=False)} if not df.empty else {}
        empty_ds_df = pd.DataFrame(columns=REQUIRED_COLUMNS)

        video_records: List[Dict[str, Any]] = []
        for ds_name in gt_datasets:
            ds_df = grouped_by_ds.get(ds_name, empty_ds_df)
            a_res = evaluate_video_assertions(
                ds_df=ds_df,
                est_nodes=est_nodes_map[ds_name],
                scale_zyx=scale_map[ds_name],
                threshold_t=chosen_t,
            )
            rec = {
                "source_group": source_group,
                "detector": detector,
                "tracker": tracker,
                "dataset": ds_name,
                "video_score": per_video_scores.get(ds_name, np.nan),
                **a_res,
            }
            video_records.append(rec)

        # 4. Aggregate variant-level metrics
        v_df = pd.DataFrame(video_records)
        summary_row = {
            "source_group": source_group,
            "detector": detector,
            "tracker": tracker,
            "n_videos": int(len(v_df)),
            "aggregate_score": round(float(agg_score), 4) if not np.isnan(agg_score) else np.nan,
            "edge_jaccard": round(float(edge_jaccard), 4) if not np.isnan(edge_jaccard) else np.nan,
            "division_jaccard": round(float(div_jaccard), 4) if not np.isnan(div_jaccard) else np.nan,
            "threshold_t": chosen_t,
            "overall_assertion_pass_rate": round(float(v_df["all_assertions_pass"].mean()), 4),
            "a1_pass_rate": round(float(v_df["a1_pass"].mean()), 4),
            "a2_pass_rate": round(float(v_df["a2_pass"].mean()), 4),
            "a3_pass_rate": round(float(v_df["a3_pass"].mean()), 4),
            "a4_pass_rate": round(float(v_df["a4_pass"].mean()), 4),
            "a5_pass_rate": round(float(v_df["a5_pass"].mean()), 4),
            "a6_pass_rate": round(float(v_df["a6_pass"].mean()), 4),
            "total_a2_merges": int(v_df["a2_violations"].sum()),
            "total_a3_dup_ids": int(v_df["a3_violations"].sum()),
            "total_a4_div_rules": int(v_df["a4_violations"].sum()),
            "total_a5_temporal": int(v_df["a5_violations"].sum()),
            "total_a6_dist_gt_14um": int(v_df["a6_violations"].sum()),
        }

        elapsed = time.perf_counter() - v_t0
        return summary_row, video_records, elapsed
    finally:
        shutil.rmtree(variant_tmp_dir, ignore_errors=True)


# ==============================================================================
# 8. Plot-Ready Markdown Report Generator
# ==============================================================================
def write_markdown_report(
    summary_df: pd.DataFrame,
    per_video_df: pd.DataFrame,
    md_path: Path,
    summary_csv_path: Path,
    per_video_csv_path: Path,
) -> None:
    """Write structured, clean GFM Markdown tables and companion CSVs for plotting."""
    summary_df.to_csv(summary_csv_path, index=False)
    per_video_df.to_csv(per_video_csv_path, index=False)

    def df_to_md_table(df: pd.DataFrame, cols: List[str]) -> str:
        sub = df[cols].copy()
        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |"
        rows = [header, sep]
        for row in sub.itertuples(index=False):
            formatted_cells = []
            for val in row:
                if isinstance(val, float):
                    formatted_cells.append("NaN" if np.isnan(val) else f"{val:.4f}")
                else:
                    formatted_cells.append(str(val))
            rows.append("| " + " | ".join(formatted_cells) + " |")
        return "\n".join(rows)

    core_cols = [
        "source_group",
        "detector",
        "tracker",
        "n_videos",
        "aggregate_score",
        "edge_jaccard",
        "division_jaccard",
        "threshold_t",
        "overall_assertion_pass_rate",
        "a1_pass_rate",
        "a2_pass_rate",
        "a3_pass_rate",
        "a4_pass_rate",
        "a5_pass_rate",
        "a6_pass_rate",
    ]

    violation_cols = [
        "source_group",
        "detector",
        "tracker",
        "total_a2_merges",
        "total_a3_dup_ids",
        "total_a4_div_rules",
        "total_a5_temporal",
        "total_a6_dist_gt_14um",
    ]

    fiji_df = summary_df[summary_df["source_group"] == "fiji_tracks"].sort_values(
        by=["aggregate_score", "overall_assertion_pass_rate"], ascending=False
    )
    sub_df = summary_df[summary_df["source_group"] == "submissions"].sort_values(
        by=["aggregate_score", "overall_assertion_pass_rate"], ascending=False
    )
    ranked_df = summary_df.sort_values(
        by=["aggregate_score", "overall_assertion_pass_rate"], ascending=False
    )

    lines = [
        "# Cell Tracking Evaluation & Assertion Benchmark Report",
        "",
        f"- **Generated (UTC)**: `{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}`",
        f"- **Ground Truth Directory**: `{GT_DIR}`",
        f"- **Total Variants Evaluated**: `{len(summary_df)}` (`{len(fiji_df)}` Fiji tracker runs + `{len(sub_df)}` submission tracker runs)",
        f"- **Node Count Tolerance Threshold (`t`)**: `{FIXED_THRESHOLD_T:.2f}` (fixed; threshold fine-tuning disabled)",
        f"- **Companion Plot-Ready CSVs**:",
        f"  - Aggregate Summary Table: `{summary_csv_path}`",
        f"  - Per-Video Assertion Metrics Table: `{per_video_csv_path}`",
        "",
        "## Assertion Definitions",
        "",
        f"- **A1 (Node Count Tolerance)**: `|N_pred - N_est| / N_est <= {FIXED_THRESHOLD_T:.2f}`, using `estimated_number_of_nodes` from each ground-truth `.geff`.",
        "- **A2 (No Cell Merges)**: Every node has in-degree `<= 1` (two distinct source node IDs cannot point to the same target node).",
        "- **A3 (No Duplicate Node IDs)**: No duplicate `node_id` entries unless consecutive in time (`t, t+1, ...`) within the same unbranched track.",
        "- **A4 (Division Branching & Cooldown)**: Every splitting parent (`out_degree > 1`) has `out_degree == 2` (two children), and neither child immediately splits (`out_degree <= 1`).",
        "- **A5 (Temporal Ordering)**: Every parent node is strictly earlier in time than its children (`t_parent < t_child`) and never reappears at or after a child's frame.",
        "- **A6 (Spatial Proximity <= 14 um)**: Every parent-child edge distance in micrometers (`voxel_delta * scale_zyx` from `.zarr`) is `<= 14.0 um`.",
        "",
        "## 1. Combined Leaderboard (All Detector + Tracker Variants)",
        "",
        df_to_md_table(ranked_df, core_cols),
        "",
    ]

    if not fiji_df.empty:
        lines.extend(
            [
                "## 2. Fiji LAP Division Tracker Results (`/mnt/data/fiji_tracks/`)",
                "",
                df_to_md_table(fiji_df, core_cols),
                "",
            ]
        )

    if not sub_df.empty:
        lines.extend(
            [
                "## 3. Submissions Results (`/mnt/data/submissions/{detector}/{tracker}/submission.csv`)",
                "",
                df_to_md_table(sub_df, core_cols),
                "",
            ]
        )

    lines.extend(
        [
            "## 4. Total Structural Violation Counts Across Test Videos",
            "",
            df_to_md_table(ranked_df, violation_cols),
            "",
        ]
    )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ==============================================================================
# 9. Main Execution Pipeline (Concurrent across MAX_WORKERS=12 processes)
# ==============================================================================
def main() -> None:
    t0 = time.perf_counter()
    gt_datasets, est_nodes_map, scale_map = load_ground_truth_metadata(GT_DIR, ZARR_DIR)
    print(f"Loaded ground-truth metadata for {len(gt_datasets)} test videos from {GT_DIR}")

    variants = discover_variants(FIJI_TRACKS_ROOT, SUBMISSIONS_ROOT)
    assert len(variants) > 0, (
        f"No tracker output files found in {FIJI_TRACKS_ROOT} or {SUBMISSIONS_ROOT}"
    )

    n_workers = min(MAX_WORKERS, len(variants))
    print(
        f"Discovered {len(variants)} (detector, tracker) variants. "
        f"Running concurrently with {n_workers} workers (fixed t = {FIXED_THRESHOLD_T:.2f})...\n"
    )

    # Convert Path objects in `variants` to strings for clean pickling across processes
    serializable_variants = [
        {
            "source_group": v["source_group"],
            "detector": v["detector"],
            "tracker": v["tracker"],
            "csv_paths": [str(p) for p in v["csv_paths"]],
        }
        for v in variants
    ]

    summary_rows: List[Dict[str, Any]] = []
    all_per_video_rows: List[Dict[str, Any]] = []

    tmp_root = Path(tempfile.mkdtemp(prefix="biohub_eval_", dir="/tmp"))
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_var = {
                executor.submit(
                    evaluate_single_variant_worker,
                    var,
                    gt_datasets,
                    est_nodes_map,
                    scale_map,
                    str(tmp_root),
                    FIXED_THRESHOLD_T,
                ): var
                for var in serializable_variants
            }

            for completed_idx, future in enumerate(
                concurrent.futures.as_completed(future_to_var), start=1
            ):
                var = future_to_var[future]
                try:
                    summary_row, video_records, elapsed = future.result()
                except Exception as exc:
                    print(
                        f"[ERROR] Failed on {var['source_group']} | {var['detector']} | {var['tracker']}: {exc}"
                    )
                    raise

                summary_rows.append(summary_row)
                all_per_video_rows.extend(video_records)

                print(
                    f"[{completed_idx:>3}/{len(variants)}] {summary_row['source_group']:<11} | "
                    f"{summary_row['detector']:<34} | {summary_row['tracker']:<10} | "
                    f"Score: {summary_row['aggregate_score']:.4f} | "
                    f"t: {summary_row['threshold_t']:.2f} | "
                    f"AssertPass: {summary_row['overall_assertion_pass_rate']:.4f} ({elapsed:.1f}s)"
                )
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    summary_df = pd.DataFrame(summary_rows)
    per_video_df = pd.DataFrame(all_per_video_rows)

    write_markdown_report(
        summary_df=summary_df,
        per_video_df=per_video_df,
        md_path=OUTPUT_MD_PATH,
        summary_csv_path=OUTPUT_SUMMARY_CSV,
        per_video_csv_path=OUTPUT_PER_VIDEO_CSV,
    )

    print("\n" + "=" * 88)
    print(f"EVALUATION COMPLETE in {time.perf_counter() - t0:.1f}s ({n_workers} workers)")
    print(f"  Markdown Report    -> {OUTPUT_MD_PATH}")
    print(f"  Summary Table CSV  -> {OUTPUT_SUMMARY_CSV}")
    print(f"  Per-Video CSV      -> {OUTPUT_PER_VIDEO_CSV}")
    print("=" * 88)


if __name__ == "__main__":
    main()[kkkkkkkkkkkkk]