"""
TrackMate Sparse LAP Cell-Division Lineage Tracker Module
=========================================================

1. HOW TO IMPORT
----------------
    from trackers import init_fiji_headless
    from trackers.lap_division_tracker import track_cell_divisions

2. EXPECTED INPUT DATA & PREPROCESSING REQUIREMENTS
---------------------------------------------------
    - `spots_df` (`pd.DataFrame`):
        * Direct output from any detector in `detectors/` (`log_detector`, `dog_detector`,
          or `stardist_3d_detector`), or any custom DataFrame containing at minimum:
            - `frame` (`int`): Time index `t`.
            - `z`, `y`, `x` (`float`): Physical centroid coordinates in calibrated units.
        * Optional columns preserved in output: `spot_id`, `z_px`, `y_px`, `x_px`, `radius`, `quality`.
        * Data Type / Preprocessing Requirement: **Pre-computed spot centroids only.** No raw
          or filtered image array is required because this module bypasses TrackMate's internal
          detectors and executes `TrackMate.execTracking()` directly on the injected `SpotCollection`.
    - `max_steps` (`int`):
        * Number of post-division time steps (`t`) to follow each daughter cell after mitosis
          (and up to `max_steps` pre-division time steps of the parent cell leading into the split).
    - `ij` (`imagej.ImageJ | None`):
        * Active headless ImageJ gateway from `init_fiji_headless()`. If `None`, initializes one.

3. OUTPUT SPECIFICATION
-----------------------
    Returns a `dict[str, pd.DataFrame]` with two isolated cell-division tables (non-dividing
    tracks are completely discarded):

    1. `"events"` (`pd.DataFrame`) — One row per isolated mitotic division event:
        - `division_id` (`int64`): Unique 0-indexed division event ID.
        - `track_id` (`int64`): Parent TrackMate lineage ID.
        - `division_frame` (`int64`): Frame `t_div` of the parent cell immediately before splitting.
        - `parent_spot_id` (`int64`): Input `spot_id` of the dividing parent cell at `t_div`.
        - `daughter_1_spot_id` (`int64`): Input `spot_id` of daughter 1 at its first post-split frame.
        - `daughter_2_spot_id` (`int64`): Input `spot_id` of daughter 2 at its first post-split frame.
        - `daughter_1_steps_tracked` (`int64`): Number of post-division steps tracked for daughter 1 (up to `max_steps`).
        - `daughter_2_steps_tracked` (`int64`): Number of post-division steps tracked for daughter 2 (up to `max_steps`).
        - `split_distance_phys` (`float64`): Euclidean physical distance between daughter 1 and daughter 2 at split.

    2. `"trajectories"` (`pd.DataFrame`) — Frame-by-frame trajectories for each isolated division:
        - `division_id` (`int64`): Foreign key linking to `events["division_id"]`.
        - `track_id` (`int64`): TrackMate lineage ID.
        - `role` (`str`): `"parent"`, `"daughter_1"`, or `"daughter_2"`.
        - `step_offset` (`int64`): Relative step from division (`0` = parent at `t_div`,
          `-1, -2, ...` = pre-division parent frames, `+1, +2, ..., +max_steps` = daughter frames).
        - `spot_id` (`int64`): Input `spot_id` from `spots_df`.
        - `frame` (`int64`): Time frame index `t`.
        - `z`, `y`, `x` (`float64`): Physical coordinates.
        - `z_px`, `y_px`, `x_px` (`float64`): Voxel coordinates (NaN if not supplied in `spots_df`).
        - `radius`, `quality` (`float64`): Spot metadata.
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import jpype
import scyjava as sj


def _trace_linear_branch(
    start_node: int,
    children_map: Dict[int, List[int]],
    max_steps: int,
) -> List[int]:
    """Follow a single daughter branch forward for up to `max_steps`, stopping if it divides again."""
    branch: List[int] = []
    curr = start_node
    for _ in range(max_steps):
        branch.append(curr)
        next_nodes = children_map.get(curr, [])
        if len(next_nodes) != 1:
            break
        curr = next_nodes[0]
    return branch


def _trace_parent_history(
    parent_node: int,
    parents_map: Dict[int, List[int]],
    children_map: Dict[int, List[int]],
    max_steps: int,
) -> List[int]:
    """Walk backward along the parent tracklet up to `max_steps` frames prior to division."""
    history = [parent_node]
    curr = parent_node
    for _ in range(max_steps):
        prev_nodes = parents_map.get(curr, [])
        if len(prev_nodes) != 1:
            break
        prev = prev_nodes[0]
        # Stop if the ancestor itself was a bifurcation node
        if len(children_map.get(prev, [])) != 1:
            break
        history.append(prev)
        curr = prev
    history.reverse()
    return history


def track_cell_divisions(
    spots_df: pd.DataFrame,
    max_steps: int = 5,
    ij: Optional[Any] = None,
    linking_max_distance: float = 15.0,
    gap_closing_max_distance: float = 15.0,
    max_frame_gap: int = 2,
    splitting_max_distance: float = 15.0,
    dt: float = 1.0,
) -> Dict[str, pd.DataFrame]:
    """
    Run TrackMate's Sparse LAP tracker on pre-computed spots and isolate cell divisions.

    Parameters
    ----------
    spots_df : pd.DataFrame
        Detected centroids table with required columns `['frame', 'z', 'y', 'x']`.
    max_steps : int
        Maximum number of post-division time steps (`t`) to track each daughter cell
        (and pre-division parent steps).
    ij : imagej.ImageJ, optional
        Initialized headless PyImageJ instance.
    linking_max_distance : float
        Max frame-to-frame linking distance in physical units.
    gap_closing_max_distance : float
        Max gap-closing distance in physical units.
    max_frame_gap : int
        Max frame gap allowed for gap closing (set `0` to disable gap closing).
    splitting_max_distance : float
        Max distance in physical units allowed between a dividing parent and its two daughters.
    dt : float
        Time interval between consecutive frames.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary with keys `"events"` and `"trajectories"` containing strictly isolated
        mitotic division events and their parent/daughter tracks.
    """
    from . import init_fiji_headless

    event_cols = [
        "division_id",
        "track_id",
        "division_frame",
        "parent_spot_id",
        "daughter_1_spot_id",
        "daughter_2_spot_id",
        "daughter_1_steps_tracked",
        "daughter_2_steps_tracked",
        "split_distance_phys",
    ]
    traj_cols = [
        "division_id",
        "track_id",
        "role",
        "step_offset",
        "spot_id",
        "frame",
        "z",
        "y",
        "x",
        "z_px",
        "y_px",
        "x_px",
        "radius",
        "quality",
    ]

    required = {"frame", "z", "y", "x"}
    if not required.issubset(spots_df.columns):
        missing = required - set(spots_df.columns)
        raise ValueError(f"spots_df is missing required columns: {missing}")

    if spots_df.empty:
        return {
            "events": pd.DataFrame(columns=event_cols),
            "trajectories": pd.DataFrame(columns=traj_cols),
        }

    if ij is None:
        ij = init_fiji_headless()

    Model = sj.jimport("fiji.plugin.trackmate.Model")
    Settings = sj.jimport("fiji.plugin.trackmate.Settings")
    Spot = sj.jimport("fiji.plugin.trackmate.Spot")
    TrackMate = sj.jimport("fiji.plugin.trackmate.TrackMate")
    SparseLAPTrackerFactory = sj.jimport(
        "fiji.plugin.trackmate.tracking.jaqaman.SparseLAPTrackerFactory"
    )

    df = spots_df.copy().reset_index(drop=True)
    if "spot_id" not in df.columns:
        df["spot_id"] = np.arange(len(df), dtype=np.int64)
    if "radius" not in df.columns:
        df["radius"] = 2.5
    if "quality" not in df.columns:
        df["quality"] = 1.0
    for px_col in ("z_px", "y_px", "x_px"):
        if px_col not in df.columns:
            df[px_col] = np.nan

    # 1. Inject pre-computed spots directly into TrackMate Model (no detector executed)
    model = Model()
    java_id_to_row: Dict[int, Dict[str, Any]] = {}

    model.beginUpdate()
    try:
        for row in df.to_dict(orient="records"):
            j_spot = Spot(
                float(row["x"]),
                float(row["y"]),
                float(row["z"]),
                float(row["radius"]),
                float(row["quality"]),
            )
            model.addSpotTo(j_spot, jpype.JInt(int(row["frame"])))
            java_id_to_row[int(j_spot.ID())] = row
    finally:
        model.endUpdate()

    # 2. Configure Sparse LAP Tracker with Track Splitting enabled & Merging disabled
    settings = Settings()
    settings.dt = float(dt)
    settings.trackerFactory = SparseLAPTrackerFactory()

    trk_settings = settings.trackerFactory.getDefaultSettings()
    trk_settings.put("LINKING_MAX_DISTANCE", float(linking_max_distance))
    trk_settings.put("GAP_CLOSING_MAX_DISTANCE", float(gap_closing_max_distance))
    trk_settings.put("MAX_FRAME_GAP", jpype.JInt(int(max_frame_gap)))
    trk_settings.put("ALLOW_TRACK_SPLITTING", True)
    trk_settings.put("SPLITTING_MAX_DISTANCE", float(splitting_max_distance))
    trk_settings.put("ALLOW_TRACK_MERGING", False)
    settings.trackerSettings = trk_settings

    tm = TrackMate(model, settings)
    if not tm.execTracking():
        raise RuntimeError(f"TrackMate LAP tracking failed: {tm.getErrorMessage()}")

    # 3. Parse lineage graph and isolate strictly binary cell division events (out-degree == 2)
    track_model = model.getTrackModel()
    event_rows: List[Dict[str, Any]] = []
    traj_rows: List[Dict[str, Any]] = []
    division_counter = 0

    for j_track_id in sorted(int(tid) for tid in track_model.trackIDs(True)):
        children_map: Dict[int, List[int]] = defaultdict(list)
        parents_map: Dict[int, List[int]] = defaultdict(list)

        for edge in track_model.trackEdges(jpype.JInt(j_track_id)):
            src_spot = track_model.getEdgeSource(edge)
            tgt_spot = track_model.getEdgeTarget(edge)
            src_jid = int(src_spot.ID())
            tgt_jid = int(tgt_spot.ID())

            # Enforce forward temporal direction
            if java_id_to_row[src_jid]["frame"] > java_id_to_row[tgt_jid]["frame"]:
                src_jid, tgt_jid = tgt_jid, src_jid

            children_map[src_jid].append(tgt_jid)
            parents_map[tgt_jid].append(src_jid)

        # Sort division nodes chronologically by frame
        split_nodes = [
            jid for jid, ch in children_map.items() if len(ch) == 2
        ]
        split_nodes.sort(key=lambda jid: (java_id_to_row[jid]["frame"], jid))

        for parent_jid in split_nodes:
            d1_jid, d2_jid = sorted(
                children_map[parent_jid],
                key=lambda jid: (
                    java_id_to_row[jid]["frame"],
                    java_id_to_row[jid]["x"],
                    java_id_to_row[jid]["y"],
                ),
            )

            parent_history = _trace_parent_history(
                parent_jid, parents_map, children_map, max_steps=max_steps
            )
            d1_branch = _trace_linear_branch(d1_jid, children_map, max_steps=max_steps)
            d2_branch = _trace_linear_branch(d2_jid, children_map, max_steps=max_steps)

            p_info = java_id_to_row[parent_jid]
            d1_info = java_id_to_row[d1_jid]
            d2_info = java_id_to_row[d2_jid]

            split_dist = float(
                np.sqrt(
                    (d1_info["z"] - d2_info["z"]) ** 2
                    + (d1_info["y"] - d2_info["y"]) ** 2
                    + (d1_info["x"] - d2_info["x"]) ** 2
                )
            )

            event_rows.append(
                {
                    "division_id": division_counter,
                    "track_id": j_track_id,
                    "division_frame": int(p_info["frame"]),
                    "parent_spot_id": int(p_info["spot_id"]),
                    "daughter_1_spot_id": int(d1_info["spot_id"]),
                    "daughter_2_spot_id": int(d2_info["spot_id"]),
                    "daughter_1_steps_tracked": len(d1_branch),
                    "daughter_2_steps_tracked": len(d2_branch),
                    "split_distance_phys": split_dist,
                }
            )

            # Record pre-division parent trajectory (step_offset <= 0, 0 == division frame)
            n_hist = len(parent_history)
            for idx, jid in enumerate(parent_history):
                info = java_id_to_row[jid]
                traj_rows.append(
                    {
                        "division_id": division_counter,
                        "track_id": j_track_id,
                        "role": "parent",
                        "step_offset": idx - (n_hist - 1),
                        "spot_id": int(info["spot_id"]),
                        "frame": int(info["frame"]),
                        "z": float(info["z"]),
                        "y": float(info["y"]),
                        "x": float(info["x"]),
                        "z_px": float(info["z_px"]),
                        "y_px": float(info["y_px"]),
                        "x_px": float(info["x_px"]),
                        "radius": float(info["radius"]),
                        "quality": float(info["quality"]),
                    }
                )

            # Record daughter 1 & daughter 2 trajectories (step_offset = +1 .. +max_steps)
            for role_name, branch in (("daughter_1", d1_branch), ("daughter_2", d2_branch)):
                for step_idx, jid in enumerate(branch, start=1):
                    info = java_id_to_row[jid]
                    traj_rows.append(
                        {
                            "division_id": division_counter,
                            "track_id": j_track_id,
                            "role": role_name,
                            "step_offset": step_idx,
                            "spot_id": int(info["spot_id"]),
                            "frame": int(info["frame"]),
                            "z": float(info["z"]),
                            "y": float(info["y"]),
                            "x": float(info["x"]),
                            "z_px": float(info["z_px"]),
                            "y_px": float(info["y_px"]),
                            "x_px": float(info["x_px"]),
                            "radius": float(info["radius"]),
                            "quality": float(info["quality"]),
                        }
                    )

            division_counter += 1

    events_df = pd.DataFrame(event_rows, columns=event_cols) if event_rows else pd.DataFrame(columns=event_cols)
    trajectories_df = pd.DataFrame(traj_rows, columns=traj_cols) if traj_rows else pd.DataFrame(columns=traj_cols)

    return {
        "events": events_df,
        "trajectories": trajectories_df,
    }