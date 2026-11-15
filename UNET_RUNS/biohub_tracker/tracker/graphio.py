"""Detections (parquet from biohub detect) -> tracks -> .geff in the organisers' format."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .link import LinkConfig, link


def read_detections(path, thr: float):
    """-> frames (list of (N,3) um arrays), voxel indices per frame, real_idx list, T. Detections below score thr are dropped."""
    full = pd.read_parquet(path)
    T = int(full["t"].max()) + 1
    dup = np.zeros(T, bool)
    if "dup_frame" in full.columns:                          # older detection files have no such column: then no frame counts as repeated
        g = full.groupby("t")["dup_frame"].first()
        dup[g.index.to_numpy()] = g.to_numpy().astype(bool)
    dup[0] = False
    real = np.cumsum(~dup) - 1
    df = full[full["score"] >= thr]
    um, vox = [], []
    for t in range(T):
        d = df[df["t"] == t]
        um.append(d[["z_um", "y_um", "x_um"]].to_numpy(float))
        vox.append(d[["z", "y", "x"]].to_numpy())
    return um, vox, real.tolist(), T


def track_video(path, thr: float, cfg: LinkConfig, keep_singletons: bool = False):
    """-> nodes DataFrame (id, t, z, y, x voxel) and edges array (source id, target id). Only nodes with an edge are kept unless keep_singletons."""
    um, vox, real, T = read_detections(path, thr)
    edges = link(um, real, cfg)
    offs = np.cumsum([0] + [len(v) for v in vox])
    e = np.array([(offs[t] + i, offs[t + 1] + j) for t, i, j in edges], dtype=np.int64).reshape(-1, 2)
    allv = np.concatenate([v.reshape(-1, 3) for v in vox]) if offs[-1] else np.zeros((0, 3))
    tt = np.concatenate([np.full(len(v), t) for t, v in enumerate(vox)]) if offs[-1] else np.zeros(0, int)
    keep = np.ones(offs[-1], bool) if keep_singletons else np.isin(np.arange(offs[-1]), e.ravel())
    newid = np.cumsum(keep) - 1
    nodes = pd.DataFrame({"id": newid[keep], "t": tt[keep], "z": allv[keep, 0], "y": allv[keep, 1], "x": allv[keep, 2]})
    return nodes, np.stack([newid[e[:, 0]], newid[e[:, 1]]], 1) if len(e) else e


def write_geff(nodes: pd.DataFrame, edges: np.ndarray, out: Path):
    import polars as pl
    import shutil
    import tracksdata as td
    g = td.graph.InMemoryGraph()
    for k in ("z", "y", "x"):
        g.add_node_attr_key(k, pl.Float64, -999999.0)
    ids = g.bulk_add_nodes(nodes.assign(**{k: nodes[k].astype(float) for k in "zyx"})[["t", "z", "y", "x"]].astype({"t": "int64"}).to_dict("records"))
    if len(edges):
        g.bulk_add_edges([{"source_id": ids[s], "target_id": ids[t]} for s, t in edges])
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        shutil.rmtree(out)
    g.to_geff(out)
