"""Detections (parquet from biohub detect) -> tracks -> .geff in the organisers' format."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .link import LinkConfig, link


FEATURES = {"raw": "raw", "dog": "dog", "erode": "erode", "P": "P"}
VARIANTS = {"v0": [], "v1": ["raw"], "v2": ["dog"], "v3": ["dog", "raw"], "v4": ["dog", "raw", "erode"]}


def read_detections(path, thr: float, feat_path=None, feats=()):
    """-> frames (list of (N,3) um arrays), voxel indices per frame, real_idx list, T, feature arrays per frame (or None).
    Detections below score thr are dropped. Features (log of the value + 1e-3, negatives clipped to 0) come from a file with the SAME rows as the detections file."""
    full = pd.read_parquet(path)
    T = int(full["t"].max()) + 1
    dup = np.zeros(T, bool)
    if "dup_frame" in full.columns:                          # older detection files have no such column: then no frame counts as repeated
        g = full.groupby("t")["dup_frame"].first()
        dup[g.index.to_numpy()] = g.to_numpy().astype(bool)
    dup[0] = False
    real = np.cumsum(~dup) - 1
    keep = (full["score"] >= thr).to_numpy()
    F = None
    if feats:
        fd = pd.read_parquet(feat_path)
        assert len(fd) == len(full), f"{feat_path}: {len(fd)} rows but {len(full)} detections - extract the features again"
        F = np.log(np.clip(fd[list(feats)].to_numpy(float), 0, None) + 1e-3)
    um, vox, fts = [], [], []
    for t in range(T):
        m = ((full["t"] == t).to_numpy()) & keep
        d = full[m]
        um.append(d[["z_um", "y_um", "x_um"]].to_numpy(float))
        vox.append(d[["z", "y", "x"]].to_numpy())
        if F is not None:
            fts.append(F[m])
    return um, vox, real.tolist(), T, (fts if F is not None else None)


def track_video(path, thr: float, cfg: LinkConfig, keep_singletons: bool = False, feat_path=None, feats=(), timer: dict | None = None):
    """-> nodes DataFrame (id, t, z, y, x voxel) and edges array (source id, target id). Only nodes with an edge are kept unless keep_singletons."""
    import time
    t0 = time.perf_counter()
    um, vox, real, T, fts = read_detections(path, thr, feat_path, feats)
    t1 = time.perf_counter()
    edges = link(um, real, cfg, fts)
    t2 = time.perf_counter()
    offs = np.cumsum([0] + [len(v) for v in vox])
    e = np.array([(offs[t] + i, offs[t + 1] + j) for t, i, j in edges], dtype=np.int64).reshape(-1, 2)
    allv = np.concatenate([v.reshape(-1, 3) for v in vox]) if offs[-1] else np.zeros((0, 3))
    tt = np.concatenate([np.full(len(v), t) for t, v in enumerate(vox)]) if offs[-1] else np.zeros(0, int)
    keep = np.ones(offs[-1], bool) if keep_singletons else np.isin(np.arange(offs[-1]), e.ravel())
    newid = np.cumsum(keep) - 1
    nodes = pd.DataFrame({"id": newid[keep], "t": tt[keep], "z": allv[keep, 0], "y": allv[keep, 1], "x": allv[keep, 2]})
    if timer is not None:
        timer["read_s"] = timer.get("read_s", 0.0) + (t1 - t0); timer["link_s"] = timer.get("link_s", 0.0) + (t2 - t1)
        timer["assemble_s"] = timer.get("assemble_s", 0.0) + (time.perf_counter() - t2)
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
