"""End-to-end on synthetic data: moving + dividing cells -> detections parquet (position noise, misses, false positives, one repeated frame)
-> tracker -> .geff -> the organisers' scorer.   Run from the biohub_tracker folder:  python -m pytest -q tests"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tracker.cli import main                                    # noqa: E402
from tracker.graphio import write_geff                          # noqa: E402

SC = np.array([1.625, 0.40625, 0.40625])


def simulate(seed=0, T=30, n0=25, p_div=0.04, dup_at=12, miss=0.03, fp_per_frame=3, noise_vox=1.0):
    """-> truth nodes DataFrame(id,t,z,y,x voxels), truth edges (n,2), detections DataFrame (parquet layout)"""
    rng = np.random.default_rng(seed)
    cells = [(rng.uniform([5, 80, 80], [35, 300, 300]), rng.normal(0, 1, 3) * [0.3, 4, 4]) for _ in range(n0)]   # (position, velocity) in voxels / frame
    nodes, edges, nid, prev = [], [], 0, None
    for t in range(T):
        cur = []
        for k, (p, v) in enumerate(cells):
            nodes.append((nid, t, *p)); cur.append(nid); nid += 1
        if prev is not None:
            for a, b in prev:
                edges.append((a, b))
        if t == T - 1:
            break
        nxt, links = [], []
        for k, (p, v) in enumerate(cells):
            step = 0 if t + 1 == dup_at else 1                              # a repeated frame: the image (and so every cell) is where it was
            if rng.random() < p_div and step:
                for s in (-1, 1):
                    nxt.append((p + v + s * np.array([0, 3, 3]), v + s * np.array([0, 1.0, 1.0]))); links.append((cur[k], None))
            else:
                nxt.append((p + v * step, v)); links.append((cur[k], None))
        # parents -> children as (node id now, index in next frame); ids of the next frame are assigned in order
        prev = []
        base = nid
        for j, (pa, _) in enumerate(links):
            prev.append((pa, base + j))
        cells = nxt
    nd = pd.DataFrame(nodes, columns=["id", "t", "z", "y", "x"])
    ed = np.array(edges, dtype=np.int64).reshape(-1, 2)
    det = []
    for t in range(T):
        d = nd[nd["t"] == t]
        d = d[rng.random(len(d)) > miss]
        pos = d[["z", "y", "x"]].to_numpy() + rng.normal(0, noise_vox, (len(d), 3)) * [0.4, 1, 1]
        fp = rng.uniform([5, 60, 60], [35, 320, 320], (fp_per_frame, 3))
        pts = np.round(np.vstack([pos, fp])).astype(int)
        sc = np.concatenate([rng.uniform(0.5, 1.0, len(d)), rng.uniform(0.02, 0.25, fp_per_frame)])
        det.append(pd.DataFrame({"t": t, "z": pts[:, 0], "y": pts[:, 1], "x": pts[:, 2], "z_um": pts[:, 0] * SC[0], "y_um": pts[:, 1] * SC[1], "x_um": pts[:, 2] * SC[2],
                                 "score": sc, "dup_frame": t == dup_at}))
    return nd, ed, pd.concat(det, ignore_index=True)


def make_dataset(tmp: Path, name="vid", **kw):
    nd, ed, det = simulate(**kw)
    gt, dd = tmp / "gt", tmp / "det"
    gt.mkdir(parents=True, exist_ok=True); dd.mkdir(parents=True, exist_ok=True)
    keep = nd.sample(frac=0.7, random_state=0)                              # sparse ground truth: 70% of the cells annotated
    ids = set(keep["id"])
    e = np.array([(a, b) for a, b in ed if a in ids and b in ids], dtype=np.int64).reshape(-1, 2)
    remap = {old: i for i, old in enumerate(keep["id"])}
    write_geff(keep.assign(id=keep["id"].map(remap)), np.array([(remap[a], remap[b]) for a, b in e], dtype=np.int64).reshape(-1, 2), gt / f"{name}.geff")
    import zarr
    g = zarr.open_group(str(gt / f"{name}.geff"), mode="r+")             # T_true: the organisers' estimate of the number of nodes (here the true one)
    meta = dict(g.attrs["geff"]); meta.setdefault("extra", {})["estimated_number_of_nodes"] = int(len(nd))
    g.attrs["geff"] = meta
    det.to_parquet(dd / f"{name}.parquet")
    return gt, dd


def test_tracker_scores_well_on_synthetic(tmp_path):
    gt, dd = make_dataset(tmp_path)
    out = tmp_path / "out"
    main(["run", "--det-dir", str(dd), "--gt-dir", str(gt), "--out-dir", str(out), "--thr", "0.3", "--score"])
    import json
    s = json.loads((out / "score.json").read_text())
    print(s)
    assert s["n_videos"] == 1
    assert s["edge_jaccard"] > 0.85, s


def test_false_positives_below_threshold_do_not_become_nodes(tmp_path):
    gt, dd = make_dataset(tmp_path, fp_per_frame=6)
    out = tmp_path / "out"
    main(["run", "--det-dir", str(dd), "--gt-dir", str(gt), "--out-dir", str(out), "--thr", "0.3"])
    import tracksdata as td
    g = td.graph.IndexedRXGraph.from_geff(str(out / "vid.geff")); g = g[0] if isinstance(g, tuple) else g
    n_det = int((pd.read_parquet(dd / "vid.parquet")["score"] >= 0.3).sum())
    assert g.num_nodes() <= n_det
    deg = np.bincount(np.concatenate([g.edge_attrs()["source_id"].to_numpy(), g.edge_attrs()["target_id"].to_numpy()]), minlength=1)
    assert (deg[deg > 0] >= 1).all()
