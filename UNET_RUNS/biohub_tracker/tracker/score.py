"""Official scoring (tracking_cellmot.metrics, vendored unchanged in tracker/official) of predicted .geff files against ground-truth .geff files.
Same procedure as the organisers' scripts/evaluate.py: voxel scale from the video's zarr metadata, T_true from the GT's estimated_number_of_nodes."""
from __future__ import annotations

from pathlib import Path

import zarr

from .official.metrics import evaluate as compute_metric, node_recall, per_sample_metrics, summarise

DEFAULT_SCALE = (1.625, 0.40625, 0.40625)


def _graph(p):
    import tracksdata as td
    r = td.graph.IndexedRXGraph.from_geff(str(p))
    return r[0] if isinstance(r, tuple) else r


def read_scale(gt_dir: Path, name: str):
    try:
        a = dict(zarr.open_group(str(gt_dir / f"{name}.zarr"), mode="r").attrs)
        return tuple(a["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"][-3:])
    except Exception:
        return DEFAULT_SCALE


def estimated_nodes(geff_path: Path) -> float:
    from geff import GeffMetadata
    try:
        v = (GeffMetadata.read(geff_path).extra or {}).get("estimated_number_of_nodes")
        return float(v) if v is not None else float("nan")
    except Exception:
        return float("nan")


def score_dir(pred_dir, gt_dir, names=None, verbose=False):
    """-> (summary dict, per-video rows). Videos without annotated cells (empty .geff) are skipped, like the organisers do (no edges to score)."""
    pred_dir, gt_dir = Path(pred_dir), Path(gt_dir)
    names = names or sorted({p.stem for p in pred_dir.glob("*.geff")} & {p.stem for p in gt_dir.glob("*.geff")})
    rows, used = [], []
    for n in names:
        gt = _graph(gt_dir / f"{n}.geff")
        if gt.num_nodes() == 0 or gt.num_edges() == 0:
            continue
        pr = _graph(pred_dir / f"{n}.geff")
        er = compute_metric(pr, gt, scale=read_scale(gt_dir, n), max_distance=7.0)
        rec = node_recall(pr, gt) if pr.num_edges() and pr.num_nodes() else 0.0
        row = per_sample_metrics(er, estimated_nodes(gt_dir / f"{n}.geff"), rec)
        row["video"] = n
        rows.append(row); used.append(n)
        if verbose:
            print(f"  {n}: edge TP/FP/FN {er.edge_tp}/{er.edge_fp}/{er.edge_fn}  div {er.division_tp}/{er.division_fp}/{er.division_fn}  pred nodes {er.num_pred_nodes}")
    s = summarise(rows)
    s["n_videos"] = len(rows)
    return s, rows
