"""Detection metrics against the ANNOTATED cells (~2.8% of all cells), computed in one pass over each frame.

  recall@Nx     : keep the top-N peaks per frame (N = frac x organisers' estimated cells per frame); share of annotated cells with a peak within radius_um (one-to-one)
  recall_all    : keep EVERY peak above `floor`, no cap
  peaks_per_est : peaks per frame / estimated cells per frame. recall is blind to false positives, this is the only false-positive signal here
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from .config import Config
from .constants import SCALE
from .data import io
from .data.norm import sample_scale
from .infer import centroid_peaks, get_device, predict_heat


def _hits(g, xyz, radius):
    if len(g) == 0 or len(xyz) == 0:
        return 0
    D = cdist(g, xyz)
    r, c = linear_sum_assignment(D)
    return int((D[r, c] <= radius).sum())


def evaluate(model, builder, cfg: Config, names, data_dir, device: str | None = None) -> pd.DataFrame:
    ec = cfg.eval
    device = device or get_device()
    rows = []
    for n in names:
        gt = io.load_gt(f"{data_dir}/{n}.geff")
        est = io.find_estimate(f"{data_dir}/{n}.geff")
        arr = io.open_video(data_dir, n)
        if not est:
            print(f"  {n}: no estimate in the .geff metadata - skipped")
            continue
        sc = sample_scale(arr)
        per_frame = est / arr.shape[0]
        hit = {f: 0 for f in ec.fracs}
        hit_all = tot = npk = nfr = 0
        for t in sorted(gt)[::max(1, len(gt) // ec.frames_per_video)][:ec.frames_per_video]:
            heat = predict_heat(model, builder, arr, t, sc, device, cfg.train.amp)
            idx, s = centroid_peaks(heat, ec.floor, ec.nms)
            xyz, order, g = idx * SCALE, np.argsort(-s), gt[t] * SCALE
            tot += len(g)
            npk += len(xyz)
            nfr += 1
            hit_all += _hits(g, xyz, ec.radius_um)
            for f in ec.fracs:
                keep = order[:int(round(f * per_frame))]
                hit[f] += _hits(g, xyz[keep], ec.radius_um)
        rows.append(dict(video=n, est_per_frame=round(per_frame, 1), annotated=tot,
                         **{f"recall@{f}x": round(hit[f] / max(tot, 1), 3) for f in ec.fracs},
                         recall_all=round(hit_all / max(tot, 1), 3), peaks_per_est=round(npk / max(nfr, 1) / per_frame, 2)))
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> dict:
    """mean of the per-video values, plus recall pooled over all annotated cells (a 6-cell video counts as much as a 74-cell one in the mean)"""
    cols = [c for c in df.columns if c.startswith("recall") or c == "peaks_per_est"]
    out = {c: round(float(df[c].mean()), 4) for c in cols}
    w = df["annotated"].to_numpy()
    out.update({f"{c}_pooled": round(float((df[c] * w).sum() / max(w.sum(), 1)), 4) for c in cols if c.startswith("recall")})
    out["n_videos"] = int(len(df))
    out["n_annotated"] = int(w.sum())
    return out


def save_eval(df: pd.DataFrame, summary: dict, out_dir: Path, tag: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"eval_{tag}.csv", index=False)
    (out_dir / f"eval_{tag}.json").write_text(json.dumps(summary, indent=2))
