"""Detection metrics against the ANNOTATED cells (~2.8% of all cells), computed in one pass over each frame.

  recall@Nx     : keep the top-N peaks per frame (N = frac x organisers' estimated cells per frame); share of annotated cells with a peak within radius_um (one-to-one)
  recall_all    : keep EVERY peak above `floor`, no cap
  peaks_per_est : peaks per frame / estimated cells per frame. recall is blind to false positives, this is the only false-positive signal here
Peaks are first thinned by distance suppression (eval.dup_um, 0 = off).
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
from .infer import centroid_peaks, dedupe_peaks, get_device, predict_heat


def _hits(g, xyz, radius):
    if len(g) == 0 or len(xyz) == 0:
        return 0
    D = cdist(g, xyz)
    r, c = linear_sum_assignment(D)
    return int((D[r, c] <= radius).sum())


def _frames(gt, n):
    return sorted(gt)[::max(1, len(gt) // n)][:n]


def _score(idx, s, g, ec, per_frame):
    """one frame: {'all': hits with every peak, 'npk': peaks, f: hits with the top-N peaks}"""
    xyz, order = idx * SCALE, np.argsort(-s)
    out = {"all": _hits(g, xyz, ec.radius_um), "npk": len(xyz)}
    for f in ec.fracs:
        out[f] = _hits(g, xyz[order[:int(round(f * per_frame))]], ec.radius_um)
    return out


def _row(n, per_frame, tot, acc, nfr, ec):
    return dict(video=n, est_per_frame=round(per_frame, 1), annotated=tot,
                **{f"recall@{f}x": round(acc[f] / max(tot, 1), 3) for f in ec.fracs},
                recall_all=round(acc["all"] / max(tot, 1), 3), peaks_per_est=round(acc["npk"] / max(nfr, 1) / per_frame, 2))


def evaluate(model, builder, cfg: Config, names, data_dir, device: str | None = None) -> pd.DataFrame:
    ec = cfg.eval
    device = device or get_device()
    rows = []
    for n in names:
        gt = io.load_gt(f"{data_dir}/{n}.geff")
        est = io.find_estimate(f"{data_dir}/{n}.geff")
        arr = io.open_video(data_dir, n)
        if not gt:
            print(f"  {n}: no annotated cells - skipped")
            continue
        if not est:
            print(f"  {n}: no estimate in the .geff metadata - skipped")
            continue
        sc = sample_scale(arr)
        per_frame = est / arr.shape[0]
        acc = {**{f: 0 for f in ec.fracs}, "all": 0, "npk": 0}
        tot = nfr = 0
        for t in _frames(gt, ec.frames_per_video):
            heat = predict_heat(model, builder, arr, t, sc, device, cfg.train.amp)
            idx, s = centroid_peaks(heat, ec.floor, ec.nms, ec.dup_um)
            for k, v in _score(idx, s, gt[t] * SCALE, ec, per_frame).items():
                acc[k] += v
            tot += len(gt[t])
            nfr += 1
        rows.append(_row(n, per_frame, tot, acc, nfr, ec))
    return pd.DataFrame(rows)


def sweep_dup(model, builder, cfg: Config, names, data_dir, dups, device: str | None = None) -> pd.DataFrame:
    """The heat map of every frame is computed ONCE; the distance suppression is then applied at each value in `dups`.
    -> one row per dup_um: mean over videos of recall@Nx, recall_all, peaks_per_est (+ pooled recall)."""
    ec = cfg.eval
    device = device or get_device()
    per = {d: [] for d in dups}
    for n in names:
        gt = io.load_gt(f"{data_dir}/{n}.geff")
        est = io.find_estimate(f"{data_dir}/{n}.geff")
        arr = io.open_video(data_dir, n)
        if not est:
            continue
        sc = sample_scale(arr)
        per_frame = est / arr.shape[0]
        acc = {d: {**{f: 0 for f in ec.fracs}, "all": 0, "npk": 0} for d in dups}
        tot = nfr = 0
        for t in _frames(gt, ec.frames_per_video):
            heat = predict_heat(model, builder, arr, t, sc, device, cfg.train.amp)
            idx0, s0 = centroid_peaks(heat, ec.floor, ec.nms, 0.0)
            for d in dups:
                idx, s = dedupe_peaks(idx0, s0, d)
                for k, v in _score(idx, s, gt[t] * SCALE, ec, per_frame).items():
                    acc[d][k] += v
            tot += len(gt[t])
            nfr += 1
        for d in dups:
            per[d].append(_row(n, per_frame, tot, acc[d], nfr, ec))
    out = []
    for d in dups:
        df = pd.DataFrame(per[d])
        s = summarise(df)
        s["dup_um"] = d
        out.append(s)
    return pd.DataFrame(out).set_index("dup_um")


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
