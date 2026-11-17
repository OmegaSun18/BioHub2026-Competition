"""Where does a model fail? Several trained variants are run on the SAME randomly sampled annotated cells and drawn side by side.

Sampling: rng = np.random.default_rng(seed) (default 8001). For each sample, a video, then a frame that has annotations, then one annotated cell of that frame.
Every variant sees exactly the same (video, frame, cell) list, so differences between columns are the models, not the sample.

Pictures (output.png): one row per sample, columns = raw frame, P (raw -> balance -> dehaze -> floor), then the RAW heat output of each model.
Linear grey scale, nearest-neighbour, no smoothing. The slice shown is the z-slice of the sampled cell, cropped around it.
  cyan ring : the sampled annotated cell      cyan dot : other annotated cells within 1 z-slice
  red x     : peaks of the model within 2 z-slices (every peak above eval.floor, no cap)
Table (output.csv): per variant and sample: hit / miss within radius_um (one-to-one over the whole frame), distance to the nearest peak,
heat at the cell centre, highest heat within radius_um of the cell, number of peaks in the frame, recall of the whole frame.
A MISS with high `heat_max_near` means the model painted the cell but a neighbouring peak took the match; a MISS with ~0 means the model saw nothing there.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from .constants import SCALE
from .data import io
from .data.norm import norm_raw, sample_scale
from .data.split import split_videos
from .features.filters import p_pipeline
from .infer import centroid_peaks, get_device, load_checkpoint, predict_heat


def sample_cells(names, data_dir, n, seed):
    """-> [(video, t, cell index k)] drawn with np.random.default_rng(seed)"""
    rng = np.random.default_rng(seed)
    vids = [str(v) for v in rng.choice(np.array(names), size=n, replace=len(names) < n)]
    out, cache = [], {}
    for v in vids:
        gt = cache.setdefault(v, io.load_gt(f"{data_dir}/{v}.geff"))
        ts = sorted(gt)
        t = int(rng.choice(ts))
        out.append((v, t, int(rng.integers(len(gt[t])))))
    return out


def _near(heat, c, radius_um):
    """highest heat within radius_um (ellipsoid in um) of voxel c"""
    r = np.ceil(radius_um / SCALE).astype(int)
    lo = np.maximum(c - r, 0)
    hi = np.minimum(c + r + 1, heat.shape)
    zz, yy, xx = np.meshgrid(*[np.arange(l, h) for l, h in zip(lo, hi)], indexing="ij")
    d = np.sqrt(((zz - c[0]) * SCALE[0]) ** 2 + ((yy - c[1]) * SCALE[1]) ** 2 + ((xx - c[2]) * SCALE[2]) ** 2)
    w = heat[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    return float(w[d <= radius_um].max())


def _match(g_um, p_um, radius):
    """one-to-one matching of annotated cells to peaks; -> bool array (hit per annotated cell)"""
    hit = np.zeros(len(g_um), bool)
    if len(g_um) and len(p_um):
        D = cdist(g_um, p_um)
        r, c = linear_sum_assignment(D)
        hit[r] = D[r, c] <= radius
    return hit


def diagnose(run_dirs, out_png, split="val", n_samples=8, seed=8001, half_xy=48, ckpt="last.pt", dup_um=None):
    device = get_device()
    models = []
    for rd in run_dirs:
        rd = Path(rd)
        m, cfg, b = load_checkpoint(rd / "ckpt" / ckpt, device)
        models.append((rd.name, m, cfg, b))
    cfg0 = models[0][2]
    data_dir = str(Path(cfg0.data.root) / cfg0.data.subdir)
    names = io.list_videos(data_dir)
    train_names, val_names = split_videos(names, cfg0.data.n_val, cfg0.data.split_seed)
    pool = {"val": val_names, "train": train_names, "all": names}[split]
    samples = sample_cells(pool, data_dir, n_samples, seed)
    ec = cfg0.eval
    dup = ec.dup_um if dup_um is None else dup_um
    ncol = 2 + len(models)
    fig, axes = plt.subplots(len(samples), ncol, figsize=(5.2 * ncol, 5.4 * len(samples)), squeeze=False)
    rows, heat_cache, frame_cache = [], {}, {}
    for si, (v, t, k) in enumerate(samples):
        gt = io.load_gt(f"{data_dir}/{v}.geff")[t]
        arr = io.open_video(data_dir, v)
        sc = sample_scale(arr)
        c = gt[k].astype(int)
        cz, cy, cx = [int(a) for a in c]
        R = torch.from_numpy(norm_raw(np.asarray(arr[t], np.float32), sc))[None, None].to(device)
        P = p_pipeline(R, cfg0.features)[0, 0].cpu().numpy()
        raw = R[0, 0].cpu().numpy()
        Z, Y, X = raw.shape
        y0 = int(np.clip(cy - half_xy, 0, max(Y - 2 * half_xy, 0)))
        x0 = int(np.clip(cx - half_xy, 0, max(X - 2 * half_xy, 0)))
        ys, xs = slice(y0, y0 + 2 * half_xy), slice(x0, x0 + 2 * half_xy)

        def draw(ax, img, title, vmax, peaks=None):
            w = 2 * half_xy
            inside = lambda p: (p[:, 1] >= y0) & (p[:, 1] < y0 + w) & (p[:, 2] >= x0) & (p[:, 2] < x0 + w)        # only points inside the crop
            ax.imshow(img[cz, ys, xs], cmap="gray", vmin=0, vmax=max(vmax, 1e-6), interpolation="nearest", aspect="equal")
            ax.set_title(title, fontsize=11)
            others = np.array([g for g in gt if abs(int(g[0]) - cz) <= 1 and not np.array_equal(g, c)]).reshape(-1, 3)    # other annotated cells near this slice
            if len(others):
                others = others[inside(others)]
                ax.plot(others[:, 2] - x0, others[:, 1] - y0, ".", color="cyan", ms=7)
            ax.scatter([cx - x0], [cy - y0], s=420, facecolors="none", edgecolors="cyan", linewidths=1.8)
            if peaks is not None and len(peaks):
                sel = peaks[np.abs(peaks[:, 0] - cz) <= 2]
                sel = sel[inside(sel)]
                ax.plot(sel[:, 2] - x0, sel[:, 1] - y0, "x", color="red", ms=9, mew=1.8)
            ax.set_xlim(-0.5, w - 0.5)                                       # the axes ARE the crop, whatever was drawn
            ax.set_ylim(w - 0.5, -0.5)
            ax.axis("off")

        draw(axes[si, 0], raw, f"{v}  t={t}  z={cz}\nraw (per-video normalised)", float(np.percentile(raw[cz, ys, xs], 99.9)))
        draw(axes[si, 1], P, "P = raw -> balance -> dehaze -> floor", float(np.percentile(P[cz, ys, xs], 99.9)))
        for mi, (mname, model, cfg, builder) in enumerate(models):
            key = (mname, v, t)
            if key not in heat_cache:
                heat_cache[key] = predict_heat(model, builder, arr, t, sc, device, cfg.train.amp)
            ht = heat_cache[key]
            idx, s = centroid_peaks(ht, ec.floor, ec.nms, dup)
            h = ht.cpu().numpy()
            hit = _match(gt * SCALE, idx * SCALE, ec.radius_um)
            d = float(cdist((c * SCALE)[None], idx * SCALE).min()) if len(idx) else float("inf")
            rec = dict(variant=mname, video=v, t=t, cell_z=cz, cell_y=cy, cell_x=cx, hit=bool(hit[k]), nearest_peak_um=round(d, 2),
                       heat_at_cell=round(float(h[cz, cy, cx]), 4), heat_max_near=round(_near(h, c, ec.radius_um), 4),
                       peaks_in_frame=len(idx), frame_recall=round(float(hit.mean()), 3))
            rows.append(rec)
            tag = "HIT" if rec["hit"] else "MISS"
            draw(axes[si, 2 + mi], h, f"{mname}: {tag}  d={d:.1f}um\nheat@cell {rec['heat_at_cell']:.2f}  max near {rec['heat_max_near']:.2f}  frame recall {rec['frame_recall']:.2f}", 1.0, idx)
    fig.suptitle(f"samples drawn with default_rng({seed}) from the {split} videos   |   heat: linear grey 0..1, raw model output   |   dup_um={dup}", fontsize=14, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.45 / (5.4 * len(samples))))
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=80)
    plt.close(fig)
    df = pd.DataFrame(rows)
    df.to_csv(out_png.with_suffix(".csv"), index=False)
    print(df.to_string(index=False))
    summ = df.groupby("variant").agg(hit_rate=("hit", "mean"), mean_frame_recall=("frame_recall", "mean"), mean_peaks=("peaks_in_frame", "mean"))
    print(summ.round(3).to_string())
    print(f"wrote {out_png} and {out_png.with_suffix('.csv')}")
    return df
