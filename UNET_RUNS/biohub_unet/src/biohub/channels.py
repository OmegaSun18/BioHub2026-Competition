"""Look at the input channels of one real frame: raw, P and every extra (dog, edge, erode, watershed), one z-slice, linear grey, nearest-neighbour.
Use it BEFORE a long run: the extras' constants (DoG size, erosion size, watershed thresholds) are guesses and this shows whether a channel is sensible."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .config import Config
from .data import io
from .data.norm import norm_raw, sample_scale
from .features.builders import EXTRAS
from .features.filters import p_pipeline
from .infer import get_device


def channels_png(cfg: Config, video: str, t: int, out_png, z: int | None = None, half_xy: int = 64):
    data_dir = str(Path(cfg.data.root) / cfg.data.subdir)
    arr = io.open_video(data_dir, video)
    sc = sample_scale(arr)
    dev = get_device()
    R = torch.from_numpy(norm_raw(np.asarray(arr[t], np.float32), sc))[None, None].to(dev)
    P = p_pipeline(R, cfg.features)
    maps = [("raw (normalised)", R), ("P = raw -> balance -> dehaze -> floor", P)] + [(e, EXTRAS[e](P, cfg.features)) for e in EXTRAS]
    z = R.shape[2] // 2 if z is None else z
    Y, X = R.shape[3:]
    y0, x0 = max(0, Y // 2 - half_xy), max(0, X // 2 - half_xy)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    for ax, (name, m) in zip(axes.ravel(), maps):
        im = m[0, 0, z, y0:y0 + 2 * half_xy, x0:x0 + 2 * half_xy].float().cpu().numpy()
        ax.imshow(im, cmap="gray", vmin=0, vmax=max(float(np.percentile(im, 99.9)), 1e-6), interpolation="nearest")
        ax.set_title(f"{name}\nmean {float(m.float().mean()):.3f}  p99.9 {float(np.percentile(im, 99.9)):.3f}", fontsize=12)
        ax.axis("off")
    fig.suptitle(f"{video}  t={t}  z={z}   (linear grey per panel)", fontsize=14)
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=80)
    plt.close(fig)
    print(f"wrote {out_png}")
