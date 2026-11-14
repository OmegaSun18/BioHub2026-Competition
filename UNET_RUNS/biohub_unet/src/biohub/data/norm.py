"""Per-video intensity normalisation: one (lo, span) per video, shared by every frame of that video and by every frame of a temporal stack
(so a difference between two frames is a real change, not a change of scale)."""
from __future__ import annotations

import numpy as np


def sample_scale(arr) -> tuple[float, float]:
    """(lo, span): 0.1th and 99.9th percentile of 3 subsampled frames (first, middle, last)."""
    T = arr.shape[0]
    v = np.concatenate([np.asarray(arr[t][::2, ::4, ::4], np.float32).ravel() for t in (0, T // 2, T - 1)])
    lo, hi = np.percentile(v, [0.1, 99.9])
    return float(lo), float(hi - lo) + 1e-6


def norm_raw(x, sc, gain: float = 1.0):
    """linear map of the video's 0.1..99.9 percentile range to 0..1, negatives clipped to 0, upper clip 4 (numpy or torch)."""
    y = (x - sc[0]) / sc[1] * gain
    return y.clamp(0, 4.0) if hasattr(y, "clamp") else np.clip(y, 0, 4.0)
