from __future__ import annotations

import numpy as np

from ..constants import SCALE


def make_heat(pos, shape, sigma_um: float) -> np.ndarray:
    """Gaussian blob (peak exactly 1) on every annotated centre. pos in patch voxel coords, shape (Z, Y, X)."""
    heat = np.zeros(shape, np.float32)
    sig = sigma_um / SCALE
    ax = [np.arange(s, dtype=np.float32) for s in shape]
    for c in pos:
        if np.any(c < -3 * sig) or np.any(c > np.array(shape) + 3 * sig):
            continue
        heat = np.maximum(heat, np.exp(-0.5 * ((ax[0] - c[0]) / sig[0]) ** 2)[:, None, None]
                          * np.exp(-0.5 * ((ax[1] - c[1]) / sig[1]) ** 2)[None, :, None]
                          * np.exp(-0.5 * ((ax[2] - c[2]) / sig[2]) ** 2)[None, None, :])
    return heat
