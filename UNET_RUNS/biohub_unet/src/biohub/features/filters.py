"""Differentiable-free, fixed-constant image filters (same result for a training crop and a whole frame)."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from ..config import FeatureCfg
from ..constants import SCALE


def _g1d(sigma, trunc=2.5):
    r = max(1, int(math.ceil(trunc * sigma)))
    x = torch.arange(-r, r + 1, dtype=torch.float32)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def gauss_s(x, sigma_um, scale):
    """separable 3D Gaussian, sigma in um, x (N,1,Z,Y,X), scale = um per voxel of x"""
    for ax in range(3):
        sg = float(sigma_um / scale[ax])
        if sg < 0.3:
            continue
        k = _g1d(sg).to(x)
        r = len(k) // 2
        pad = [0] * 6
        pad[(2 - ax) * 2] = pad[(2 - ax) * 2 + 1] = r
        shp = [1] * 5
        shp[2 + ax] = -1
        x = F.conv3d(F.pad(x, pad, mode="replicate"), k.view(shp))
    return x


def dehaze_t(x, cell_um=12.0, smooth_um=4.0, down=4):
    """subtract a slowly varying background (grey opening at cell size on a down-sampled copy)"""
    s = F.avg_pool3d(x, (1, down, down), ceil_mode=True)
    sc = SCALE * np.array([1, down, down], np.float32)
    k = [max(1, int(round(cell_um / sc[a]))) | 1 for a in range(3)]
    pad = tuple(v // 2 for v in k)
    bg = F.max_pool3d(-F.max_pool3d(-s, k, 1, pad), k, 1, pad)
    bg = F.interpolate(gauss_s(bg, smooth_um, sc), size=x.shape[2:], mode="trilinear", align_corners=False)
    return (x - bg).clamp(min=0)


def balance_t(I, fc: FeatureCfg):
    """intensity balancing: dim up, bright down = I / (local mean + eps), clipped, gamma"""
    s = F.avg_pool3d(I, (1, fc.down, fc.down), ceil_mode=True)
    sc = SCALE * np.array([1, fc.down, fc.down], np.float32)
    loc = F.interpolate(gauss_s(s, fc.bal_loc_um, sc), size=I.shape[2:], mode="trilinear", align_corners=False)
    return ((I / (loc + fc.bal_eps)).clamp(0, fc.bal_max) / fc.bal_max) ** fc.bal_gamma


def p_pipeline(R, fc: FeatureCfg):
    """raw (per-video normalised, (N,1,Z,Y,X)) -> BALANCE -> DEHAZE -> FLOOR (subtract floor, clip at 0)"""
    return (dehaze_t(balance_t(R, fc), fc.cell_um, fc.smooth_um, fc.down) - fc.floor).clamp(min=0)
