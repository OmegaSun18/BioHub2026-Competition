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


# ---------- extra channels, all computed from P of frame t ----------
def dog_t(P, um=2.5, gain=4.0):
    """difference of Gaussians (blob detector): sigma = um/1.414 and 1.6x that, in um"""
    s = um / 1.414
    return (gauss_s(P, s, SCALE) - gauss_s(P, 1.6 * s, SCALE)) * gain


def edge_t(x, smooth_um=0.5, gain=2.0):
    """3D gradient magnitude in intensity per um (z and xy comparable)"""
    v = gauss_s(x, smooth_um, SCALE)
    d = torch.tensor([-0.5, 0.0, 0.5])
    sm = torch.tensor([0.25, 0.5, 0.25])
    g2 = 0
    for ax in range(3):
        ks = [sm, sm, sm]
        ks[ax] = d
        k = (ks[0].view(3, 1, 1) * ks[1].view(1, 3, 1) * ks[2].view(1, 1, 3)).to(x)[None, None]
        g2 = g2 + (F.conv3d(F.pad(v, (1,) * 6, mode="replicate"), k) / float(SCALE[ax])) ** 2
    return torch.sqrt(g2 + 1e-8) * gain


def erode_t(P, um=2.0):
    """grey erosion (min filter) with a box of +-um per axis: compact cores survive, thin bridges between touching cells vanish"""
    k = [2 * max(1, int(round(um / float(SCALE[a])))) + 1 for a in range(3)]
    return -F.max_pool3d(-P, k, 1, [v // 2 for v in k])


def watershed_t(P, down=2, smooth_um=1.6, min_dist=3, thr=0.05):
    """Marker-controlled watershed on -smoothed P (markers = local maxima of smoothed P); output = 1 on the lines between basins, 0 elsewhere.
    Runs on the CPU (skimage) per sample, on an xy-down-sampled copy, and the boundary map is up-sampled again (nearest)."""
    from scipy import ndimage as ndi
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed
    B, _, Z, Y, X = P.shape
    out = np.zeros((B, 1, Z, Y, X), np.float32)
    sc = SCALE * np.array([1, down, down], np.float32)
    sig = [smooth_um / float(v) for v in sc]
    for b in range(B):
        p = P[b, 0, :, ::down, ::down].float().cpu().numpy()
        S = ndi.gaussian_filter(p, sig)
        pk = peak_local_max(S, min_distance=min_dist, threshold_abs=thr, exclude_border=False)
        if len(pk) < 2:
            continue
        m = np.zeros(S.shape, np.int32)
        m[tuple(pk.T)] = np.arange(1, len(pk) + 1)
        lab = watershed(-S, m, mask=S > 0.6 * thr)
        bd = np.zeros(S.shape, bool)
        for ax in range(3):
            d = np.diff(lab, axis=ax) != 0
            sl = [slice(None)] * 3
            sl[ax] = slice(0, -1)
            bd[tuple(sl)] |= d
            sl[ax] = slice(1, None)
            bd[tuple(sl)] |= d
        bd &= lab > 0
        up = np.repeat(np.repeat(bd, down, axis=1), down, axis=2)[:, :Y, :X]
        out[b, 0, :, :up.shape[1], :up.shape[2]] = up
    return torch.from_numpy(out).to(P.device)
