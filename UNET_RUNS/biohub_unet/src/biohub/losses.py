from __future__ import annotations

import torch
import torch.nn.functional as F


def centroid_loss(logit, heat, w_bg: float = 0.5):
    """CenterNet focal loss. Annotated centres are the positives; EVERY other voxel is background (weight w_bg). No ignore zones."""
    p = torch.sigmoid(logit.float()).clamp(1e-6, 1 - 1e-6)
    pos = (heat >= 0.99).float()
    l_pos = -((1 - p) ** 2) * torch.log(p) * pos
    l_neg = -((1 - heat) ** 4) * (p ** 2) * torch.log(1 - p) * (1 - pos) * w_bg
    return (l_pos.sum() + l_neg.sum()) / pos.sum().clamp(min=1)


def soft_count(p, nms=(3, 7, 7), floor=0.05, tau=0.2, w=0.04):
    """Differentiable number of detections in a frame: every local maximum of the heat above `floor` counts sigmoid((p - tau)/w). p (Z,Y,X)."""
    mx = F.max_pool3d(p[None, None], nms, 1, tuple(k // 2 for k in nms))[0, 0]
    pk = ((p == mx) & (p > floor)).float()
    return (torch.sigmoid((p - tau) / w) * pk).sum()


def eroded(p, erode=(1, 5, 5)):
    """Grey-scale erosion of the heat (min filter): thin ridges vanish, compact blobs survive. Differentiable."""
    return -F.max_pool3d(-p[None, None], tuple(erode), 1, tuple(k // 2 for k in erode))[0, 0]


def band(r, lo, hi):
    """0 inside [lo, hi], grows linearly outside"""
    return F.relu(lo - r) + F.relu(r - hi)
