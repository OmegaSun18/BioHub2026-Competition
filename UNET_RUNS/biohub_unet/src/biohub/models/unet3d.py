from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelCfg, resolve_model
from ..registry import MODELS


def _norm(kind, c):
    return nn.GroupNorm(8, c) if kind == "group" else nn.BatchNorm3d(c)


def _blk(cin, cout, norm="group", dropout=0.0):
    layers = [nn.Conv3d(cin, cout, 3, padding=1, bias=False), _norm(norm, cout), nn.GELU(),
              nn.Conv3d(cout, cout, 3, padding=1, bias=False), _norm(norm, cout), nn.GELU()]
    if dropout > 0:
        layers.append(nn.Dropout3d(dropout))              # channel dropout; absent when 0 so the layer names of the small model are unchanged
    return nn.Sequential(*layers)


@MODELS.register("unet3d")
class CentroidNet(nn.Module):
    """Fully 3D U-Net. Output: one logit channel, the centre heat, same size as the input."""

    def __init__(self, in_ch, ch=(16, 32, 64, 128, 128), pools=((1, 2, 2), (1, 2, 2), (2, 2, 2), (2, 2, 2)), norm="group", dropout=0.0):
        super().__init__()
        self.pools = [tuple(p) for p in pools]
        self.enc = nn.ModuleList([_blk(in_ch if i == 0 else ch[i - 1], ch[i], norm, dropout) for i in range(len(ch))])
        self.dec = nn.ModuleList([_blk(ch[i + 1] + ch[i], ch[i], norm, dropout) for i in range(len(ch) - 1)])
        self.head = nn.Conv3d(ch[0], 1, 1)
        with torch.no_grad():
            self.head.bias.fill_(-4.0)                                  # heat starts ~empty

    def forward(self, x):
        skips = []
        for i, e in enumerate(self.enc):
            x = e(x)
            if i < len(self.pools):
                skips.append(x)
                x = F.max_pool3d(x, self.pools[i])
        for i in reversed(range(len(self.dec))):
            x = F.interpolate(x, size=skips[i].shape[2:], mode="trilinear", align_corners=False)
            x = self.dec[i](torch.cat([x, skips[i]], 1))
        return self.head(x)


def build_model(mc: ModelCfg, in_ch: int) -> nn.Module:
    r = resolve_model(mc)
    return MODELS.get(mc.kind)(in_ch, tuple(r["ch"]), tuple(tuple(p) for p in mc.pools), r["norm"], r["dropout"])
