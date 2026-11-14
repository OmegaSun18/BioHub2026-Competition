"""Checkpoints, whole-frame prediction, peak picking."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import config as C
from .data import io
from .data.norm import norm_raw
from .features import build_features
from .models import build_model


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def amp_dtype(amp: str):
    return torch.bfloat16 if amp == "bf16" else torch.float16


def save_checkpoint(path, model, cfg: C.Config, step: int):
    torch.save({"model": model.state_dict(), "cfg": C.to_dict(cfg), "step": step}, path)


def load_checkpoint(path, device: str | None = None):
    """-> (model in eval mode, cfg stored in the checkpoint, feature builder). The architecture and the feature kind come from the checkpoint,
    so a model can never be run with the wrong inputs."""
    device = device or get_device()
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = C.from_dict(ck["cfg"])
    builder = build_features(cfg.features)
    model = build_model(cfg.model, builder.out_ch)
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), cfg, builder


def cut(f, m):
    """remove the margin that was only there so balance / dehaze are valid at the border"""
    return f[..., m[0]:f.shape[-3] - m[0], m[1]:f.shape[-2] - m[1], m[2]:f.shape[-1] - m[2]]


def full_stack(arr, t: int, sc, builder) -> torch.Tensor:
    """frames t-before .. t+after of the WHOLE volume, normalised -> (1, F, Z, Y, X)"""
    ts = io.neighbour_ts(t, arr.shape[0], builder.before, builder.after)
    return torch.from_numpy(norm_raw(io.read_stack_full(arr, ts), sc))[None]


@torch.no_grad()
def predict_heat(model, builder, arr, t: int, sc, device: str, amp: str = "bf16"):
    """frame t of one video -> heat probabilities (Z, Y, X)"""
    model.eval()
    f = builder(full_stack(arr, t, sc, builder).to(device))
    with torch.autocast(device, dtype=amp_dtype(amp), enabled=device == "cuda" and amp != "off"):
        out = model(f)
    return torch.sigmoid(out[0, 0].float())


def centroid_peaks(heat, floor: float = 0.1, nms=(3, 7, 7)):
    """local maxima of the heat (window nms voxels) above floor -> (voxel idx (N,3), scores (N,))"""
    mx = F.max_pool3d(heat[None, None], tuple(nms), 1, tuple(k // 2 for k in nms))[0, 0]
    idx = torch.nonzero((heat == mx) & (heat > floor))
    return idx.cpu().numpy(), heat[idx[:, 0], idx[:, 1], idx[:, 2]].cpu().numpy()
