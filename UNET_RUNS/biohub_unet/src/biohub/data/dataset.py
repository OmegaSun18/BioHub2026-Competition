from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from . import io
from .norm import norm_raw, sample_scale
from .targets import make_heat


class CentroidDataset(Dataset):
    """Random patches: (F, z, y, x) normalised frames t-before .. t+after (F = 1 for the raw model) + the centre-heat target of frame t.
    One brightness gain and one flip are applied to the whole stack, so the frames of a stack stay consistent with each other."""

    def __init__(self, names, data_dir, length, patch, margin, sigma_um, before=0, after=0, p_pos=0.7):
        self.names, self.dir, self.length, self.p_pos = list(names), str(data_dir), length, p_pos
        self.patch, self.margin, self.sigma_um, self.before, self.after = tuple(patch), tuple(margin), sigma_um, before, after
        self.gt = {n: io.load_gt(Path(data_dir) / f"{n}.geff") for n in self.names}
        self.index = [(n, t, k) for n in self.names for t, p in self.gt[n].items() for k in range(len(p))]
        self._arr, self._scale, self.rng = {}, {}, None

    def __len__(self):
        return self.length

    def arr(self, n):
        if n not in self._arr:
            self._arr[n] = io.open_video(self.dir, n)
            self._scale[n] = sample_scale(self._arr[n])
        return self._arr[n]

    def scale(self, n):
        self.arr(n)
        return self._scale[n]

    def __getitem__(self, i):
        if self.rng is None:
            wi = torch.utils.data.get_worker_info()
            self.rng = np.random.default_rng(None if wi is None else wi.seed % 2**32)
        r, P = self.rng, self.patch
        if r.random() < self.p_pos and self.index:
            n, t, k = self.index[r.integers(len(self.index))]
            c = self.gt[n][t][k]
            o = np.array([c[a] - r.integers(2, P[a] - 2) for a in range(3)])
        else:
            n = self.names[r.integers(len(self.names))]
            t = int(r.integers(self.arr(n).shape[0]))
            o = None
        arr = self.arr(n)
        T = arr.shape[0]
        hi = np.array(arr.shape[1:]) - np.array(P)
        o = r.integers(0, hi + 1) if o is None else np.clip(o, 0, hi)
        mg = np.array(self.margin)
        crop = tuple(p + 2 * m for p, m in zip(P, mg))
        ts = io.neighbour_ts(t, T, self.before, self.after)
        stack = io.read_stack_crop(arr, ts, o - mg, crop)                                  # (F, Z, Y, X)
        stack = norm_raw(stack, self._scale[n], r.uniform(0.8, 1.25))                      # percentile-normalise + one random gain for the stack
        heat = make_heat(self.gt[n].get(t, np.empty((0, 3), np.int32)) - o, P, self.sigma_um)
        if r.random() < 0.5:
            stack, heat = stack[:, :, ::-1], heat[:, ::-1]
        if r.random() < 0.5:
            stack, heat = stack[..., ::-1], heat[..., ::-1]
        return torch.from_numpy(stack.copy()), torch.from_numpy(heat.copy())[None]
