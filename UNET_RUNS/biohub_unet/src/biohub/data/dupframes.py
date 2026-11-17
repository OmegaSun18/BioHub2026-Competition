"""Bit-identical consecutive frames: in some videos the acquisition repeated the previous frame, which breaks any motion model
(the next real frame looks like a jump of two time steps). This module only finds them; the detect stage writes the flag into every detection file."""
from __future__ import annotations

import numpy as np


def dup_flags(arr, max_frames: int | None = None) -> list[bool]:
    """flags[t] is True when frame t of the (T, Z, Y, X) array is bit-identical to frame t-1 (frame 0 is never flagged). Reads every frame once."""
    T = arr.shape[0] if max_frames is None else min(arr.shape[0], max_frames)
    flags, prev = [False] * T, None
    for t in range(T):
        cur = np.asarray(arr[t])
        flags[t] = prev is not None and cur.shape == prev.shape and bool(np.array_equal(cur, prev))
        prev = cur
    return flags
