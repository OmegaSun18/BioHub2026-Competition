from __future__ import annotations

import numpy as np


def split_videos(names, n_val: int = 20, seed: int = 7):
    """Random split of the videos that exist in the DEV data: n_val held out for validation, the rest for training.
    There is no test split here: the test half of the time axis is a separate, untouched data tree that the tuning code never opens."""
    rng = np.random.default_rng(seed)
    perm = [names[i] for i in rng.permutation(len(names))]
    val, train = sorted(perm[:n_val]), sorted(perm[n_val:])
    cnt = lambda L: {c: sum(n.startswith(c + "_") for n in L) for c in sorted({n.split("_")[0] for n in names})}
    print(f"train {len(train)} videos {cnt(train)} | val {len(val)} videos {cnt(val)}")
    return train, val
