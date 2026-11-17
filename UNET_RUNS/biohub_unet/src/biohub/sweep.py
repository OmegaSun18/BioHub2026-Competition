"""Duplicate-suppression sweep: for each finished run, the heat of every evaluation frame is computed once and the distance suppression is applied at each dup_um.
Pick the smallest distance where peaks_per_est drops without recall_all / recall@1.0x falling: too large a distance deletes real neighbours in dense clusters."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data import io
from .data.split import split_videos
from .evaluate import sweep_dup
from .infer import get_device, load_checkpoint


def sweep(run_dirs, dups, split="val", out_csv="dup_sweep.csv", ckpt="last.pt"):
    device = get_device()
    frames = []
    for rd in run_dirs:
        rd = Path(rd)
        model, cfg, builder = load_checkpoint(rd / "ckpt" / ckpt, device)
        data_dir = str(Path(cfg.data.root) / cfg.data.subdir)
        names = io.list_videos(data_dir)
        train_names, val_names = split_videos(names, cfg.data.n_val, cfg.data.split_seed)
        pool = {"val": val_names, "train": train_names, "all": names}[split]
        df = sweep_dup(model, builder, cfg, pool, data_dir, dups, device)
        df.insert(0, "variant", rd.name)
        frames.append(df.reset_index())
    out = pd.concat(frames, ignore_index=True)
    cols = ["variant", "dup_um", "recall@0.8x", "recall@1.0x", "recall@1.2x", "recall_all", "peaks_per_est", "recall@1.0x_pooled", "recall_all_pooled", "n_videos", "n_annotated"]
    out = out[[c for c in cols if c in out.columns]]
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(out.round(3).to_string(index=False))
    print(f"wrote {out_csv}")
    return out
