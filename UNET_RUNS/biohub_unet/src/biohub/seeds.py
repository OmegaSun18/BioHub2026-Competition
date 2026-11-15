"""Seed noise: the same variant trained with different seeds, scored the same way (validation videos, distance suppression dup_um).
Runs are grouped by the feature kind and seed stored in their checkpoint, so run folder names do not matter.
The spread inside one variant (max - min over seeds) is the yardstick for a difference BETWEEN variants: a gap smaller than that spread is not evidence of a better input."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data import io
from .data.split import split_videos
from .evaluate import sweep_dup
from .infer import get_device, load_checkpoint

METRICS = ["recall@0.8x", "recall@1.0x", "recall@1.2x", "recall_all", "peaks_per_est"]


def seeds(run_dirs, dup_um=3.0, split="val", out_csv="seeds.csv", ckpt="last.pt"):
    device = get_device()
    rows = []
    for rd in run_dirs:
        rd = Path(rd)
        model, cfg, builder = load_checkpoint(rd / "ckpt" / ckpt, device)
        data_dir = str(Path(cfg.data.root) / cfg.data.subdir)
        names = io.list_videos(data_dir)
        train_names, val_names = split_videos(names, cfg.data.n_val, cfg.data.split_seed)
        pool = {"val": val_names, "train": train_names, "all": names}[split]
        r = sweep_dup(model, builder, cfg, pool, data_dir, [dup_um], device).iloc[0].to_dict()
        r.update(variant=cfg.features.kind, seed=cfg.train.seed, run=str(rd))
        rows.append(r)
    runs = pd.DataFrame(rows).sort_values(["variant", "seed"]).reset_index(drop=True)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    runs.to_csv(out_csv, index=False)
    print(f"\nevery run, dup_um={dup_um}, {split} videos:")
    print(runs[["variant", "seed"] + METRICS].round(3).to_string(index=False))
    summ = []
    for v, g in runs.groupby("variant", sort=False):
        row = {"variant": v, "n_seeds": len(g)}
        for m in METRICS:
            row[m] = f"{g[m].mean():.3f} +- {g[m].std(ddof=1) if len(g) > 1 else float('nan'):.3f}"
            row[m + " range"] = round(float(g[m].max() - g[m].min()), 3)
        summ.append(row)
    summ = pd.DataFrame(summ)
    print("\nmean +- std over seeds (std needs >= 2 seeds; with 3 it is itself rough), and max-min range:")
    print(summ[["variant", "n_seeds"] + METRICS].to_string(index=False))
    print(summ[["variant"] + [m + " range" for m in METRICS]].to_string(index=False))
    summ.to_csv(Path(out_csv).with_name(Path(out_csv).stem + "_summary.csv"), index=False)
    print(f"wrote {out_csv} and {Path(out_csv).stem}_summary.csv")
    return runs, summ
