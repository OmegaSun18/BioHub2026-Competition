"""biohub train | evaluate | detect | track | run

  biohub train    --config configs/raw.yaml                       train + evaluate on the validation videos
  biohub evaluate --run-dir runs/raw/<stamp> [--split val|train]  re-evaluate a finished run
  biohub detect   --run-dir runs/raw/<stamp>                      write detections/<video>.parquet for every frame
  biohub sweep    --run-dirs runs/combos/raw runs/combos/temporal_diff --dup-um 0 2 3 4 5 6 8   choose the duplicate distance with data
  biohub seeds    --run-dirs runs/combos/raw runs/seeds/raw_s8002 ...   mean / std / range over seeds per variant
  biohub grid     --hours 6                                   base input x extras x model size, unattended, resumable
  biohub dupframes --data-dir /kaggle/working/split_dataset/test    frames that are bit-identical to the previous one, per video
  biohub channels --video 6bba_ab78413d --t 10 --set data.root=...   look at every input channel of one frame
  biohub run      --config c.yaml --stages train,evaluate,detect  any chain of stages; completed stages are skipped on a re-run
  biohub diagnose --run-dirs runs/combos/raw runs/combos/p_only --out output.png    same sampled cells (default_rng(8001)) drawn for every variant
Any config value can be overridden:  --set train.steps=500 --set data.root=/data/dev
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from . import config as C
from .data import io
from .data.split import split_videos
from .pipeline import Context, Pipeline


def _common(p):
    p.add_argument("--config", help="YAML config (omit with --run-dir to reuse the run's own config.yaml)")
    p.add_argument("--run-dir", help="existing run directory (default for train: runs/<name>/<timestamp>)")
    p.add_argument("--set", action="append", default=[], metavar="a.b=c", help="override a config value")
    p.add_argument("--checkpoint", help="use this checkpoint instead of <run-dir>/ckpt/last.pt")
    p.add_argument("--force", action="store_true", help="redo stages that are already done")
    p.add_argument("--frac", type=float, default=1.0, help="work on this random fraction of the videos (whole videos, every frame; the same videos for every model)")
    p.add_argument("--frac-seed", type=int, default=8001, help="np.random.default_rng(seed) draws the videos of --frac")


def _context(a, default_stage_names) -> tuple[Context, list[str]]:
    run_dir = Path(a.run_dir) if a.run_dir else None
    cfg_path = a.config or (run_dir / "config.yaml" if run_dir and (run_dir / "config.yaml").exists() else None)
    cfg = C.load_config(cfg_path, a.set)
    if run_dir is None:
        run_dir = Path(cfg.out_dir) / cfg.name / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    C.save_config(cfg, run_dir / "config.yaml")
    data_dir = str(Path(cfg.data.root) / cfg.data.subdir)
    names = io.list_videos(data_dir)
    if not names:
        raise FileNotFoundError(f"no *.zarr videos in {data_dir}")
    train_names, val_names = split_videos(names, cfg.data.n_val, cfg.data.split_seed)
    split = getattr(a, "split", "val")
    chosen = {"val": val_names, "train": train_names, "all": names}[split]
    if a.frac < 1.0:
        rng = np.random.default_rng(a.frac_seed)
        chosen = sorted(rng.choice(chosen, max(1, round(a.frac * len(chosen))), replace=False).tolist())
        print(f"using {len(chosen)} of the videos ({a.frac:.0%}, seed {a.frac_seed})")
    ov = {"checkpoint": a.checkpoint} if a.checkpoint else {}
    ctx = Context(cfg, run_dir, data_dir, chosen, ov,
                  {"train_names": train_names, "val_names": chosen, "tag": split, "max_frames": getattr(a, "max_frames", None)})
    print(f"run dir: {run_dir}")
    return ctx, default_stage_names


def main(argv=None):
    ap = argparse.ArgumentParser(prog="biohub", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train", help="train, then evaluate on the validation videos")
    _common(p)
    p = sub.add_parser("evaluate", help="evaluate a checkpoint")
    _common(p)
    p.add_argument("--split", choices=["val", "train", "all"], default="val")
    p = sub.add_parser("detect", help="write detections for every frame")
    _common(p)
    p.add_argument("--split", choices=["val", "train", "all"], default="val")
    p.add_argument("--max-frames", type=int, default=None)
    p = sub.add_parser("track", help="link detections (not implemented yet)")
    _common(p)
    p = sub.add_parser("diagnose", help="draw several trained variants on the same randomly sampled annotated cells -> output.png + output.csv")
    p.add_argument("--run-dirs", nargs="+", required=True, help="run directories of the variants to compare (each needs ckpt/last.pt)")
    p.add_argument("--out", default="output.png")
    p.add_argument("--split", choices=["val", "train", "all"], default="val")
    p.add_argument("--n-samples", type=int, default=8)
    p.add_argument("--seed", type=int, default=8001, help="np.random.default_rng(seed) draws the videos, frames and cells")
    p.add_argument("--ckpt", default="last.pt")
    p.add_argument("--dup-um", type=float, default=None, help="distance suppression in um (default: eval.dup_um of the first run)")
    p = sub.add_parser("sweep", help="recall / peaks-per-estimate of finished runs at several duplicate-suppression distances (no retraining)")
    p.add_argument("--run-dirs", nargs="+", required=True)
    p.add_argument("--dup-um", nargs="+", type=float, default=[0, 2, 3, 4, 5, 6, 8])
    p.add_argument("--split", choices=["val", "train", "all"], default="val")
    p.add_argument("--out", default="dup_sweep.csv")
    p.add_argument("--ckpt", default="last.pt")
    p = sub.add_parser("seeds", help="score runs of the same variants trained with different seeds; mean, std and range per variant")
    p.add_argument("--run-dirs", nargs="+", required=True)
    p.add_argument("--dup-um", type=float, default=3.0)
    p.add_argument("--split", choices=["val", "train", "all"], default="val")
    p.add_argument("--out", default="seeds.csv")
    p.add_argument("--ckpt", default="last.pt")
    p = sub.add_parser("channels", help="draw every input channel (raw, P, dog, edge, erode, watershed) of one real frame")
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--video", required=True)
    p.add_argument("--t", type=int, default=10)
    p.add_argument("--z", type=int, default=None)
    p.add_argument("--out", default="channels.png")
    p = sub.add_parser("grid", help="run base input x extra channels x model size, time-budgeted, resumable; ranks the runs")
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--root", default="runs/grid")
    p.add_argument("--data-root", default="/kaggle/working/split_dataset")
    p.add_argument("--subdir", default="dev")
    p.add_argument("--hours", type=float, default=6.0, help="stop starting new runs after this many hours")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--dup-um", type=float, default=3.0)
    p.add_argument("--top-k", type=int, default=4, help="how many of the best small runs get medium and large versions in round 5")
    p.add_argument("--run-timeout-min", type=float, default=120.0)
    p.add_argument("--bases", nargs="+", default=None)
    p.add_argument("--extras", nargs="+", default=None, help="subset of: none dog edge erode watershed dog+edge+erode")
    p.add_argument("--sizes", nargs="+", default=None)
    p.add_argument("--dry-run", action="store_true", help="print the plan and the parameter counts, run nothing")
    p.add_argument("--set", action="append", default=[], metavar="a.b=c", help="extra config override passed to every run")
    p.add_argument("--inprocess", action="store_true", help="run each training inside this process (tests); default is one subprocess per run")
    p = sub.add_parser("dupframes", help="count the frames that are bit-identical to the previous frame, per video (reads every frame once)")
    p.add_argument("--data-dir", required=True, help="folder with the *.zarr videos, e.g. /kaggle/working/split_dataset/test")
    p.add_argument("--out", default="dup_frames.csv")
    p = sub.add_parser("leaderboard", help="rank finished runs of a grid folder")
    p.add_argument("--root", default="runs/grid")
    p.add_argument("--top", type=int, default=30)
    p = sub.add_parser("run", help="run a chain of stages")
    _common(p)
    p.add_argument("--stages", required=True, help="comma separated, e.g. train,evaluate,detect")
    p.add_argument("--split", choices=["val", "train", "all"], default="val")
    p.add_argument("--max-frames", type=int, default=None)
    a = ap.parse_args(argv)

    if a.cmd == "diagnose":
        from .diagnose import diagnose
        diagnose(a.run_dirs, a.out, a.split, a.n_samples, a.seed, ckpt=a.ckpt, dup_um=a.dup_um)
        return
    if a.cmd == "channels":
        from .channels import channels_png
        channels_png(C.load_config(a.config, a.set), a.video, a.t, a.out, a.z)
        return
    if a.cmd == "grid":
        from . import grid
        a.bases, a.extras, a.sizes = a.bases or grid.BASES, a.extras or list(grid.EXTRA_SETS), a.sizes or grid.SIZES
        grid.run_grid(a)
        return
    if a.cmd == "dupframes":
        import pandas as pd
        from .data.dupframes import dup_flags
        rows = []
        for n in io.list_videos(a.data_dir):
            f = dup_flags(io.open_video(a.data_dir, n))
            rows.append({"video": n, "frames": len(f), "n_dup": sum(f), "dup_frames": " ".join(str(t) for t, x in enumerate(f) if x)})
        df = pd.DataFrame(rows)
        df.to_csv(a.out, index=False)
        bad = df[df.n_dup > 0]
        print(f"{len(df)} videos, {len(bad)} with repeated frames, {int(df.n_dup.sum())} repeated frames of {int(df.frames.sum())} ({df.n_dup.sum() / max(df.frames.sum(), 1):.2%})")
        print(bad.groupby(bad.video.str[:4]).agg(videos=("video", "count"), repeated=("n_dup", "sum")).to_string() if len(bad) else "no repeated frames")
        print(f"table: {a.out}")
        return
    if a.cmd == "leaderboard":
        from .grid import leaderboard
        print(leaderboard(a.root, a.top, Path(a.root) / "leaderboard.csv").to_string(index=False))
        return
    if a.cmd == "seeds":
        from .seeds import seeds
        seeds(a.run_dirs, a.dup_um, a.split, a.out, a.ckpt)
        return
    if a.cmd == "sweep":
        from .sweep import sweep
        sweep(a.run_dirs, a.dup_um, a.split, a.out, a.ckpt)
        return
    stages = {"train": ["train", "evaluate"], "evaluate": ["evaluate"], "detect": ["detect"], "track": ["track"]}.get(a.cmd) or a.stages.split(",")
    ctx, stages = _context(a, stages)
    Pipeline(stages).run(ctx, force=a.force)


if __name__ == "__main__":
    main()
