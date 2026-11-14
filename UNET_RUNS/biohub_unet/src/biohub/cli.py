"""biohub train | evaluate | detect | track | run

  biohub train    --config configs/raw.yaml                       train + evaluate on the validation videos
  biohub evaluate --run-dir runs/raw/<stamp> [--split val|train]  re-evaluate a finished run
  biohub detect   --run-dir runs/raw/<stamp>                      write detections/<video>.parquet for every frame
  biohub run      --config c.yaml --stages train,evaluate,detect  any chain of stages; completed stages are skipped on a re-run
Any config value can be overridden:  --set train.steps=500 --set data.root=/data/dev
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

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
    p = sub.add_parser("run", help="run a chain of stages")
    _common(p)
    p.add_argument("--stages", required=True, help="comma separated, e.g. train,evaluate,detect")
    p.add_argument("--split", choices=["val", "train", "all"], default="val")
    p.add_argument("--max-frames", type=int, default=None)
    a = ap.parse_args(argv)

    stages = {"train": ["train", "evaluate"], "evaluate": ["evaluate"], "detect": ["detect"], "track": ["track"]}.get(a.cmd) or a.stages.split(",")
    ctx, stages = _context(a, stages)
    Pipeline(stages).run(ctx, force=a.force)


if __name__ == "__main__":
    main()
