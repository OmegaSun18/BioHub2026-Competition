"""Detection-side stages: train, evaluate, detect."""
from __future__ import annotations

import pandas as pd

from ..constants import SCALE
from ..data import io
from ..data.dupframes import dup_flags
from ..data.norm import sample_scale
from ..evaluate import evaluate, save_eval, summarise
from ..infer import centroid_peaks, get_device, load_checkpoint, predict_heat
from ..registry import STAGES
from ..train import train
from .stage import Context, Stage


@STAGES.register("train")
class TrainStage(Stage):
    name, consumes, produces = "train", (), ("checkpoint",)

    def run(self, ctx: Context) -> None:
        train(ctx.cfg, ctx.options["train_names"], ctx.data_dir, ctx.run_dir)


@STAGES.register("evaluate")
class EvaluateStage(Stage):
    name, consumes, produces = "evaluate", ("checkpoint",), ("metrics",)

    def run(self, ctx: Context) -> None:
        model, cfg, builder = load_checkpoint(ctx.path("checkpoint"))
        cfg.eval = ctx.cfg.eval                                       # eval settings come from this run's config / --set, not from training time
        df = evaluate(model, builder, cfg, ctx.options["val_names"], ctx.data_dir)
        print(df.to_string(index=False))
        s = summarise(df)
        print(s)
        save_eval(df, s, ctx.run_dir, ctx.options.get("tag", "val"))


@STAGES.register("detect")
class DetectStage(Stage):
    """Every peak above eval.floor of every frame -> detections/<video>.parquet. This file is the hand-over to tracking."""
    name, consumes, produces = "detect", ("checkpoint",), ("detections",)

    def run(self, ctx: Context) -> None:
        model, cfg, builder = load_checkpoint(ctx.path("checkpoint"))
        cfg.eval = ctx.cfg.eval
        device, out = get_device(), ctx.path("detections")
        out.mkdir(parents=True, exist_ok=True)
        max_frames = ctx.options.get("max_frames")
        for n in ctx.names:
            arr = io.open_video(ctx.data_dir, n)
            sc = sample_scale(arr)
            flags = dup_flags(arr, max_frames)          # frame t bit-identical to frame t-1 (the acquisition repeated it)
            reuse = builder.n_frames == 1               # a single-frame model gives the same heat for an identical frame: skip the network, reuse the peaks
            rows, last = [], None
            for t in range(len(flags)):
                if reuse and flags[t] and last is not None:
                    idx, s = last
                else:
                    idx, s = centroid_peaks(predict_heat(model, builder, arr, t, sc, device, cfg.train.amp), cfg.eval.floor, cfg.eval.nms, cfg.eval.dup_um)
                last = (idx, s)
                um = idx * SCALE
                rows.append(pd.DataFrame({"t": t, "z": idx[:, 0], "y": idx[:, 1], "x": idx[:, 2], "z_um": um[:, 0], "y_um": um[:, 1], "x_um": um[:, 2], "score": s,
                                          "dup_frame": flags[t]}))
            pd.concat(rows, ignore_index=True).to_parquet(out / f"{n}.parquet")
            print(f"  {n}: {sum(len(r) for r in rows)} detections in {len(rows)} frames" + (f", {sum(flags)} repeated frame(s)" if any(flags) else ""))
