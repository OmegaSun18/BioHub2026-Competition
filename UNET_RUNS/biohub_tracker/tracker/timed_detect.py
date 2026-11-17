"""Detections of a finished model, with the time of every part measured separately.
Same results as `biohub detect` (same functions, same eval settings, same parquet columns incl. dup_frame), but with a stopwatch around:
  model_load_s   loading the checkpoint (NOT part of any per-video time)
  warmup_s       one untimed-in-the-stats forward pass on the first frame, so lazy one-time start-up (thread pools, kernels) is not charged to video 1
  per video:     dupcheck_s (finding repeated frames) | read_s (frames from disk + normalisation) | filter_s (the feature channels) | net_s (the network) | peaks_s (peak search + 3 um duplicate suppression)
  detect_s = dupcheck + read + filter + net + peaks of that video = the detection time of the video. Writing the parquet is timed apart (write_s) and not included.
   python -m tracker.timed_detect --run-dir <run with ckpt/last.pt + config.yaml> --data-dir <split dir> --out-dir <dir> [--frac 0.2 --frac-seed 8001] [--device cpu]"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

pc = time.perf_counter


def select_videos(names, frac, seed):
    """Exactly what `biohub ... --split all --frac F --frac-seed S` does."""
    if frac >= 1.0:
        return list(names)
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(names, max(1, round(frac * len(names))), replace=False).tolist())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--split", choices=["all", "val"], default="all", help="val = the held-out validation videos of the model's own train/val split (what `biohub ... --split val` uses); apply to DEV data")
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--frac-seed", type=int, default=8001)
    ap.add_argument("--max-videos", type=int, default=None, help="only the first N of the selected videos (for a quick timing)")
    ap.add_argument("--dup-um", type=float, default=3.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=None)
    a = ap.parse_args(argv)

    from biohub import config as C
    from biohub.constants import SCALE
    from biohub.data import io
    from biohub.data.dupframes import dup_flags
    from biohub.data.norm import sample_scale
    from biohub.infer import centroid_peaks, full_stack, load_checkpoint

    if a.threads:
        torch.set_num_threads(a.threads)
    dev = a.device
    sync = (lambda: torch.cuda.synchronize()) if dev.startswith("cuda") else (lambda: None)
    run, out = Path(a.run_dir), Path(a.out_dir)
    (out / "detections").mkdir(parents=True, exist_ok=True)
    rcfg = C.load_config(run / "config.yaml", [f"eval.dup_um={a.dup_um}"])
    ev = rcfg.eval

    load0 = os.getloadavg()[0]
    t0 = pc()
    model, cfg, builder = load_checkpoint(run / "ckpt" / "last.pt", dev)
    model_load_s = pc() - t0
    model.eval()
    names = io.list_videos(a.data_dir)
    if a.split == "val":
        from biohub.data.split import split_videos
        names = split_videos(names, rcfg.data.n_val, rcfg.data.split_seed)[1]
    names = select_videos(names, a.frac, a.frac_seed)
    if a.max_videos:
        names = names[:a.max_videos]
    reuse = builder.n_frames == 1

    def one_frame(arr, t, sc, tm):
        s0 = pc(); stack = full_stack(arr, t, sc, builder).to(dev); sync(); s1 = pc()
        f = builder(stack); sync(); s2 = pc()
        with torch.no_grad():
            heat = torch.sigmoid(model(f)[0, 0].float())
        sync(); s3 = pc()
        idx, sc_ = centroid_peaks(heat, ev.floor, ev.nms, ev.dup_um); s4 = pc()
        if tm is not None:
            tm["read_s"] += s1 - s0; tm["filter_s"] += s2 - s1; tm["net_s"] += s3 - s2; tm["peaks_s"] += s4 - s3
        return idx, sc_

    arr0 = io.open_video(a.data_dir, names[0])
    w0 = pc(); one_frame(arr0, 0, sample_scale(arr0), None); warmup_s = pc() - w0      # result discarded

    vids = []
    for n in names:
        arr = io.open_video(a.data_dir, n)
        tm = dict(read_s=0.0, filter_s=0.0, net_s=0.0, peaks_s=0.0)
        d0 = pc(); sc = sample_scale(arr); flags = dup_flags(arr); dupcheck_s = pc() - d0
        rows, last, computed = [], None, 0
        for t in range(len(flags)):
            if reuse and flags[t] and last is not None:
                idx, s = last
            else:
                idx, s = one_frame(arr, t, sc, tm); computed += 1
            last = (idx, s)
            um = idx * SCALE
            rows.append(pd.DataFrame({"t": t, "z": idx[:, 0], "y": idx[:, 1], "x": idx[:, 2], "z_um": um[:, 0], "y_um": um[:, 1], "x_um": um[:, 2], "score": s, "dup_frame": flags[t]}))
        w0 = pc(); pd.concat(rows, ignore_index=True).to_parquet(out / "detections" / f"{n}.parquet"); write_s = pc() - w0
        detect_s = dupcheck_s + sum(tm.values())
        vids.append(dict(video=n, frames=len(flags), dup_frames=int(sum(flags)), frames_computed=computed, dupcheck_s=dupcheck_s, **tm, detect_s=detect_s, write_s=write_s,
                         detections=int(sum(len(r) for r in rows))))
        print(f"  {n}: {detect_s:.1f}s detect ({detect_s / max(1, len(flags)):.2f} s/frame)  read {tm['read_s']:.1f} filter {tm['filter_s']:.1f} net {tm['net_s']:.1f} peaks {tm['peaks_s']:.1f}", flush=True)
    tot = {k: float(sum(v[k] for v in vids)) for k in ("dupcheck_s", "read_s", "filter_s", "net_s", "peaks_s", "detect_s", "write_s")}
    summ = dict(run_dir=str(run), device=dev, torch_threads=torch.get_num_threads(), cpu_count=os.cpu_count(), loadavg_start=load0, loadavg_end=os.getloadavg()[0],
                model_load_s=model_load_s, warmup_s=warmup_s, n_videos=len(vids), total=tot, mean_detect_s_per_video=tot["detect_s"] / len(vids),
                mean_detect_s_per_frame=tot["detect_s"] / max(1, sum(v["frames"] for v in vids)), videos=vids, frac=a.frac, frac_seed=a.frac_seed, split=a.split)
    (out / "timing_detect.json").write_text(json.dumps(summ, indent=1))
    print(f"model load {model_load_s:.1f}s (not counted) | warm-up {warmup_s:.1f}s (not counted) | detection {tot['detect_s']:.0f}s for {len(vids)} videos = {summ['mean_detect_s_per_video']:.1f} s/video, {summ['mean_detect_s_per_frame']:.2f} s/frame")


if __name__ == "__main__":
    main()
