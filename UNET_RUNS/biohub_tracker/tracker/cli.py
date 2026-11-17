"""python -m tracker.cli run   --det-dir <run>/detections --gt-dir <data dir> --out-dir <dir> [--thr .. --c-div ..] [--score]
   python -m tracker.cli sweep --det-dir ... --gt-dir ... --out-dir ... --thr 0.2,0.3,0.4 --c-div 6,10,16 --c-birth 8,12
Scores only videos that have detections AND a .geff in --gt-dir. TUNE ON DEV ONLY."""
from __future__ import annotations

import argparse
import itertools
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

from .graphio import VARIANTS, track_video, write_geff
from .link import LinkConfig
from .score import score_dir


def run(det_dir, out_dir, cfg: LinkConfig, thr: float, names=None, feat_dir=None, feats=(), timing: dict | None = None):
    det_dir, out_dir = Path(det_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = names or sorted(p.stem for p in det_dir.glob("*.parquet"))
    for n in names:
        tm = {} if timing is not None else None
        nodes, edges = track_video(det_dir / f"{n}.parquet", thr, cfg, feat_path=(Path(feat_dir) / f"{n}.parquet") if feat_dir else None, feats=feats, timer=tm)
        w0 = time.perf_counter()
        write_geff(nodes, edges, out_dir / f"{n}.geff")
        if timing is not None:
            tm["write_s"] = time.perf_counter() - w0
            tm["track_s"] = tm["link_s"] + tm["assemble_s"]                 # the tracking itself; reading the detection file and writing the .geff are listed apart
            timing[n] = tm
    return names


def _floats(s):
    return [float(x) for x in s.split(",")]


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("run", "sweep"):
        p = sub.add_parser(c)
        p.add_argument("--det-dir", required=True); p.add_argument("--gt-dir", required=True); p.add_argument("--out-dir", required=True)
        p.add_argument("--no-division", action="store_true")
        p.add_argument("--variant", default="v0", choices=sorted(VARIANTS), help="v0 motion only | v1 +intensity | v2 +DoG | v3 +DoG+intensity | v4 +DoG+intensity+erosion")
        p.add_argument("--feat-dir", default=None, help="folder of <video>.parquet written by tracker.features (needed for v1-v4)")
        p.add_argument("--w-app", default=None)
        if c == "run":
            p.add_argument("--thr", type=float, default=0.3); p.add_argument("--c-div", type=float, default=11.0); p.add_argument("--c-birth", type=float, default=12.0)
            p.add_argument("--score", action="store_true"); p.add_argument("--verbose", action="store_true")
        else:
            p.add_argument("--thr", default="0.2,0.3,0.4"); p.add_argument("--c-div", default="8,10,11,11.5"); p.add_argument("--c-birth", default="12")
    a = ap.parse_args(argv)
    base = LinkConfig(allow_division=not a.no_division)
    feats = VARIANTS[a.variant]
    if feats and not a.feat_dir:
        ap.error(f"--variant {a.variant} needs --feat-dir")
    wa = [float(x) for x in a.w_app.split(",")] if a.w_app else [1.0]
    out = Path(a.out_dir)
    if a.cmd == "run":
        cfg = replace(base, c_div=a.c_div, c_birth=a.c_birth, c_death=a.c_birth, w_app=wa[0])
        timing = {}
        t0 = time.time(); names = run(a.det_dir, out, cfg, a.thr, feat_dir=a.feat_dir, feats=feats, timing=timing); print(f"tracked {len(names)} videos in {time.time() - t0:.0f}s -> {out}")
        tot = {k: sum(v[k] for v in timing.values()) for k in ("read_s", "link_s", "assemble_s", "track_s", "write_s")}
        print(f"tracking {tot['track_s']:.1f}s total = {tot['track_s'] / max(1, len(names)):.2f} s/video  (reading detections {tot['read_s']:.1f}s and writing .geff {tot['write_s']:.1f}s are NOT included)")
        (out / "timing_track.json").write_text(json.dumps({"n_videos": len(names), "total": tot, "mean_track_s_per_video": tot["track_s"] / max(1, len(names)), "videos": timing}, indent=1))
        if a.score:
            s, _ = score_dir(out, a.gt_dir, verbose=a.verbose)
            print(json.dumps(s, indent=1)); (out / "score.json").write_text(json.dumps({**s, "cfg": asdict(cfg), "thr": a.thr, "variant": a.variant}, indent=1))
        return
    res = []
    for thr, cd, cb, w in itertools.product(_floats(a.thr), _floats(a.c_div), _floats(a.c_birth), wa if feats else [0.0]):
        cfg = replace(base, c_div=cd, c_birth=cb, c_death=cb, w_app=w)
        d = out / f"thr{thr}_div{cd}_birth{cb}_w{w}"
        run(a.det_dir, d, cfg, thr, feat_dir=a.feat_dir, feats=feats)
        s, _ = score_dir(d, a.gt_dir)
        res.append(dict(variant=a.variant, thr=thr, c_div=cd, c_birth=cb, w_app=w, score=s["score"], adj_edge_j=s["adj_edge_jaccard"], edge_j=s["edge_jaccard"], div_j=s["division_jaccard"], n_videos=s["n_videos"]))
        print(res[-1], flush=True)
    df = pd.DataFrame(res).sort_values("score", ascending=False)
    df.to_csv(out / "sweep.csv", index=False)
    print("\n" + df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
