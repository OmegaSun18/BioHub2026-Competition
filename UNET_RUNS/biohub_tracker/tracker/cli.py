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

from .graphio import track_video, write_geff
from .link import LinkConfig
from .score import score_dir


def run(det_dir, out_dir, cfg: LinkConfig, thr: float, names=None):
    det_dir, out_dir = Path(det_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = names or sorted(p.stem for p in det_dir.glob("*.parquet"))
    for n in names:
        nodes, edges = track_video(det_dir / f"{n}.parquet", thr, cfg)
        write_geff(nodes, edges, out_dir / f"{n}.geff")
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
        if c == "run":
            p.add_argument("--thr", type=float, default=0.3); p.add_argument("--c-div", type=float, default=11.0); p.add_argument("--c-birth", type=float, default=12.0)
            p.add_argument("--score", action="store_true"); p.add_argument("--verbose", action="store_true")
        else:
            p.add_argument("--thr", default="0.2,0.3,0.4"); p.add_argument("--c-div", default="8,10,11,11.5"); p.add_argument("--c-birth", default="12")
    a = ap.parse_args(argv)
    base = LinkConfig(allow_division=not a.no_division)
    out = Path(a.out_dir)
    if a.cmd == "run":
        cfg = replace(base, c_div=a.c_div, c_birth=a.c_birth, c_death=a.c_birth)
        t0 = time.time(); names = run(a.det_dir, out, cfg, a.thr); print(f"tracked {len(names)} videos in {time.time() - t0:.0f}s -> {out}")
        if a.score:
            s, _ = score_dir(out, a.gt_dir, verbose=a.verbose)
            print(json.dumps(s, indent=1)); (out / "score.json").write_text(json.dumps({**s, "cfg": asdict(cfg), "thr": a.thr}, indent=1))
        return
    res = []
    for thr, cd, cb in itertools.product(_floats(a.thr), _floats(a.c_div), _floats(a.c_birth)):
        cfg = replace(base, c_div=cd, c_birth=cb, c_death=cb)
        d = out / f"thr{thr}_div{cd}_birth{cb}"
        run(a.det_dir, d, cfg, thr)
        s, _ = score_dir(d, a.gt_dir)
        res.append(dict(thr=thr, c_div=cd, c_birth=cb, score=s["score"], adj_edge_j=s["adj_edge_jaccard"], edge_j=s["edge_jaccard"], div_j=s["division_jaccard"], n_videos=s["n_videos"]))
        print(res[-1], flush=True)
    df = pd.DataFrame(res).sort_values("score", ascending=False)
    df.to_csv(out / "sweep.csv", index=False)
    print("\n" + df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
