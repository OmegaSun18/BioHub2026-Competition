"""Stage contract. Stages talk to each other ONLY through files inside the run directory, so any stage can run in its own process or container:

    train ──> checkpoint ──> detect ──> detections/<video>.parquet ──> track ──> tracks/<video>.geff
                  └──────> evaluate ──> eval_<split>.json

A stage declares what it `consumes` and `produces` (artifact kinds). The pipeline finds each artifact at its fixed place in the run directory
(ARTIFACTS), checks the inputs exist before the stage starts, and writes `.done/<stage>.json` when it finishes so a re-run skips completed stages.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..config import Config
from ..registry import STAGES

ARTIFACTS = {                         # artifact kind -> path inside the run directory
    "checkpoint": "ckpt/last.pt",
    "detections": "detections",       # one <video>.parquet per video: columns t, z, y, x (voxel idx), z_um, y_um, x_um, score
    "tracks": "tracks",               # one <video>.geff per video (organisers' format)
    "metrics": "eval_val.json",
}


@dataclass
class Context:
    cfg: Config
    run_dir: Path
    data_dir: str
    names: list                       # videos this run works on
    overrides: dict = field(default_factory=dict)   # artifact kind -> explicit path (e.g. a checkpoint trained in another run)
    options: dict = field(default_factory=dict)     # stage options from the command line

    def path(self, kind: str) -> Path:
        if kind in self.overrides:
            return Path(self.overrides[kind])
        if kind == "metrics":                                           # eval_<split>.json: val, train or all (the evaluate stage writes the file of the split it was given)
            return self.run_dir / f"eval_{self.options.get('tag', 'val')}.json"
        return self.run_dir / ARTIFACTS[kind]


class Stage:
    name = ""
    consumes: tuple = ()
    produces: tuple = ()

    def run(self, ctx: Context) -> None:
        raise NotImplementedError


class Pipeline:
    def __init__(self, stage_names: list[str]):
        self.stages = [STAGES.get(n)() for n in stage_names]

    def run(self, ctx: Context, force: bool = False) -> None:
        done_dir = ctx.run_dir / ".done"
        done_dir.mkdir(parents=True, exist_ok=True)
        for st in self.stages:
            marker = done_dir / f"{st.name}.json"
            if marker.exists() and not force and all(ctx.path(k).exists() for k in st.produces):
                print(f"[{st.name}] already done - skipped (use --force to redo)")
                continue
            missing = [k for k in st.consumes if not ctx.path(k).exists()]
            if missing:
                raise FileNotFoundError(f"stage '{st.name}' needs {missing} but {[str(ctx.path(k)) for k in missing]} do not exist")
            t0 = time.time()
            print(f"[{st.name}] start")
            st.run(ctx)
            gone = [k for k in st.produces if not ctx.path(k).exists()]
            if gone:
                raise RuntimeError(f"stage '{st.name}' finished but did not write {gone}")
            marker.write_text(json.dumps({"stage": st.name, "seconds": round(time.time() - t0, 1), "finished": time.strftime("%Y-%m-%d %H:%M:%S")}))
            print(f"[{st.name}] done in {time.time() - t0:.0f}s")
