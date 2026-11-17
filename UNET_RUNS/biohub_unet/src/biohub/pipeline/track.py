"""Tracking stage: the slot is here, the implementation is not.

Contract (so tracking can be developed and containerised independently of detection):
  input  : <run_dir>/detections/<video>.parquet   columns t, z, y, x, z_um, y_um, x_um, score
  output : <run_dir>/tracks/<video>.geff          nodes = detections kept, edges = links between consecutive frames (organisers' format)
  config : add a `track:` section to config.py and read it from ctx.cfg
"""
from __future__ import annotations

from ..registry import STAGES
from .stage import Context, Stage


@STAGES.register("track")
class TrackStage(Stage):
    name, consumes, produces = "track", ("detections",), ("tracks",)

    def run(self, ctx: Context) -> None:
        raise NotImplementedError("tracking is not implemented yet; see the contract in this file's docstring")
