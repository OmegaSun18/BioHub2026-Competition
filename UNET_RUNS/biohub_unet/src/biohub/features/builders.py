"""Feature builders: normalised frame stack (B, F, Z, Y, X) -> network input (B, C, Z, Y, X).

The stack always holds frames t-before .. t+after of the SAME video, already normalised with that video's scale.
A builder declares how many neighbours it needs (`before`, `after`) and how many channels it returns (`out_ch`);
the dataset, the trainer, inference and evaluation all read those two numbers from the builder, so adding a variant touches only this file.

  raw             [R_t, P_t]                              single frame (the baseline)
  temporal_stack  [R_t-1, R_t, R_t+1, P_t]                neighbouring raw frames stacked as channels (early fusion)
  temporal_diff   [R_t, R_t - R_t-1, R_t+1 - R_t, P_t]    same information as temporal_stack, written as differences (experiment)

P_t is always computed from frame t alone: raw -> balance -> dehaze -> floor.
"""
from __future__ import annotations

import torch

from ..config import FeatureCfg
from ..registry import FEATURES
from .filters import p_pipeline


class FeatureBuilder:
    before = 0
    after = 0
    out_ch = 0

    def __init__(self, fc: FeatureCfg):
        self.fc = fc

    @property
    def n_frames(self) -> int:
        return self.before + 1 + self.after

    def centre(self, stack):
        return stack[:, self.before:self.before + 1].float()

    def p(self, stack):
        return p_pipeline(self.centre(stack), self.fc)

    def __call__(self, stack):
        raise NotImplementedError


@FEATURES.register("raw")
class RawP(FeatureBuilder):
    before, after, out_ch = 0, 0, 2

    def __call__(self, stack):
        return torch.cat([self.centre(stack), self.p(stack)], 1).float()


@FEATURES.register("temporal_stack")
class TemporalStack(FeatureBuilder):
    before, after, out_ch = 1, 1, 4

    def __call__(self, stack):
        return torch.cat([stack.float(), self.p(stack)], 1).float()


@FEATURES.register("temporal_diff")
class TemporalDiff(FeatureBuilder):
    before, after, out_ch = 1, 1, 4

    def __call__(self, stack):
        s = stack.float()
        prev, cur, nxt = s[:, 0:1], s[:, 1:2], s[:, 2:3]
        return torch.cat([cur, cur - prev, nxt - cur, self.p(stack)], 1).float()


def build_features(fc: FeatureCfg) -> FeatureBuilder:
    return FEATURES.get(fc.kind)(fc)
