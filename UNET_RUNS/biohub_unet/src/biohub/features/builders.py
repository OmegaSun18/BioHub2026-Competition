"""Feature builders: normalised frame stack (B, F, Z, Y, X) -> network input (B, C, Z, Y, X).

The stack always holds frames t-before .. t+after of the SAME video, already normalised with that video's scale.
A builder declares how many neighbours it needs (`before`, `after`) and how many channels it returns (`out_ch`);
the dataset, the trainer, inference and evaluation all read those two numbers from the builder, so adding a variant touches only this file.

  raw_only        [R_t]                                   single frame, no engineered feature (input ablation)
  p_only          [P_t]                                   the engineered feature alone (input ablation)
  raw             [R_t, P_t]                              single frame (the baseline)
  temporal_stack  [R_t-1, R_t, R_t+1, P_t]                neighbouring raw frames stacked as channels (early fusion)
  temporal_diff   [R_t, R_t - R_t-1, R_t+1 - R_t, P_t]    same information as temporal_stack, written as differences (experiment)
  temporal_full   [R_t-1, R_t, R_t+1, R_t - R_t-1, R_t+1 - R_t, P_t]   stack AND differences

Extras (config `features.extras`, any combination) append one channel each, all computed from P_t:
  dog        difference of Gaussians (blob detector)      edge       3D gradient magnitude
  erode      grey erosion (separates touching cells)      watershed  basin boundary lines of a marker watershed (CPU, slower)

P_t is always computed from frame t alone: raw -> balance -> dehaze -> floor.
"""
from __future__ import annotations

import torch

from ..config import FeatureCfg
from ..registry import FEATURES
from .filters import dog_t, edge_t, erode_t, p_pipeline, watershed_t


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


@FEATURES.register("raw_only")
class RawOnly(FeatureBuilder):
    before, after, out_ch = 0, 0, 1

    def __call__(self, stack):
        return self.centre(stack)


@FEATURES.register("p_only")
class POnly(FeatureBuilder):
    before, after, out_ch = 0, 0, 1

    def __call__(self, stack):
        return self.p(stack).float()


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


@FEATURES.register("temporal_full")
class TemporalFull(FeatureBuilder):
    before, after, out_ch = 1, 1, 6

    def __call__(self, stack):
        s = stack.float()
        prev, cur, nxt = s[:, 0:1], s[:, 1:2], s[:, 2:3]
        return torch.cat([s, cur - prev, nxt - cur, self.p(stack)], 1).float()


EXTRAS = {
    "dog": lambda P, fc: dog_t(P, fc.dog_um, fc.dog_gain),
    "edge": lambda P, fc: edge_t(P, fc.edge_smooth_um, fc.edge_gain),
    "erode": lambda P, fc: erode_t(P, fc.erode_um),
    "watershed": lambda P, fc: watershed_t(P, fc.ws_down, fc.ws_smooth_um, fc.ws_min_dist, fc.ws_thr),
}


class WithExtras(FeatureBuilder):
    """a base builder + one extra channel per name in `extras`, each computed from P of frame t"""

    def __init__(self, base: FeatureBuilder, extras):
        super().__init__(base.fc)
        self.base, self.extras = base, list(extras)
        self.before, self.after, self.out_ch = base.before, base.after, base.out_ch + len(self.extras)

    def __call__(self, stack):
        P = self.base.p(stack)
        return torch.cat([self.base(stack)] + [EXTRAS[e](P, self.fc).to(P.dtype) for e in self.extras], 1).float()


def build_features(fc: FeatureCfg) -> FeatureBuilder:
    base = FEATURES.get(fc.kind)(fc)
    bad = [e for e in fc.extras if e not in EXTRAS]
    if bad:
        raise KeyError(f"unknown extras {bad}. Available: {sorted(EXTRAS)}")
    return WithExtras(base, fc.extras) if fc.extras else base
