"""Typed config. One YAML per run (optionally `inherit: base.yaml`), `--set a.b=c` overrides from the command line.
The fully resolved config is written next to every run's outputs, and stored inside every checkpoint."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict, fields, is_dataclass
from pathlib import Path

import yaml


@dataclass
class DataCfg:
    root: str = "/data"             # folder that contains `subdir/` with <video>.zarr and <video>.geff
    subdir: str = "train"
    n_val: int = 20                 # videos held out for validation (random, fixed seed); the rest train
    split_seed: int = 7
    patch: list = field(default_factory=lambda: [16, 64, 64])      # training patch (z, y, x), ~26 um cube
    margin: list = field(default_factory=lambda: [12, 48, 48])     # extra context so balance / dehaze are valid at the patch border
    sigma_um: float = 2.0           # Gaussian blob painted on each annotated centre


@dataclass
class FeatureCfg:
    kind: str = "raw"               # base input: raw_only | p_only | raw | temporal_stack | temporal_diff | temporal_full
    extras: list = field(default_factory=list)   # extra channels computed from P of frame t: dog | edge | erode | watershed
    dog_um: float = 2.5             # DoG radius (sigma = um/1.414, second sigma 1.6x), x dog_gain
    dog_gain: float = 4.0
    edge_smooth_um: float = 0.5     # 3D gradient magnitude of P, x edge_gain
    edge_gain: float = 2.0
    erode_um: float = 2.0           # grey erosion of P with a box of +-erode_um (per axis, in voxels of that axis): thin bridges between touching cells vanish
    ws_down: int = 2                # watershed on an xy-down-sampled copy (the boundary map is up-sampled again): ~5x cheaper
    ws_smooth_um: float = 1.6
    ws_min_dist: int = 3            # min distance (voxels of the down-sampled grid) between watershed markers
    ws_thr: float = 0.05            # markers need smoothed P above this
    floor: float = 0.02             # P = (dehaze(balance(raw)) - floor).clamp(0)
    bal_eps: float = 0.05
    bal_max: float = 4.0
    bal_gamma: float = 0.7
    bal_loc_um: float = 6.0
    cell_um: float = 12.0
    smooth_um: float = 4.0
    down: int = 4


SIZES = {   # width presets. medium / large use BatchNorm + channel dropout.   [dropout values are guesses, untested]
    "small": dict(ch=[16, 32, 64, 128, 128], norm="group", dropout=0.0),
    "medium": dict(ch=[32, 64, 128, 256, 256], norm="batch", dropout=0.1),
    "large": dict(ch=[48, 96, 192, 384, 384], norm="batch", dropout=0.2),
}


@dataclass
class ModelCfg:
    kind: str = "unet3d"
    size: str = "small"             # small | medium | large (see SIZES); ch / norm / dropout below override the preset when set
    ch: list | None = None
    norm: str | None = None         # group | batch
    dropout: float | None = None
    pools: list = field(default_factory=lambda: [[1, 2, 2], [1, 2, 2], [2, 2, 2], [2, 2, 2]])


@dataclass
class TrainCfg:
    steps: int = 2000
    bs: int = 8
    lr: float = 1e-3
    workers: int = 4
    amp: str = "bf16"               # fp16 | bf16 | off   (bf16 needs no loss scaling and avoids the early non-finite steps of fp16)
    log: int = 50
    ckpt_every: int = 1000
    max_bad: int = 30
    p_pos: float = 0.7              # share of patches centred around an annotated cell
    w_bg: float = 0.5               # weight of background voxels in the heat loss
    seed: int = 8001                # seeds torch, numpy, the patch sampler of every loader worker and the count-loss frame sampler
    count_w: float = 1.0            # count loss (band penalty). 0 for count_w and total_w switches it off
    total_w: float = 1.0
    frame_band: list = field(default_factory=lambda: [0.75, 1.25])
    total_band: list = field(default_factory=lambda: [0.9, 1.1])
    erode: list = field(default_factory=lambda: [1, 5, 5])
    ema_a: float = 0.1
    count_start: int = 300
    count_every: int = 4


@dataclass
class EvalCfg:
    floor: float = 0.02
    nms: list = field(default_factory=lambda: [3, 7, 7])
    radius_um: float = 7.0
    dup_um: float = 0.0             # drop a peak that lies within dup_um of a stronger one (0 = off). Choose it with `biohub sweep`, not by guessing
    frames_per_video: int = 6
    fracs: list = field(default_factory=lambda: [0.8, 1.0, 1.2])


@dataclass
class Config:
    name: str = "run"
    data: DataCfg = field(default_factory=DataCfg)
    features: FeatureCfg = field(default_factory=FeatureCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)
    out_dir: str = "runs"           # runs/<name>/<timestamp>/


def resolve_model(mc: ModelCfg) -> dict:
    """-> {'ch', 'norm', 'dropout'} with the preset of mc.size filled in wherever the config leaves a value unset"""
    if mc.size not in SIZES:
        raise KeyError(f"unknown model size '{mc.size}'. Available: {sorted(SIZES)}")
    d = dict(SIZES[mc.size])
    for k in ("ch", "norm", "dropout"):
        if getattr(mc, k) is not None:
            d[k] = getattr(mc, k)
    return d


def _merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _build(cls, d: dict):
    kw = {}
    known = {f.name: f for f in fields(cls)}
    for k, v in d.items():
        if k not in known:
            raise KeyError(f"unknown config key '{k}' for {cls.__name__} (valid: {sorted(known)})")
        t = known[k].type
        sub = {"DataCfg": DataCfg, "FeatureCfg": FeatureCfg, "ModelCfg": ModelCfg, "TrainCfg": TrainCfg, "EvalCfg": EvalCfg}.get(t if isinstance(t, str) else getattr(t, "__name__", ""))
        kw[k] = _build(sub, v) if sub is not None and isinstance(v, dict) else v
    return cls(**kw)


def _read_yaml(path: Path) -> dict:
    d = yaml.safe_load(path.read_text()) or {}
    parent = d.pop("inherit", None)
    return _merge(_read_yaml((path.parent / parent).resolve()), d) if parent else d


def _coerce(s: str):
    return yaml.safe_load(s)        # "3" -> 3, "0.5" -> 0.5, "[1,2]" -> [1, 2], "raw" -> "raw", "true" -> True


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    d = _read_yaml(Path(path)) if path else {}
    for item in overrides or []:
        key, _, val = item.partition("=")
        if not _:
            raise ValueError(f"override '{item}' must look like section.key=value")
        cur = d
        parts = key.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = _coerce(val)
    return _build(Config, d)


def to_dict(cfg: Config) -> dict:
    return asdict(cfg)


def from_dict(d: dict) -> Config:
    return _build(Config, d)


def save_config(cfg: Config, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(to_dict(cfg), sort_keys=False))
