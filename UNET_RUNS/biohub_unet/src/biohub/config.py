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
    kind: str = "raw"               # registered feature builder: raw | temporal_stack | temporal_diff
    floor: float = 0.02             # P = (dehaze(balance(raw)) - floor).clamp(0)
    bal_eps: float = 0.05
    bal_max: float = 4.0
    bal_gamma: float = 0.7
    bal_loc_um: float = 6.0
    cell_um: float = 12.0
    smooth_um: float = 4.0
    down: int = 4


@dataclass
class ModelCfg:
    kind: str = "unet3d"
    ch: list = field(default_factory=lambda: [16, 32, 64, 128, 128])
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
    seed: int = 0
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
