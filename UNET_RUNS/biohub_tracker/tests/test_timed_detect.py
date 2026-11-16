"""timed_detect must give the SAME detections as `biohub detect` (single-frame and temporal model, with a repeated frame) and a sane timing file. Needs biohub (BIOHUB_SRC=<biohub_unet>/src)."""
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if os.environ.get("BIOHUB_SRC"):
    sys.path.insert(0, os.environ["BIOHUB_SRC"])
torch = pytest.importorskip("torch")
pytest.importorskip("biohub")
import zarr                                                     # noqa: E402

from tracker.timed_detect import main as timed_main, select_videos   # noqa: E402


def make_ckpt(run: Path, kind: str, extras: str):
    from biohub import config as C
    from biohub.features.builders import build_features
    from biohub.models import build_model
    torch.manual_seed(0)
    cfg = C.load_config(None, [f"features.kind={kind}", f"features.extras=[{extras}]", "model.size=small"])
    model = build_model(cfg.model, build_features(cfg.features).out_ch)
    with torch.no_grad():
        model.head.bias.fill_(-1.5)
    (run / "ckpt").mkdir(parents=True)
    torch.save({"cfg": C.to_dict(cfg), "model": model.state_dict()}, run / "ckpt" / "last.pt")
    C.save_config(cfg, run / "config.yaml")


def make_videos(d: Path, n=3):
    d.mkdir(parents=True)
    rng = np.random.default_rng(1)
    for i in range(n):
        a = zarr.open_group(str(d / f"v{i}.zarr"), mode="w").create_array("0", shape=(4, 8, 40, 40), chunks=(1, 8, 40, 40), dtype="uint16")
        base = rng.integers(100, 2000, (8, 40, 40)).astype("uint16")
        for t in range(4):
            a[t] = base if (i == 1 and t == 1) else rng.integers(100, 2000, (8, 40, 40)).astype("uint16")      # video 1: frame 1 is a copy of ... see below
        if i == 1:
            a[1] = np.asarray(a[0])                                                                      # repeated frame


@pytest.mark.parametrize("kind,extras", [("raw_only", "erode"), ("temporal_diff", "erode")])
def test_same_detections_as_biohub_detect(tmp_path, kind, extras):
    from biohub.cli import main as cli_main
    make_videos(tmp_path / "root" / "dev")
    run_a, run_b = tmp_path / "a", tmp_path / "b"
    make_ckpt(run_a, kind, extras)
    shutil.copytree(run_a, run_b)
    cli_main(["detect", "--run-dir", str(run_a), "--split", "all", "--set", f"data.root={tmp_path / 'root'}", "--set", "data.subdir=dev", "--set", "eval.dup_um=3.0"])
    out = tmp_path / "timed"
    timed_main(["--run-dir", str(run_b), "--data-dir", str(tmp_path / "root" / "dev"), "--out-dir", str(out), "--device", "cpu"])
    for p in sorted((run_a / "detections").glob("*.parquet")):
        pd.testing.assert_frame_equal(pd.read_parquet(p), pd.read_parquet(out / "detections" / p.name), check_exact=False, rtol=1e-4, atol=1e-5)
    tm = json.loads((out / "timing_detect.json").read_text())
    assert tm["n_videos"] == 3 and tm["model_load_s"] > 0 and tm["warmup_s"] > 0
    for v in tm["videos"]:
        assert abs(v["detect_s"] - (v["dupcheck_s"] + v["read_s"] + v["filter_s"] + v["net_s"] + v["peaks_s"])) < 1e-9     # model load / warm-up / parquet writing are not in it
    assert tm["videos"][1]["dup_frames"] == 1
    if kind == "raw_only":
        assert tm["videos"][1]["frames_computed"] == 3          # the repeated frame reused the previous peaks, no network call


def test_select_videos_matches_cli_rule():
    names = [f"v{i:03d}" for i in range(199)]
    got = select_videos(names, 0.2, 8001)
    rng = np.random.default_rng(8001)
    assert got == sorted(rng.choice(names, max(1, round(0.2 * len(names))), replace=False).tolist()) and len(got) == 40
