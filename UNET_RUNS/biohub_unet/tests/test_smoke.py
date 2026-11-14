"""End-to-end smoke test on tiny synthetic videos (CPU, seconds): every feature kind trains, saves, reloads, evaluates and detects."""
import numpy as np
import pytest
import zarr

import biohub.data.io as io
from biohub import config as C
from biohub.cli import main
from biohub.features import build_features

T, SHAPE = 5, (24, 96, 96)


@pytest.fixture()
def data(tmp_path, monkeypatch):
    d = tmp_path / "data" / "train"
    d.mkdir(parents=True)
    rng = np.random.default_rng(0)
    gts = {}
    for i in range(4):
        n = f"6bba_{i:08x}"
        g = zarr.open_group(str(d / f"{n}.zarr"), mode="w")
        vol = rng.random((T, *SHAPE), dtype=np.float32) * 0.1
        pts = np.array([[12, 30, 30], [12, 60, 60], [8, 40, 70]])
        for t in range(T):
            for p in pts:
                vol[t, p[0] - 1:p[0] + 2, p[1] - 3:p[1] + 4, p[2] - 3:p[2] + 4] += 1.0
        g.create_array("0", data=vol)
        (d / f"{n}.geff").mkdir()
        gts[n] = {t: pts.astype(np.int32) for t in range(T)}
    monkeypatch.setattr(io, "load_gt", lambda p: gts[p.stem if hasattr(p, "stem") else str(p).split("/")[-1][:-5]])
    monkeypatch.setattr(io, "find_estimate", lambda p: 3.0 * T)
    return tmp_path / "data"


def test_builders_channels():
    for kind, ch, nf in (("raw", 2, 1), ("temporal_stack", 4, 3), ("temporal_diff", 4, 3)):
        b = build_features(C.FeatureCfg(kind=kind))
        assert (b.out_ch, b.n_frames) == (ch, nf)


def test_neighbour_clamp():
    assert io.neighbour_ts(0, 5, 1, 1) == [0, 0, 1] and io.neighbour_ts(4, 5, 1, 1) == [3, 4, 4]


@pytest.mark.parametrize("kind", ["raw", "temporal_stack", "temporal_diff"])
def test_train_eval_detect(data, tmp_path, kind):
    run = tmp_path / f"run_{kind}"
    common = ["--run-dir", str(run), "--set", f"data.root={data}", "--set", "data.n_val=1", "--set", "data.patch=[8,32,32]", "--set", "data.margin=[4,16,16]",
              "--set", "train.steps=6", "--set", "train.bs=2", "--set", "train.workers=0", "--set", "train.amp=off", "--set", "train.count_start=2",
              "--set", "train.count_every=2", "--set", "train.log=2", "--set", "train.ckpt_every=3", "--set", f"features.kind={kind}", "--set", "eval.frames_per_video=2"]
    main(["train", *common])
    assert (run / "ckpt" / "last.pt").exists() and (run / "eval_val.json").exists() and (run / "config.yaml").exists()
    main(["detect", *common, "--max-frames", "2"])
    assert list((run / "detections").glob("*.parquet"))
    main(["train", *common])        # second call: finished stages are skipped
