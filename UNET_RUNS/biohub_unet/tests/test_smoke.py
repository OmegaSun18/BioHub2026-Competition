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
    for kind, ch, nf in (("raw_only", 1, 1), ("p_only", 1, 1), ("raw", 2, 1), ("temporal_stack", 4, 3), ("temporal_diff", 4, 3)):
        b = build_features(C.FeatureCfg(kind=kind))
        assert (b.out_ch, b.n_frames) == (ch, nf)


def test_neighbour_clamp():
    assert io.neighbour_ts(0, 5, 1, 1) == [0, 0, 1] and io.neighbour_ts(4, 5, 1, 1) == [3, 4, 4]


@pytest.mark.parametrize("kind", ["raw_only", "p_only", "raw", "temporal_stack", "temporal_diff"])
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


def test_diagnose_same_samples_and_png(data, tmp_path):
    runs = []
    for kind in ("raw_only", "temporal_stack"):
        run = tmp_path / "combos" / kind
        main(["train", "--run-dir", str(run), "--set", f"data.root={data}", "--set", "data.n_val=2", "--set", "data.patch=[8,32,32]", "--set", "data.margin=[4,16,16]",
              "--set", "train.steps=4", "--set", "train.bs=2", "--set", "train.workers=0", "--set", "train.amp=off", "--set", "train.count_start=2",
              "--set", "train.count_every=2", "--set", "train.ckpt_every=4", "--set", f"features.kind={kind}", "--set", "eval.frames_per_video=2"])
        runs.append(str(run))
    out = tmp_path / "combos" / "output.png"
    main(["diagnose", "--run-dirs", *runs, "--out", str(out), "--n-samples", "3"])
    from biohub.diagnose import diagnose
    small = tmp_path / "combos" / "small.png"
    diagnose(runs, small, "val", 3, 8001, half_xy=16)                            # crop much smaller than the 96 px frame: points outside must not stretch the axes
    from PIL import Image
    a, b = Image.open(out).size, Image.open(small).size
    assert a == b
    assert out.exists() and out.stat().st_size > 10_000 and out.with_suffix(".csv").exists()
    import pandas as pd
    df = pd.read_csv(out.with_suffix(".csv"))
    a, b = (df[df.variant == v][["video", "t", "cell_z", "cell_y", "cell_x"]].reset_index(drop=True) for v in ("raw_only", "temporal_stack"))
    assert a.equals(b) and len(a) == 3                      # every variant is drawn on the same sampled cells
    from biohub.diagnose import sample_cells
    n = sorted(__import__("biohub").data.io.list_videos(data / "train"))
    assert sample_cells(n, str(data / "train"), 3, 8001) == sample_cells(n, str(data / "train"), 3, 8001)   # seed 8001 is deterministic


def test_dataset_reproducible(data):
    from biohub.data.dataset import CentroidDataset
    n = io.list_videos(data / "train")
    mk = lambda: CentroidDataset(n, data / "train", 4, (8, 32, 32), (4, 16, 16), 2.0, 0, 0, 0.7, 8001)
    a, b = mk()[0], mk()[0]
    assert np.array_equal(a[0].numpy(), b[0].numpy()) and np.array_equal(a[1].numpy(), b[1].numpy())


def test_dedupe_distance_in_um():
    from biohub.infer import dedupe_peaks
    idx = np.array([[10, 50, 50], [11, 50, 50], [10, 50, 53], [10, 80, 80]])            # voxels: (1 z-slice = 1.625 um), (3 x-voxels = 1.22 um)
    s = np.array([0.5, 0.9, 0.4, 0.8], np.float32)
    k, ks = dedupe_peaks(idx, s, 2.5)           # B(0.9) is 1.63 um from A and 2.03 um from C (z and x add up): both go; D is far
    assert sorted(np.round(ks.astype(float), 2).tolist()) == [0.8, 0.9]
    assert len(dedupe_peaks(idx, s, 2.0)[0]) == 3                                          # at 2.0 um C (2.03 um from B) survives
    assert len(dedupe_peaks(idx, s, 0.0)[0]) == 4                                          # 0 = off
    assert len(dedupe_peaks(idx, s, 1.0)[0]) == 4                                          # nothing is closer than 1 um
    assert len(dedupe_peaks(idx, s, 20.0)[0]) == 1                                         # D is 17.3 um from B, so at 20 um only the strongest peak is left


def test_sweep(data, tmp_path):
    run = tmp_path / "combos" / "raw"
    main(["train", "--run-dir", str(run), "--set", f"data.root={data}", "--set", "data.n_val=2", "--set", "data.patch=[8,32,32]", "--set", "data.margin=[4,16,16]",
          "--set", "train.steps=4", "--set", "train.bs=2", "--set", "train.workers=0", "--set", "train.amp=off", "--set", "train.count_start=2",
          "--set", "train.count_every=2", "--set", "train.ckpt_every=4", "--set", "eval.frames_per_video=2"])
    out = tmp_path / "sweep.csv"
    main(["sweep", "--run-dirs", str(run), "--dup-um", "0", "3", "6", "--out", str(out)])
    import pandas as pd
    df = pd.read_csv(out)
    assert list(df.dup_um) == [0, 3, 6] and df.peaks_per_est.is_monotonic_decreasing      # more suppression never adds peaks


def test_seeds_table(data, tmp_path):
    runs = []
    for seed in (8001, 8002):
        run = tmp_path / f"s{seed}"
        main(["train", "--run-dir", str(run), "--set", f"data.root={data}", "--set", "data.n_val=2", "--set", "data.patch=[8,32,32]", "--set", "data.margin=[4,16,16]",
              "--set", "train.steps=4", "--set", "train.bs=2", "--set", "train.workers=0", "--set", "train.amp=off", "--set", "train.count_start=2",
              "--set", "train.count_every=2", "--set", "train.ckpt_every=4", "--set", f"train.seed={seed}", "--set", "eval.frames_per_video=2"])
        runs.append(str(run))
    out = tmp_path / "seeds.csv"
    main(["seeds", "--run-dirs", *runs, "--dup-um", "3", "--out", str(out)])
    import pandas as pd
    df = pd.read_csv(out)
    assert sorted(df.seed) == [8001, 8002] and set(df.variant) == {"raw"}
    assert (tmp_path / "seeds_summary.csv").exists()


def test_extras_and_sizes_train(data, tmp_path):
    """medium (BatchNorm + dropout) with every extra channel trains, saves, reloads and evaluates; the count loss runs (eval-mode forward)"""
    run = tmp_path / "medium_extras"
    main(["train", "--run-dir", str(run), "--set", f"data.root={data}", "--set", "data.n_val=1", "--set", "data.patch=[8,32,32]", "--set", "data.margin=[4,16,16]",
          "--set", "train.steps=6", "--set", "train.bs=2", "--set", "train.workers=0", "--set", "train.amp=off", "--set", "train.count_start=2", "--set", "train.count_every=2",
          "--set", "train.ckpt_every=6", "--set", "features.kind=temporal_full", "--set", "features.extras=[dog,edge,erode,watershed]", "--set", "model.size=medium",
          "--set", "eval.frames_per_video=2"])
    from biohub.infer import load_checkpoint
    m, cfg, b = load_checkpoint(run / "ckpt" / "last.pt", "cpu")
    assert b.out_ch == 10 and cfg.model.size == "medium" and sum(p.numel() for p in m.parameters()) > 10e6
    assert any(type(x).__name__ == "BatchNorm3d" for x in m.modules()) and any(type(x).__name__ == "Dropout3d" for x in m.modules())
    assert (run / "eval_val.json").exists()


def test_old_checkpoint_config_still_builds():
    """a checkpoint config saved before size/norm/dropout existed (explicit ch list) must rebuild the same small model"""
    from biohub import config as C
    from biohub.models import build_model
    cfg = C.from_dict({"model": {"kind": "unet3d", "ch": [16, 32, 64, 128, 128], "pools": [[1, 2, 2], [1, 2, 2], [2, 2, 2], [2, 2, 2]]}})
    m = build_model(cfg.model, 2)
    assert abs(sum(p.numel() for p in m.parameters()) / 1e6 - 3.67) < 0.01
    assert not any(type(x).__name__ in ("BatchNorm3d", "Dropout3d") for x in m.modules())


def test_grid_plan_and_run(data, tmp_path, capsys):
    from biohub import grid
    head, rest = grid.plan()
    assert len(head) + len(rest) == 6 * 6 * 3 and len(set(head + rest)) == 108 and head[0].name == "raw_only+none@small"
    out = tmp_path / "grid"
    common = ["--root", str(out), "--data-root", str(data), "--subdir", "train", "--steps", "4", "--hours", "5", "--inprocess", "--bases", "raw", "--extras", "none", "dog",
              "--sizes", "small", "medium", "--top-k", "1", "--config", "configs/base.yaml",
              "--set", "data.n_val=2", "--set", "data.patch=[8,32,32]", "--set", "data.margin=[4,16,16]", "--set", "train.bs=2", "--set", "train.workers=0",
              "--set", "train.amp=off", "--set", "train.count_start=2", "--set", "train.count_every=2", "--set", "train.ckpt_every=4", "--set", "eval.frames_per_video=2"]
    main(["grid", *common])
    import pandas as pd
    lb = pd.read_csv(out / "leaderboard.csv")
    assert set(lb.name) == {"raw+none@small", "raw+dog@small", "raw+none@medium", "raw+dog@medium"}           # fixed part + round 5 (top-1 small x medium)
    assert (lb.dup_um == 3.0).all()
    main(["grid", *common])                                                                                  # second call: everything is skipped
    main(["leaderboard", "--root", str(out)])


def test_channels_png(data, tmp_path):
    out = tmp_path / "ch.png"
    main(["channels", "--config", "configs/base.yaml", "--set", f"data.root={data}", "--set", "data.subdir=train", "--video", "6bba_00000000", "--t", "1", "--out", str(out)])
    assert out.stat().st_size > 10_000


def test_evaluate_detect_copy_on_other_tree(data, tmp_path, monkeypatch):
    """the run_test.sh flow: a finished run is COPIED (ckpt + config), then detected on another data tree (no labels needed) (subdir changed); the source run is untouched"""
    import shutil
    shutil.copytree(data / "train", data / "test")
    src, dst = tmp_path / "src", tmp_path / "dst"
    common = ["--set", f"data.root={data}", "--set", "data.n_val=1", "--set", "data.patch=[8,32,32]", "--set", "data.margin=[4,16,16]", "--set", "train.steps=4",
              "--set", "train.bs=2", "--set", "train.workers=0", "--set", "train.amp=off", "--set", "train.count_start=2", "--set", "train.count_every=2",
              "--set", "train.log=2", "--set", "eval.frames_per_video=2", "--set", "features.kind=p_only", "--set", "features.extras=[dog]"]
    main(["train", "--run-dir", str(src), *common])
    cfg_before = (src / "config.yaml").read_text()
    (dst / "ckpt").mkdir(parents=True)
    shutil.copy(src / "ckpt" / "last.pt", dst / "ckpt" / "last.pt")
    shutil.copy(src / "config.yaml", dst / "config.yaml")
    main(["run", "--run-dir", str(dst), "--stages", "detect", "--split", "all", "--set", f"data.root={data}", "--set", "data.subdir=test", "--set", "eval.dup_um=3.0"])
    assert len(list((dst / "detections").glob("*.parquet"))) == 4 and not (dst / "eval_all.json").exists()     # detect needs no labels
    assert "subdir: test" in (dst / "config.yaml").read_text()
    assert (src / "config.yaml").read_text() == cfg_before and not (src / "eval_all.json").exists()
    # a video without annotated cells (empty .geff) is skipped by evaluate, not a crash; detect still covers it
    orig = io.load_gt
    monkeypatch.setattr(io, "load_gt", lambda p: {} if "6bba_00000000" in str(p) else orig(p))
    main(["run", "--run-dir", str(dst), "--stages", "evaluate", "--split", "all", "--set", f"data.root={data}", "--set", "data.subdir=test", "--set", "eval.dup_um=3.0"])
    import json
    assert json.loads((dst / "eval_all.json").read_text())["n_videos"] == 3
    # --frac: a seeded subset of whole videos, the same for every call
    dst2 = tmp_path / "dst2"
    (dst2 / "ckpt").mkdir(parents=True)
    shutil.copy(src / "ckpt" / "last.pt", dst2 / "ckpt" / "last.pt")
    shutil.copy(src / "config.yaml", dst2 / "config.yaml")
    main(["run", "--run-dir", str(dst2), "--stages", "detect", "--split", "all", "--frac", "0.5", "--set", f"data.root={data}", "--set", "data.subdir=test"])
    got = sorted(p.stem for p in (dst2 / "detections").glob("*.parquet"))
    assert len(got) == 2
    import numpy as _np
    names = sorted(p.stem for p in (data / "test").glob("*.zarr"))
    assert got == sorted(_np.random.default_rng(8001).choice(names, 2, replace=False).tolist())


def test_repeated_frames_flagged_and_reused(data, tmp_path):
    """a frame that is bit-identical to the previous one is flagged in the detections; a single-frame model reuses the previous peaks"""
    import pandas as pd
    from biohub.data.dupframes import dup_flags
    names = io.list_videos(data / "train")
    arr = zarr.open(str(data / "train" / f"{names[0]}.zarr"), mode="r+")["0"]
    arr[2] = arr[1]
    assert dup_flags(io.open_video(data / "train", names[0])) == [False, False, True, False, False]
    assert not any(dup_flags(io.open_video(data / "train", names[1])))
    run = tmp_path / "run"
    common = ["--run-dir", str(run), "--set", f"data.root={data}", "--set", "data.n_val=1", "--set", "data.patch=[8,32,32]", "--set", "data.margin=[4,16,16]",
              "--set", "train.steps=4", "--set", "train.bs=2", "--set", "train.workers=0", "--set", "train.amp=off", "--set", "train.count_start=2",
              "--set", "train.count_every=2", "--set", "train.log=2", "--set", "features.kind=p_only", "--set", "eval.frames_per_video=2"]
    main(["train", *common])
    main(["detect", *common, "--split", "all"])
    d = pd.read_parquet(run / "detections" / f"{names[0]}.parquet")
    assert d.groupby("t").dup_frame.first().tolist() == [False, False, True, False, False]
    a, b = d[d.t == 1], d[d.t == 2]
    assert len(a) == len(b) and (a[["z", "y", "x"]].to_numpy() == b[["z", "y", "x"]].to_numpy()).all() and (a.score.to_numpy() == b.score.to_numpy()).all()
    main(["dupframes", "--data-dir", str(data / "train"), "--out", str(tmp_path / "dup.csv")])
    t = pd.read_csv(tmp_path / "dup.csv")
    assert int(t.n_dup.sum()) == 1 and t[t.n_dup > 0].video.iloc[0] == names[0]
