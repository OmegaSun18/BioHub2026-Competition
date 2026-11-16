"""Needs the biohub package (set BIOHUB_SRC=<biohub_unet>/src if it is not installed). Skipped otherwise."""
import json
import os
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

from tracker.cli import main                                    # noqa: E402
from tracker.features import main as feat_main                  # noqa: E402
from test_synth import make_dataset                             # noqa: E402


def toy_video(path, T=3, shape=(12, 64, 64), peaks=((6, 20, 20), (6, 40, 44))):
    arr = zarr.open_group(str(path), mode="w").create_array("0", shape=(T, *shape), chunks=(1, *shape), dtype="uint16")
    zz, yy, xx = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
    for t in range(T):
        v = np.full(shape, 100.0)
        for k, (z, y, x) in enumerate(peaks):
            v += (1000 + 800 * k) * np.exp(-(((zz - z) * 1.625) ** 2 + ((yy - y) * 0.40625) ** 2 + ((xx - x) * 0.40625) ** 2) / (2 * 2.0 ** 2))
        arr[t] = v.astype("uint16")


def test_feature_extraction_rows_align_and_bright_cell_is_brighter(tmp_path):
    from biohub import config as C
    run = tmp_path / "run"; (run / "ckpt").mkdir(parents=True); (run / "detections").mkdir()
    cfg = C.load_config(None, ["features.kind=raw_only"])
    torch.save({"cfg": C.to_dict(cfg)}, run / "ckpt" / "last.pt")
    data = tmp_path / "data"; data.mkdir(); toy_video(data / "v.zarr")
    det = pd.DataFrame({"t": [0, 0, 1, 1, 2, 2], "z": [6, 6] * 3, "y": [20, 40] * 3, "x": [20, 44] * 3, "score": 0.9, "dup_frame": False})
    for k, s in (("z_um", 1.625), ("y_um", 0.40625), ("x_um", 0.40625)):
        det[k] = det[k[0]] * s
    det.to_parquet(run / "detections" / "v.parquet")
    feat_main(["--run-dir", str(run), "--data-dir", str(data), "--device", "cpu"])
    f = pd.read_parquet(run / "features" / "v.parquet")
    assert len(f) == len(det) and set(f.columns) == {"raw", "P", "dog", "erode"}
    assert (f.iloc[1::2]["raw"].to_numpy() > f.iloc[0::2]["raw"].to_numpy()).all()       # the second peak was drawn brighter
    assert np.isfinite(f.to_numpy()).all()


def test_variants_run_end_to_end_with_features(tmp_path):
    gt, dd = make_dataset(tmp_path)
    det = pd.read_parquet(dd / "vid.parquet")
    fd = tmp_path / "feat"; fd.mkdir()
    rng = np.random.default_rng(0)
    pd.DataFrame({k: rng.uniform(0.2, 1.0, len(det)) for k in ("raw", "P", "dog", "erode")}).to_parquet(fd / "vid.parquet")
    for v in ("v1", "v2", "v3", "v4"):
        out = tmp_path / f"o_{v}"
        main(["run", "--det-dir", str(dd), "--gt-dir", str(gt), "--out-dir", str(out), "--thr", "0.3", "--variant", v, "--feat-dir", str(fd), "--score"])
        assert json.loads((out / "score.json").read_text())["variant"] == v
