"""Converts a cached spots folder (<video>_spots.parquet with spot_id, frame, z, y, x [um], z_px, y_px, x_px, radius, quality - the format of dog_detector / log_detector)
into the tracker's detection format:  <out>/<detector>/detections/<video>.parquet  with  t, z, y, x (voxel, rounded), z_um, y_um, x_um, score (= quality), dup_frame (= False).
No detector output is changed; only columns are renamed/derived. Run:
   python3 scripts/convert_spots.py --src /mnt/data/biohub_working/cached_spots_dog_detector --out /mnt/data/biohub_working/split_dataset/biohub_runs/test_spots
The detector name is the folder name without the 'cached_spots_' prefix (e.g. dog_detector)."""
import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--sub", default="", help="extra folder level between the detector name and detections/, e.g. test_0.2 to match the Kaggle cache layout")
    a = ap.parse_args()
    src = Path(a.src)
    name = a.name or src.name.removeprefix("cached_spots_")
    dst = Path(a.out) / name / a.sub / "detections"
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted(src.rglob("*_spots.parquet"))
    assert files, f"no *_spots.parquet under {src}"
    q = []
    for f in files:
        d = pd.read_parquet(f)
        miss = {"frame", "z", "y", "x", "z_px", "y_px", "x_px", "quality"} - set(d.columns)
        assert not miss, f"{f.name}: missing columns {miss}"
        o = pd.DataFrame({"t": d["frame"].astype("int64"), "z": d["z_px"].round().astype("int64"), "y": d["y_px"].round().astype("int64"), "x": d["x_px"].round().astype("int64"),
                          "z_um": d["z"].astype(float), "y_um": d["y"].astype(float), "x_um": d["x"].astype(float), "score": d["quality"].astype(float), "dup_frame": False})
        o.to_parquet(dst / f.name.removesuffix("_spots.parquet").join(["", ".parquet"]))
        q.append(d["quality"])
    qq = pd.concat(q)
    print(f"{name}: {len(files)} videos -> {dst}   quality min {qq.min():.3g} median {qq.median():.3g} max {qq.max():.3g}   frames 0..{int(o['t'].max())} in the last file")


if __name__ == "__main__":
    main()
