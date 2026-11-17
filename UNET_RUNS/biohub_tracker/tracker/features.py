"""Appearance features at every detection, for tracker variants v1-v4. Needs the biohub package (imports its filters, so the numbers are the very channels the models saw).
For every video: reads the frames, computes P (balance -> dehaze -> floor) and the DoG / erosion channels with the model's own config, and samples them at the detected voxel.
Output: <out-dir>/<video>.parquet with the SAME rows (same order) as detections/<video>.parquet and columns  raw, P, dog, erode.
   python -m tracker.features --run-dir <run> --data-dir <split dir> [--device cuda]"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def extract_video(arr, det: pd.DataFrame, fc, device: str, builder) -> pd.DataFrame:
    from biohub.data.norm import norm_raw, sample_scale
    from biohub.features.filters import dog_t, erode_t
    from biohub.infer import full_stack
    sc = sample_scale(arr)
    out = {k: np.zeros(len(det), np.float32) for k in ("raw", "P", "dog", "erode")}
    for t, rows in det.groupby("t").indices.items():
        if len(rows) == 0:
            continue
        with torch.no_grad():
            stack = full_stack(arr, int(t), sc, builder).to(device)
            R = builder.centre(stack)
            P = builder.p(stack)
            D, E = dog_t(P, fc.dog_um, fc.dog_gain), erode_t(P, fc.erode_um)
        z, y, x = (torch.as_tensor(det.iloc[rows][k].to_numpy(), device=device, dtype=torch.long) for k in ("z", "y", "x"))
        for name, vol in (("raw", R), ("P", P), ("dog", D), ("erode", E)):
            out[name][rows] = vol[0, 0][z, y, x].float().cpu().numpy()
    return pd.DataFrame(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--det-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    from biohub import config as C
    from biohub.data import io
    from biohub.features.builders import build_features
    run = Path(a.run_dir)
    ck = torch.load(run / "ckpt" / "last.pt", map_location="cpu", weights_only=False)
    fc = C.from_dict(ck["cfg"]).features
    builder = build_features(fc)
    det_dir, out_dir = Path(a.det_dir or run / "detections"), Path(a.out_dir or run / "features")
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in sorted(det_dir.glob("*.parquet")):
        if (out_dir / p.name).exists():
            continue
        det = pd.read_parquet(p)
        extract_video(io.open_video(a.data_dir, p.stem), det, fc, a.device, builder).to_parquet(out_dir / p.name)
        print(f"  {p.stem}: {len(det)} detections", flush=True)


if __name__ == "__main__":
    main()
