"""
StarDist-3D Volumetric Object Detector Module
=============================================

1. HOW TO IMPORT
----------------
    from detectors.stardist_3d_detector import detect_spots_stardist_3d

2. EXPECTED INPUT DATA & PREPROCESSING REQUIREMENTS
---------------------------------------------------
    - `volume` (`np.ndarray`):
        * Shape: 4D `(T, Z, Y, X)` or 5D `(T, C, Z, Y, X)` numeric array (e.g., `uint16`, `float32`).
        * Data Type / Preprocessing Requirement: **RAW fluorescence intensity volumes (or mildly
          denoised/dehazed intensity volumes).** Do NOT pass binary masks, label images, or
          zero-mean derivative images (LoG/DoG), as StarDist relies on positive nuclear intensity
          gradients and internally performs percentile normalization (`pmin`, `pmax`).
    - `scale_zyx` (`tuple[float, float, float]`):
        * Physical voxel spacing `(dz, dy, dx)` in micrometers (e.g., `(1.625, 0.40625, 0.40625)`).
    - `model_name` & `model_dir`:
        * Either provide a pretrained StarDist3D model name (e.g., `"3D_demo"`) with `model_dir=None`,
          or provide a custom/fine-tuned model directory (`model_dir="/path/to/models"`) and
          `model_name="my_stardist_3d"`.
    - Note: Because Fiji's `TrackMate-StarDist` plugin only supports 2D (+ time) images, this module
      executes native volumetric `StarDist3D` frame-by-frame in Python and exports centroids in the
      exact same standardized schema as `log_detector` and `dog_detector`.

3. OUTPUT SPECIFICATION
-----------------------
    Returns a `pandas.DataFrame` with columns:
        - `spot_id` (`int64`): Unique 0-indexed spot identifier.
        - `frame` (`int64`): Time frame index `t` in `[0, T - 1]`.
        - `z`, `y`, `x` (`float64`): Physical centroid coordinates in calibrated units.
        - `z_px`, `y_px`, `x_px` (`float64`): Sub-voxel centroid coordinates in array index space.
        - `radius` (`float64`): Equivalent spherical radius in physical units derived from segmented volume.
        - `quality` (`float64`): StarDist center probability score (`prob`) for the segmented object.
"""

from typing import Optional, Tuple
import numpy as np
import pandas as pd
from csbdeep.utils import normalize
from stardist.models import StarDist3D
from skimage.measure import regionprops_table


def detect_spots_stardist_3d(
    volume: np.ndarray,
    scale_zyx: Tuple[float, float, float] = (1.625, 0.40625, 0.40625),
    model_name: str = "3D_demo",
    model_dir: Optional[str] = None,
    target_channel: int = 1,
    prob_thresh: Optional[float] = None,
    nms_thresh: Optional[float] = None,
    norm_percentiles: Tuple[float, float] = (1.0, 99.8),
    n_tiles: Optional[Tuple[int, int, int]] = None,
    min_voxel_volume: int = 10,
) -> pd.DataFrame:
    """
    Run volumetric 3D StarDist object detection across all frames of a 4D/5D time-lapse.

    Parameters
    ----------
    volume : np.ndarray
        Raw (or lightly denoised) 4D `(T, Z, Y, X)` or 5D `(T, C, Z, Y, X)` intensity volume.
    scale_zyx : tuple of float
        Physical voxel dimensions `(dz, dy, dx)`.
    model_name : str
        Name of pretrained or fine-tuned StarDist3D model (default `"3D_demo"`).
    model_dir : str, optional
        Directory containing custom/fine-tuned StarDist3D model weights (`None` loads pretrained).
    target_channel : int
        1-indexed channel to extract if `volume` is 5D `(T, C, Z, Y, X)` (default `1`).
    prob_thresh : float, optional
        Object center probability threshold (`None` uses model's calibrated threshold).
    nms_thresh : float, optional
        Non-maximum suppression IoU threshold (`None` uses model's calibrated threshold).
    norm_percentiles : tuple of float
        Lower and upper percentiles `(pmin, pmax)` for per-frame intensity normalization.
    n_tiles : tuple of int, optional
        Tiling grid `(nz, ny, nx)` for large 3D volumes to avoid GPU/CPU OOM (e.g., `(1, 4, 4)`).
    min_voxel_volume : int
        Minimum voxel count required to retain a segmented 3D nucleus.

    Returns
    -------
    pd.DataFrame
        Standardized detected spots table.
    """
    if volume.ndim == 4:
        vol_tzyx = volume
    elif volume.ndim == 5:
        c_idx = int(target_channel) - 1
        if not (0 <= c_idx < volume.shape[1]):
            raise ValueError(
                f"target_channel={target_channel} out of bounds for volume with {volume.shape[1]} channels."
            )
        vol_tzyx = volume[:, c_idx]
    else:
        raise ValueError(
            f"Expected 4D (T, Z, Y, X) or 5D (T, C, Z, Y, X) array, got shape {volume.shape}."
        )

    if model_dir is not None:
        model = StarDist3D(None, name=model_name, basedir=model_dir)
    else:
        model = StarDist3D.from_pretrained(model_name)

    dz, dy, dx = scale_zyx
    voxel_vol = float(dz * dy * dx)
    pmin, pmax = norm_percentiles

    rows = []
    spot_counter = 0

    for t in range(vol_tzyx.shape[0]):
        frame_raw = vol_tzyx[t].astype(np.float32, copy=False)
        frame_norm = normalize(frame_raw, pmin, pmax, axis=(0, 1, 2))

        labels, details = model.predict_instances(
            frame_norm,
            prob_thresh=prob_thresh,
            nms_thresh=nms_thresh,
            n_tiles=n_tiles,
        )

        if labels.max() == 0:
            continue

        props = regionprops_table(
            labels,
            properties=("label", "centroid", "area"),
        )
        probs = np.asarray(details.get("prob", []), dtype=np.float64)

        for i, lbl in enumerate(props["label"]):
            vox_count = int(props["area"][i])
            if vox_count < min_voxel_volume:
                continue

            z_px = float(props["centroid-0"][i])
            y_px = float(props["centroid-1"][i])
            x_px = float(props["centroid-2"][i])

            # Equivalent sphere radius in physical units: r = ((3 * V) / (4 * pi))^(1/3)
            phys_vol = vox_count * voxel_vol
            eq_radius = float(((3.0 * phys_vol) / (4.0 * np.pi)) ** (1.0 / 3.0))
            quality = float(probs[lbl - 1]) if (lbl - 1) < len(probs) else 1.0

            rows.append(
                {
                    "spot_id": spot_counter,
                    "frame": int(t),
                    "z": z_px * dz,
                    "y": y_px * dy,
                    "x": x_px * dx,
                    "z_px": z_px,
                    "y_px": y_px,
                    "x_px": x_px,
                    "radius": eq_radius,
                    "quality": quality,
                }
            )
            spot_counter += 1

    columns = ["spot_id", "frame", "z", "y", "x", "z_px", "y_px", "x_px", "radius", "quality"]
    if not rows:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(rows, columns=columns)