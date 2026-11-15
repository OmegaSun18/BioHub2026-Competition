"""
TrackMate Laplacian of Gaussian (LoG) Cell Detector Module
==========================================================

1. HOW TO IMPORT
----------------
    from fiji_detectors import init_fiji_headless
    from fiji_detectors.log_detector import detect_spots_log

2. EXPECTED INPUT DATA & PREPROCESSING REQUIREMENTS
---------------------------------------------------
    - `volume` (`np.ndarray`):
        * Shape: 4D `(T, Z, Y, X)` or 5D `(T, C, Z, Y, X)` numeric array (e.g., `uint16`, `float32`).
        * Data Type / Preprocessing Requirement: **RAW or mildly denoised fluorescence intensity data.**
          Do NOT feed pre-filtered LoG or DoG edge maps into this module—TrackMate's
          `LogDetectorFactory` computes the Gaussian smoothing and Laplacian second-derivative
          filtering internally. Passing pre-filtered LoG/DoG images will cause double-derivative
          ringing and false satellite detections. Light background subtraction or dehazing on raw
          intensities is acceptable.
    - `scale_zyx` (`tuple[float, float, float]`):
        * Physical voxel spacing `(dz, dy, dx)` in micrometers (e.g., `(1.625, 0.40625, 0.40625)`).
    - `ij` (`imagej.ImageJ | None`):
        * Active headless ImageJ gateway from `init_fiji_headless()`. If `None`, initializes one.

3. OUTPUT SPECIFICATION
-----------------------
    Returns a `pandas.DataFrame` with columns:
        - `spot_id` (`int64`): Unique 0-indexed spot identifier.
        - `frame` (`int64`): Time frame index `t` in `[0, T - 1]`.
        - `z`, `y`, `x` (`float64`): Physical centroid coordinates in calibrated units.
        - `z_px`, `y_px`, `x_px` (`float64`): Sub-voxel centroid coordinates in array index space.
        - `radius` (`float64`): Spot radius in physical units.
        - `quality` (`float64`): TrackMate LoG filter response quality score.
"""

from typing import Any, Optional, Tuple
import numpy as np
import pandas as pd
import jpype
import scyjava as sj


def detect_spots_log(
    volume: np.ndarray,
    ij: Optional[Any] = None,
    scale_zyx: Tuple[float, float, float] = (1.625, 0.40625, 0.40625),
    radius: float = 2.5,
    threshold: float = 5.0,
    target_channel: int = 1,
    do_subpixel_localization: bool = True,
    do_median_filtering: bool = False,
) -> pd.DataFrame:
    """
    Run TrackMate's built-in 3D/4D LoG detector (detection only, no tracking).
    """
    from . import init_fiji_headless, volume_to_calibrated_imageplus

    if ij is None:
        ij = init_fiji_headless()

    Model = sj.jimport("fiji.plugin.trackmate.Model")
    Settings = sj.jimport("fiji.plugin.trackmate.Settings")
    TrackMate = sj.jimport("fiji.plugin.trackmate.TrackMate")
    LogDetectorFactory = sj.jimport("fiji.plugin.trackmate.detection.LogDetectorFactory")

    imp = volume_to_calibrated_imageplus(ij, volume, scale_zyx=scale_zyx)

    model = Model()
    settings = Settings(imp)
    detector_factory = LogDetectorFactory()
    settings.detectorFactory = detector_factory

    det_settings = detector_factory.getDefaultSettings()
    det_settings.put("DO_SUBPIXEL_LOCALIZATION", bool(do_subpixel_localization))
    det_settings.put("RADIUS", float(radius))
    det_settings.put("TARGET_CHANNEL", jpype.JInt(target_channel))
    det_settings.put("THRESHOLD", float(threshold))
    det_settings.put("DO_MEDIAN_FILTERING", bool(do_median_filtering))
    settings.detectorSettings = det_settings

    # TrackMate 8.x returns None (Java null) when valid, or an error String when invalid
    check_res = detector_factory.checkSettings(det_settings)
    if isinstance(check_res, bool):
        if not check_res:
            raise RuntimeError("TrackMate LoG detector settings invalid.")
    elif check_res is not None:
        raise RuntimeError(f"TrackMate LoG detector settings invalid: {check_res}")

    tm = TrackMate(model, settings)
    if not tm.execDetection():
        raise RuntimeError(f"TrackMate LoG detection failed: {tm.getErrorMessage()}")

    dz, dy, dx = scale_zyx
    rows = []
    spots = model.getSpots()
    spots.setVisible(True)
    # Pass False (visibleSpotsOnly=False) so all spots from execDetection() are retrieved
    for idx, spot in enumerate(spots.iterable(False)):
        x_phys = float(spot.getFeature("POSITION_X"))
        y_phys = float(spot.getFeature("POSITION_Y"))
        z_phys = float(spot.getFeature("POSITION_Z"))
        rows.append(
            {
                "spot_id": idx,
                "frame": int(round(float(spot.getFeature("FRAME")))),
                "z": z_phys,
                "y": y_phys,
                "x": x_phys,
                "z_px": z_phys / dz,
                "y_px": y_phys / dy,
                "x_px": x_phys / dx,
                "radius": float(spot.getFeature("RADIUS")),
                "quality": float(spot.getFeature("QUALITY")),
            }
        )

    columns = ["spot_id", "frame", "z", "y", "x", "z_px", "y_px", "x_px", "radius", "quality"]
    if not rows:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(rows, columns=columns)
    df.sort_values(["frame", "spot_id"], inplace=True, ignore_index=True)
    df["spot_id"] = np.arange(len(df), dtype=np.int64)
    return df