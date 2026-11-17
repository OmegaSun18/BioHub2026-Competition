"""
TrackMate & 3D StarDist Detection Modules.

Provides pure, detection-only functions returning standardized centroid DataFrames.
"""

from typing import Any, Tuple
import numpy as np
import jpype
import scyjava as sj
import imagej

from .log_detector import detect_spots_log
from .dog_detector import detect_spots_dog
from .stardist_3d_detector import detect_spots_stardist_3d


def init_fiji_headless(
    fiji_path: str = "/opt/Fiji.app",
    java_version: str = "21",
    java_vendor: str = "zulu",
) -> Any:
    """
    Safely initialize a headless PyImageJ gateway once per Python process.

    Parameters
    ----------
    fiji_path : str
        Path to the local Fiji installation.
    java_version : str
        Required Java version for TrackMate (default '21').
    java_vendor : str
        JDK vendor constraint for scyjava (default 'zulu').

    Returns
    -------
    imagej.ImageJ
        Initialized headless ImageJ gateway.
    """
    if not jpype.isJVMStarted():
        sj.config.set_java_constraints(fetch=True, vendor=java_vendor, version=java_version)
        sj.config.add_option("-Djava.awt.headless=true")
    return imagej.init(fiji_path, mode="headless")


def volume_to_calibrated_imageplus(
    ij: Any,
    volume: np.ndarray,
    scale_zyx: Tuple[float, float, float] = (1.625, 0.40625, 0.40625),
) -> Any:
    """
    Convert a 4D (T, Z, Y, X) or 5D (T, C, Z, Y, X) NumPy array into a calibrated ImagePlus.
    """
    if volume.ndim == 4:
        t_dim, z_dim, y_dim, x_dim = volume.shape
        c_dim = 1
        arr = volume
    elif volume.ndim == 5:
        t_dim, c_dim, z_dim, y_dim, x_dim = volume.shape
        # Rearrange (T, C, Z, Y, X) -> (T, Z, C, Y, X) for ImageJ hyperstack layout
        arr = np.transpose(volume, (0, 2, 1, 3, 4))
    else:
        raise ValueError(
            f"Expected 4D (T, Z, Y, X) or 5D (T, C, Z, Y, X) array, got shape {volume.shape}."
        )

    imp = ij.py.to_imageplus(np.ascontiguousarray(arr))
    imp.setDimensions(int(c_dim), int(z_dim), int(t_dim))

    cal = imp.getCalibration()
    cal.pixelDepth = float(scale_zyx[0])
    cal.pixelHeight = float(scale_zyx[1])
    cal.pixelWidth = float(scale_zyx[2])
    return imp


__all__ = [
    "init_fiji_headless",
    "volume_to_calibrated_imageplus",
    "detect_spots_log",
    "detect_spots_dog",
    "detect_spots_stardist_3d",
]