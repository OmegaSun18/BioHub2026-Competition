"""
TrackMate Cell-Division Lineage Tracking Modules.

Injects pre-computed spot centroids into TrackMate's tracking engine and isolates
mitotic division events (parent -> two daughters tracked for up to `t` steps).
"""

from typing import Any
import jpype
import scyjava as sj
import imagej

from .lap_division_tracker import track_cell_divisions


def init_fiji_headless(
    fiji_path: str = "/opt/Fiji.app",
    java_version: str = "21",
    java_vendor: str = "zulu",
) -> Any:
    """
    Safely initialize a headless PyImageJ gateway once per Python process.
    """
    if not jpype.isJVMStarted():
        sj.config.set_java_constraints(fetch=True, vendor=java_vendor, version=java_version)
        sj.config.add_option("-Djava.awt.headless=true")
    return imagej.init(fiji_path, mode="headless")


__all__ = [
    "init_fiji_headless",
    "track_cell_divisions",
]