"""Bridge with Napari software.

Note:
    This package requires napari. (pip install napari)
"""

from byotrack.napari.utils import (
    detections_to_napari_points,
    detections_to_napari_segmentation,
    precompute_optical_flow,
    tracks_to_napari_tracks,
)
from byotrack.napari.viewer import (
    add_detections,
    add_optical_flow,
    add_tracks,
    add_video,
    visualize,
    visualize_flow_deformation,
)

__all__ = [
    "add_detections",
    "add_optical_flow",
    "add_tracks",
    "add_video",
    "detections_to_napari_points",
    "detections_to_napari_segmentation",
    "precompute_optical_flow",
    "tracks_to_napari_tracks",
    "visualize",
    "visualize_flow_deformation",
]
