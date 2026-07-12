"""Bridge with the Graph Exchange File Format (GEFF).

Note:
    This package requires geff. (pip install geff)

"""

from byotrack.geff.io import (
    load_detections_from_geff,
    load_detections_from_zarr,
    load_tracks_from_geff,
    load_video_from_geff,
    load_video_from_zarr,
    save_detections_to_zarr,
    save_tracks_to_geff,
    save_video_to_zarr,
)

__all__ = [
    "load_detections_from_geff",
    "load_detections_from_zarr",
    "load_tracks_from_geff",
    "load_video_from_geff",
    "load_video_from_zarr",
    "save_detections_to_zarr",
    "save_tracks_to_geff",
    "save_video_to_zarr",
]
