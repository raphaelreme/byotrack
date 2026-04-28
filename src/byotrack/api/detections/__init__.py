"""Detection-related API."""

from byotrack.api.detections.bbox_detections import BBoxDetections
from byotrack.api.detections.detections import Detections, as_detections, fast_relabel, relabel_consecutive
from byotrack.api.detections.point_detections import PointDetections
from byotrack.api.detections.segmentation_detections import SegmentationDetections

__all__ = [
    "BBoxDetections",
    "Detections",
    "PointDetections",
    "SegmentationDetections",
    "as_detections",
    "fast_relabel",
    "relabel_consecutive",
]
