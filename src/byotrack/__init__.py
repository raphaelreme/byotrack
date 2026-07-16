"""ByoTrack.

A Python library for tracking biological objects in microscopy videos (2D and 3D).

ByoTrack provides a fast, modular, and research-friendly tracking framework that integrates
seamlessly with the Python scientific ecosystem and established bioimage analysis platforms such as
Fiji, Icy, and Napari.

It defines a modular tracking API that can be easily extended to design and evaluate new methods,
and includes implementations of several state-of-the-art detection and tracking approaches.

Pipeline:
---------

    Video → Detection → Detection Refinement → Linking → Track Refinement → Tracks

* **Detection**
    * WaveletDetector [2]: PyTorch wavelet-based spot detector (inspired by Icy [1])
    * StarDistDetector [3]: Wrapper for StarDist deep learning detector
* **Detection Refinement**
    * Detection filtering (intensity and size criteria)
    * Watershed (semantic → instance segmentation)
* **Linking**
    * Nearest-neighbor (Euclidean, optical flow, Kalman/SKT [9], KOFT [9], adaptive gating [12])
    * EMHT [4]: Wrapper around the Icy implementation
    * TrackMate / u-track [7]: Wrapper around Fiji's TrackMate implementation [6, 8]
    * TrackAstra wrapper
* **Track Refinement**
    * Cleaning: remove outlier tracks based on length and motion criteria
    * EMC2 [5]: gap closing via tracklet stitching
    * Interpolation: replace missed detections with interpolated positions
    * Smoothing: RTS optimal Kalman smoother
* **Optical Flow**
    * OpenCV (2D only) and Scikit-Image (2D + 3D) wrappers
    * Used for linking, stitching, and interpolation
* **Datasets**
    * Cell Tracking Challenge (CTC) [10]
    * SINETRA [11]
* **Metrics**
    * CTC segmentation, detection, and tracking metrics


Getting started:
----------------

```python
import byotrack
import byotrack.napari.viewer  # Requires Napari

# Load some specific implementations
from byotrack.implementation.detector.wavelet import WaveletDetector
from byotrack.implementation.linker.frame_by_frame.kalman_linker import KalmanLinkerParameters, KalmanLinker
from byotrack.implementation.refiner.interpolater import ForwardBackwardInterpolater

# Read a video from a path, normalize and select channel
video = byotrack.Video(video_path)  # File/folder on disk or directly from array in RAM
video = video.normalize(q_min=0.01, q_max=0.999)[..., :1]  # First channel. Ignore other channels if any.

byotrack.napari.visualize(video)

# First, let's detect targets with the Wavelet Spot Detector
## Smaller scale <=> search for smaller spots
## The noise threshold is linear with k. If you increase it, you will retrieve less spots.
detector = WaveletDetector(scale=1, k=3.0, min_area=5.0)
detections_sequence = detector.run(video)

byotrack.napari.visualize(video, detections_sequence)

# Then, let's use a KalmanLinker with default parameters estimated from the detections
specs = KalmanLinkerParameters(  # Check the documentation for more parameters
    track_building="detection",  # Position from detections (non-smoothed)
).estimate(detections_sequence)
linker = kalman_linker.KalmanLinker(specs)
tracks = linker.run(video, detections_sequence)

# Last, you may filter short tracks and interpolate over missed detections if any
tracks = [track for track in tracks if len(track) > 50]  # Keep only tracks than last 50 frames
tracks = ForwardBackwardInterpolater().run(video, tracks)  # Interpolate over missed detections

# Final visualization
byotrack.napari.viewer.visualize(video, detections_sequence, tracks)

# Export tracks
byotrack.Track.save(tracks, output_path)
```

Please refer to the official documentation: https://byotrack.readthedocs.io/en/latest/

"""

import importlib.metadata

# Import API and main features but no implementation
from byotrack import fiji, icy, video
from byotrack._env import NUMBA_CACHE, ZSTD_SEG
from byotrack.api.detections.bbox_detections import BBoxDetections
from byotrack.api.detections.detections import (
    Detections,
    DetectionsLike,
    as_detections,
    fast_relabel,
    labels_of,
    relabel_consecutive,
)
from byotrack.api.detections.point_detections import PointDetections
from byotrack.api.detections.segmentation_detections import SegmentationDetections
from byotrack.api.detector import BatchDetector, DetectionsRefiner, Detector, GroundTruthDetector
from byotrack.api.features_extractor import FeaturesExtractor, MultiFeaturesExtractor
from byotrack.api.linker import Linker, OnlineLinker
from byotrack.api.optical_flow.optical_flow import OpticalFlow
from byotrack.api.refiner import Refiner
from byotrack.api.tracker import BatchMultiStepTracker, MultiStepTracker, Tracker
from byotrack.api.tracking_graph import TrackingGraph
from byotrack.api.tracks import Track, update_detections_from_tracks
from byotrack.video import Video, VideoTransformConfig

__version__ = importlib.metadata.version("byotrack")
__all__ = [
    "NUMBA_CACHE",
    "ZSTD_SEG",
    "BBoxDetections",
    "BatchDetector",
    "BatchMultiStepTracker",
    "Detections",
    "DetectionsLike",
    "DetectionsRefiner",
    "Detector",
    "FeaturesExtractor",
    "GroundTruthDetector",
    "Linker",
    "MultiFeaturesExtractor",
    "MultiStepTracker",
    "OnlineLinker",
    "OpticalFlow",
    "PatchDetections",
    "PointDetections",
    "Refiner",
    "SegmentationDetections",
    "Track",
    "Tracker",
    "TrackingGraph",
    "Video",
    "VideoTransformConfig",
    "as_detections",
    "fast_relabel",
    "fiji",
    "icy",
    "labels_of",
    "relabel_consecutive",
    "update_detections_from_tracks",
    "video",
]
