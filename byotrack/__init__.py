"""ByoTrack

Unified python API for biological particle tracking (2D/3D).

Many bioimage informatics tools already implement their own tracking tools (Icy [1], ImageJ [6], ...)
but most of them are implemented in Java which makes it difficult for non-Java developers to experiment
with the code. It is also difficult to integrate deep learning algorithms (mainly developed in Python)
into these software.

We provide a unified python API for tracking that can be easily extended with new (and old) algorithms.
We also provide implementations of well-known algorithms following our API. ByoTrack is based on numpy,
pytorch and numba allowing fast computations with the access to the full python ecosystem.

Overview:
* Video
    * Able to read most classical format (supported by opencv) + tiff
    * Supports 2D and 3D.
* Particle Tracking
    * MultiStepTracker (Detect / Link / Refine)
* Particle Detections
    * Wavelet Detector [2] (Similar as the one in Icy [1] but coded in pytorch)
    * Stardist [3] (Inference only. Training should be done with the
        [official implementation](https://github.com/stardist/stardist))
* Particle Linking
    * Nearest neighbors using optical flow, kalman filters or both (KOFT) [9]
    * EMHT [4] (Wraps the implementation in Icy [1], requires Icy to be installed)
    * u-track / TrackMate [7] (Wraps the TrackMate [6, 8] implementation in ImageJ/Fiji, requires Fiji to be installed)
* Tracks Refining
    * Cleaning
    * EMC2 [5]: Track stitching (gap closing)
    * Interpolate missing positions
* Optical Flow
    * Support for Open-CV and Scikit-Image algorithms. Can be used for particle linking, track stitching
    and interpolations.
* Datasets
    * Supports for loading annotations/video from datasets.
    * Cell Tracking Challenge (CTC) [10]
    * SINETRA [11]
* Metrics:
    * Support for some segmentation/detection/tracking metrics.
        Currently, only CTC metrics are provided. More to come...


Getting started:
----------------

```python
import byotrack

# Load some specific implementations
from byotrack.implementation.detector.wavelet import WaveletDetector
from byotrack.implementation.linker.icy_emht import IcyEMHTLinker
from byotrack.implementation.refiner.cleaner import Cleaner
from byotrack.implementation.refiner.stitching import EMC2Stitcher

# Read a video from a path, normalize and aggregate channels
video = byotrack.Video(video_path)
transform_config = VideoTransformConfig(aggregate=True, normalize=True, q_min=0.01, q_max=0.999)
video.set_transform(transform_config)

# Create a multi step tracker
## First the detector
## Smaller scale <=> search for smaller spots
## The noise threshold is linear with k. If you increase it, you will retrieve less spots.
detector = WaveletDetector(scale=1, k=3.0, min_area=5)

## Second the linker
## Hyperparameters are automatically chosen by Icy
linker = IcyEMHTLinker(icy_path)

## Finally refiners
## If needed you can add Cleaning and Stitching operations
refiners = []
if True:
    refiners.append(Cleaner(5, 3.5))  # Split tracks on position jumps and drop small ones
    refiners.append(EMC2Stitcher())  # Merge tracks if they track the same particle

tracker = byotrack.MultiStepTracker(detector, linker, refiners)

# Run the tracker
tracks = tracker.run(video)

# Save tracks
byotrack.Track.save(tracks, output_path)
```

Please refer to the ![official documentation](https://byotrack.readthedocs.io/en/latest/?badge=latest)

"""

# Import API and main features but no implementation
from byotrack._env import NUMBA_CACHE
from byotrack.api.detector.detections import Detections
from byotrack.api.detector.detector import BatchDetector, Detector
from byotrack.api.linker import Linker, OnlineLinker
from byotrack.api.features_extractor import FeaturesExtractor, MultiFeaturesExtractor
from byotrack.api.optical_flow.optical_flow import OpticalFlow
from byotrack.api.refiner import Refiner
from byotrack.api.tracker import BatchMultiStepTracker, MultiStepTracker, Tracker
from byotrack.api.tracks import Track
from byotrack.video import Video, VideoTransformConfig


__version__ = "1.3.6"
