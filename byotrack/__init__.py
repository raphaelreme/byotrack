"""ByoTrack

Unified python API for biological particle tracking.

We provide a unified python API for tracking that can be easily
extended with new (and old) algorithms. We also provide
implementations of well-known algorithms following our API.

Overview:
* Video
    * Able to read classical format (supported by opencv) + tiff
* Particle Tracking
    * MultiStepTracker (Detect / Link / Refine)
* Particle Detections
    * Wavelet Detector [2] (Similar as the one in Icy [1] but coded in pytorch)
    * Stardist [3] (Inference only. Training should be done with the official implementation)
* Particle Linking
    * EMHT [4] (Wraps the implementation in Icy [1], requires Icy to be installed)
    * u-track / TrackMate [7] (Wraps TrackMate [6, 8] implementation in ImageJ/Fiji, requires Fiji to be installed)
* Tracks Refining
    * Cleaning
    * EMC2 [5]: Track stitching (gap closing)
    * Interpolate missing positions



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
from byotrack.api.detector.detections import Detections
from byotrack.api.detector.detector import BatchDetector, Detector
from byotrack.api.linker import Linker, OnlineLinker
from byotrack.api.features_extractor import FeaturesExtractor, MultiFeaturesExtractor
from byotrack.api.optical_flow.optical_flow import OpticalFlow
from byotrack.api.refiner import Refiner
from byotrack.api.tracker import BatchMultiStepTracker, MultiStepTracker, Tracker
from byotrack.api.tracks import Track
from byotrack.video import Video, VideoTransformConfig


__version__ = "1.0.3"
