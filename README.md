# ByoTrack
[![Lint and Test](https://github.com/raphaelreme/byotrack/actions/workflows/tests.yml/badge.svg)](https://github.com/raphaelreme/byotrack/actions/workflows/tests.yml)
[![Documentation Status](https://readthedocs.org/projects/byotrack/badge/?version=latest)](https://byotrack.readthedocs.io/en/latest/?badge=latest)

![pipeline](docs/source/images/tracking.svg)

**ByoTrack** is a Python library that enables tracking of biological object in videos (2D or 3D).

Many bioimage informatics tools already implement their own tracking tools (Icy [1], ImageJ [6], ...) but most of them are implemented in Java which makes it difficult for non-Java developers to experiment with the code. It is also difficult to integrate deep learning algorithms (mainly developed in Python) into these software.

We provide a unified python API for tracking that can be easily extended with new (and old) algorithms. We also provide implementations of well-known algorithms following our API. ByoTrack is based on numpy, pytorch and numba allowing fast computations with the access to the full python ecosystem.

Overview:
* Video
    * Able to read most classical format (supported by opencv) + tiff
    * Supports 2D and 3D.
* Particle Tracking
    * MultiStepTracker (Detect / Link / Refine)
* Particle Detections
    * Wavelet Detector [2] (Similar as the one in Icy [1] but coded in pytorch)
    * Stardist [3] (Inference only. Training should be done with the [official implementation](https://github.com/stardist/stardist))
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


## Install

```bash
$ pip install byotrack
```

Some tracker implementations require additional dependencies that are not installed with the library, to use them you need to install their dependencies on your own.
Here is the complete list:


- StarDistDetector
    - stardist (+ tensorflow): [Install stardist](https://github.com/stardist/stardist#installation)
- KalmanLinker & KOFTLinker
    - torch_kf: [Install torch-kf](https://github.com/raphaelreme/torch-kf#install)
- IcyEMHTLinker
    - Icy: [Download Icy](https://icy.bioimageanalysis.org/download/)
    - Spot Tracking Blocks plugin: [Install an Icy plugin](https://icy.bioimageanalysis.org/tutorial/how-to-install-an-icy-plugin/)
- TrackMateLinker
    - Fiji: [Download Fiji](https://imagej.net/downloads)
    - tifffile: [Install tifffile](https://github.com/cgohlke/tifffile#quickstart)
- SkimageOpticalFlow
    - scikit-image: [Install scikit-image](https://scikit-image.org/docs/stable/user_guide/install.html)

For visualization, with `byotrack.visualize` module you need to [install matplotlib](https://matplotlib.org/stable/install/index.html).

## Getting started

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

Please refer to the [official documentation](https://byotrack.readthedocs.io/en/latest/).

## Contribute

In coming...

## Cite us

```bibtex
@article{hanson2024automatic,
  title={Automatic monitoring of neural activity with single-cell resolution in behaving Hydra},
  author={Hanson, Alison and Reme, Raphael and Telerman, Noah and Yamamoto, Wataru and Olivo-Marin, Jean-Christophe and Lagache, Thibault and Yuste, Rafael},
  journal={Scientific Reports},
  volume={14},
  number={1},
  pages={5083},
  year={2024},
  publisher={Nature Publishing Group UK London}
}
```

## References


* [1] F. De Chaumont, S. Dallongeville, N. Chenouard, et al., "Icy:
      an open bioimage informatics platform for extended reproducible
      research", Nature methods, vol. 9, no. 7, pp. 690–696, 2012.
* [2] J.-C. Olivo-Marin, "Extraction of spots in biological images
      using multiscale products", Pattern Recognition, vol. 35, no. 9,
      pp. 1989–1996, 2002.
* [3] U. Schmidt, M. Weigert, C. Broaddus, and G. Myers, "Cell detection
      with star-convex polygons", in Medical Image Computing and
      Computer Assisted Intervention–MICCAI 2018: 21st International
      Conference, Granada, Spain, September 16-20, 2018, Proceedings,
      Part II 11. Springer, 2018, pp. 265–273.
* [4] N. Chenouard, I. Bloch, and J.-C. Olivo-Marin, "Multiple hypothesis
      tracking for cluttered biological image sequences",
      IEEE transactions on pattern analysis and machine intelligence,
      vol. 35, no. 11, pp. 2736–3750, 2013.
* [5] T. Lagache, A. Hanson, J. Perez-Ortega, et al., "Tracking calcium
      dynamics from individual neurons in behaving animals",
      PLoS computational biology, vol. 17, pp. e1009432, 10 2021.
* [6] J. Schindelin, I. Arganda-Carreras, E. Frise, et al., "Fiji:
      an open-source platform for biological-image analysis", Nature
      Methods, 9(7), 676–682, 2012.
* [7] K. Jaqaman, D. Loerke, M. Mettlen, et al., "Robust single-particle
      tracking in live-cell time-lapse sequences.", Nature Methods, 5(8),
      695–702, 2008.
* [8] J.-Y. Tinevez, N. Perry, J. Schindelin, et al., "TrackMate: An
      open and extensible platform for single-particle tracking.",
      Methods, 115, 80–90, 2017.
* [9] R. Reme, A. Newson, E. Angelini, J.-C. Olivo-Marin and T. Lagache,
      "Particle tracking in biological images with optical-flow enhanced
      kalman filtering", in International Symposium on Biomedical Imaging
      (ISBI2024).
