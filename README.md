# ByoTrack

[![Lint and Test](https://github.com/raphaelreme/byotrack/actions/workflows/tests.yml/badge.svg)](https://github.com/raphaelreme/byotrack/actions/workflows/tests.yml)\
[![Documentation Status](https://readthedocs.org/projects/byotrack/badge/?version=latest)](https://byotrack.readthedocs.io/en/latest/?badge=latest)

![pipeline](docs/source/images/tracking.svg)

**ByoTrack** *is a Python library for tracking biological objects in microscopy videos (2D and 3D)*.

Its goal is to provide a **fast, modular, and research-friendly tracking framework** that integrates
seamlessly with the **Python scientific ecosystem** and established **bioimage analysis platforms** such as
*Fiji*, *Icy* and *Napari*.

ByoTrack defines a **modular tracking API** that can be easily extended to design and evaluate new methods.
It also includes implementations of several **state-of-the-art detection and tracking** approaches following this API.

Some components are **implemented natively** in Python (e.g. *WaveletDetector*, *KalmanLinker*, *KOFT*, *RTSSmoother*,
*EMC2Stitcher*), while others **wrap existing tools** (e.g. *StarDistDetector*, *IcyEMHTLinker*, *TrackMateLinker*),
integrating with external software.

In addition, ByoTrack provides utilities for **data loading and preprocessing**, as well
as **evaluation and visualization** of tracking results.

> [!NOTE]
> ByoTrack has been primarily developed for scenarios involving up to a few thousand targets in 2D or 3D microscopy data.
> Some components assume that individual frames fit in memory, which may limit scalability to very large 3D volumes.
> If you encounter limitations with your use case, feel free to open an issue or contribute a pull request.

🏆 **ByoTrack (PAST-FR)** won the [Cell Linking Benchmark](https://celltrackingchallenge.net/latest-clb-results/) of
the *Cell Tracking Challenge* with its **SKT/KOFT** implementation
(see our [paper](https://ieeexplore.ieee.org/abstract/document/10635656/) for details).

------------------------------------------------------------------------

## Installation

### pip

``` bash
pip install byotrack
```

Some components require additional dependencies that are not installed with the library by default.
For these components, you need to install their specific dependencies. Here is the complete list:

- *StarDistDetector*
    - StarDist (+ Tensorflow): [Install StarDist](https://github.com/stardist/stardist#installation)
- *IcyEMHTLinker*
    - Icy: [Download Icy](https://icy.bioimageanalysis.org/download/)
    - Spot Tracking Blocks plugin: [Install an Icy plugin](https://icy.bioimageanalysis.org/tutorial/how-to-install-an-icy-plugin/)
- *TrackMateLinker*
    - Fiji: [Download Fiji](https://imagej.net/downloads)
- *TrackOnStraLinker*
    - TrackAstra: [Install trackastra](https://github.com/weigertlab/trackastra#installation)

For visualization, with `byotrack.visualize` module you need to [install Matplotlib](https://matplotlib.org/stable/install/index.html).


### From source

```bash
git clone git@github.com:raphaelreme/byotrack.git  # OR https://github.com/raphaelreme/byotrack.git
cd byotrack
pip install .
```

------------------------------------------------------------------------

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

Please refer to the [official documentation](https://byotrack.readthedocs.io/en/latest/) (https://byotrack.readthedocs.io/en/latest/).


------------------------------------------------------------------------

## Tracking Pipeline

ByoTrack implements a modular **multi-step tracking pipeline**:

    Video → Detection → Detection Refinement → Linking → Track Refinement


### Detection

Detect objects in each frame.

Implemented detectors:

-   **Wavelet detector** \[2\]\
    Similar to the Icy implementation but rewritten in PyTorch

-   **StarDist** \[3\]\
    Wrapper for inference (training is performed using the official
    StarDist library)

### Detection Refinement

Filter, refine, split and merge detections.

Implemented detection refiners:

-   **Detection filtering**\
    Filter detections based on intensity and size criteria.

-   **Watershed**\
    Convert semantic (binary) segmentation into instance segmentation with Watershed\
    Can also be applied to instance segmentation to refine the instances found


### Linking

Associate detections across frames.

Implemented linkers:

-   **Nearest-neighbor linking**

    -   Euclidean
    -   Optical flow
    -   Kalman filtering (SKT) \[9\]
    -   KOFT (Kalman + Optical Flow Tracker) \[9\]
    -   optional adaptive gating \[12\]

-   **EMHT** \[4\]\
    Wrapper around the Icy implementation

-   **TrackMate / u-track** \[7\]\
    Wrapper around Fiji's TrackMate implementation \[6,8\]

### Track Refinement

Post-processing operations applied to tracks:

-   **Cleaning**\
    Remove outliers tracks based on length and motion criteria

-   **Gap Closing / Stitching**\
    Tracklet Stitching via **EMC2** algorithm \[5\]

-   **Interpolation**\
    Replace miss-detection by an interpolated position\
    Extrapolate tracks on the full temporal sequence

-   **Smoothing**\
    RTS optimal Kalman smoother

------------------------------------------------------------------------

## Data, Evaluation & Utilities

### Data input

ByoTrack supports:

-   Most standard video formats via **OpenCV**
-   **TIFF stacks**
-   Folder of images (sorted by name)
-   Dedicated Python loading of the video (np.array) and detections (converted as a Detections object)

Note that microscope private formats are not supported but can be converted into TIFF manually using [bftools](https://bio-formats.readthedocs.io/en/latest/)

### Optical flow

Optical flow can be used for:

-   Linking detections
-   Gap closing / Stitching
-   Interpolation

Currently provide wrappers around implementations from:

-   OpenCV (TVL1, Farneback) (**Only 2D**)
-   Scikit-Image (ILK, TVL1) (**2D + 3D**)

### Datasets

Built-in loaders for common benchmarks:

-   **Cell Tracking Challenge (CTC)** \[10\]
-   **SINETRA** \[11\]


### Metrics

Evaluation utilities for:

-   Segmentation
-   Detection
-   Tracking

Currently implemented:

-   **Cell Tracking Challenge metrics**

More metrics will be added in future releases.


------------------------------------------------------------------------

## Cell Tracking Challenge

Our submission (PAST-FR) to the [**Cell Linking Benchmark**](https://celltrackingchallenge.net/latest-clb-results/) of the Cell Tracking Challenge is available in the [examples/ctc](examples/ctc/README.md) folder.



------------------------------------------------------------------------

## Contributing

Contributions are very welcome! Feel free to open an issue or submit a pull request.

Typical contributions could include:

-   New detections or linking algorithms
-   Dataset loaders
-   Evaluation metrics
-   New data format
-   Track analysis methods

See the contribution guidelines.

------------------------------------------------------------------------

## Cite us

If you use ByoTrack in your research, please cite:

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
      research", Nature methods, 2012.
* [2] J.-C. Olivo-Marin, "Extraction of spots in biological images
      using multiscale products", Pattern Recognition, 2002.
* [3] U. Schmidt, M. Weigert, C. Broaddus, and G. Myers, "Cell detection
      with star-convex polygons", MICCAI, 2018.
* [4] N. Chenouard, I. Bloch, and J.-C. Olivo-Marin, "Multiple hypothesis
      tracking for cluttered biological image sequences", IEEE TPAMI, 2013.
* [5] T. Lagache, A. Hanson, J. Perez-Ortega, et al., "Tracking calcium
      dynamics from individual neurons in behaving animals",
      PLoS Computational Biology, 2021.
* [6] J. Schindelin, I. Arganda-Carreras, E. Frise, et al., "Fiji:
      an open-source platform for biological-image analysis", Nature
      Methods, 2012.
* [7] K. Jaqaman, D. Loerke, M. Mettlen, et al., "Robust single-particle
      tracking in live-cell time-lapse sequences.", Nature Methods, 2008.
* [8] J.-Y. Tinevez, N. Perry, J. Schindelin, et al., "TrackMate: An
      open and extensible platform for single-particle tracking.",
      Methods, 2017.
* [9] R. Reme, A. Newson, E. Angelini, J.-C. Olivo-Marin and T. Lagache,
      "Particle tracking in biological images with optical-flow enhanced
      kalman filtering", IEEE ISBI, 2024.
* [10] M. Maška, V. Ulman, D. Svoboda, P. Matula, et al., "A benchmark for
       comparison of cell tracking algorithms", in Bioinformatics, 2014.
* [11] R. Reme, A. Newson, E. Angelini, J.-C. Olivo-Marin and T. Lagache,
       "SINETRA: a Versatile Framework for Evaluating Single Neuron Tracking
       in Behaving Animals", IEEE ISBI, 2025.
* [12] A. Genovesio, Z. Belhassine, and J.-C. Olivo-Marin, "Adaptive gating
       in Gaussian Bayesian multi-target tracking", IEEE ICIP, 2004.
