# ByoTrack

[![Lint and Test](https://github.com/raphaelreme/byotrack/actions/workflows/tests.yml/badge.svg)](https://github.com/raphaelreme/byotrack/actions/workflows/tests.yml)\
[![Documentation Status](https://readthedocs.org/projects/byotrack/badge/?version=latest)](https://byotrack.readthedocs.io/en/latest/?badge=latest)

![pipeline](docs/source/images/tracking.svg)

**ByoTrack** is a **Python library for tracking biological objects in
microscopy videos (2D and 3D)**.

The goal of ByoTrack is to provide a **fast, modular, and
research-friendly tracking framework** that integrates naturally with
the **Python scientific ecosystem** and existing **bioimage analysis
platforms**.

Many classical bioimage tools (such as **Icy** and **Fiji/ImageJ**)
provide powerful tracking algorithms, but they are primarily implemented
in **Java**, which makes experimentation and integration with modern
Python-based methods---especially **deep learning models**---more
difficult.

ByoTrack bridges this gap by providing a **clean Python API for object
tracking pipelines**, allowing researchers to easily combine:

-   Classical computer vision algorithms
-   Modern deep learning detectors
-   Existing bioimage software
-   Custom research methods


------------------------------------------------------------------------

## Key Features

### Fast and scalable

-   Built with **NumPy**, **PyTorch**, and **Numba**
-   Efficient computations for large microscopy datasets
-   Designed to scale to long videos and dense tracking problems

### Support for 2D and 3D microscopy

-   Track objects in **2D or 3D videos**
-   Compatible with common microscopy file formats

### Online tracking

-   Most algorithms can process frames **sequentially** never loading the full video in memory
-   Suitable for **real-time pipelines**

### Modular architecture

Tracking is decomposed into independent components (Tracking-By-Detection):

    Video → Detection → Detection Refinement → Linking → Track Refinement

Each component can be **replaced or extended**, making ByoTrack
ideal for research and development of new tracking approaches.


### Deep learning integration

Works naturally with Python ML frameworks for both detections and tracking.

-   Detections converted from np.array or torch.tensor
-   Detectors may wrap Tensorflow or PyTorch DL-solutions (see **StarDistDetector**)
-   Linking may exploit deep-learning computed costs (see **TrackOnStraLinker**)
-   Linkers may use its own DL features extraction methods online.

### Interoperability with bioimage tools

ByoTrack integrates with established platforms:

-   **Fiji / ImageJ**
-   **Icy**
-   **Napari** (in progress)

This allows users to combine **existing tracking algorithms** with
Python workflows.


------------------------------------------------------------------------

## Installation

``` bash
pip install byotrack
```

Some implementations require additional dependencies that are not installed with the library, to use them you need to install their dependencies on your own.
Here is the complete list:


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

## Cell Tracking Challenge

Our submission to the **Cell Linking Benchmark** of the Cell Tracking Challenge is available in the [examples/ctc](examples/ctc/README.md) folder.


------------------------------------------------------------------------

## Contributing

Contributions are welcome.

Typical contributions include:

-   New detections or linking algorithms
-   Dataset loaders
-   Evaluation metrics

Guidelines will be added soon.

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
* [10] M. Maška, V. Ulman, D. Svoboda, P. Matula, et al., "A benchmark for
       comparison of cell tracking algorithms", in Bioinformatics, 2014.
* [11] R. Reme, A. Newson, E. Angelini, J.-C. Olivo-Marin and T. Lagache,
       "SINETRA: a Versatile Framework for Evaluating Single Neuron Tracking
       in Behaving Animals", arXiv preprint arXiv:2411.09462, 2024.
* [12] A. Genovesio, Z. Belhassine, and J.-C. Olivo-Marin, "Adaptive gating
       in Gaussian Bayesian multi-target tracking", in 2004 International
       Conference on Image Processing (ICIP'04).
