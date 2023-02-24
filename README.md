# Byotrack

[![Documentation Status](https://readthedocs.org/projects/byotrack/badge/?version=latest)](https://byotrack.readthedocs.io/en/latest/?badge=latest)

Unified python API for biological particle tracking.

Many bioimage informatics tools already implement their own tracking tools (Icy, ImageJ, TrackMate...) but most of them are implemented in Java which makes it difficult for non-Java developers to experiment with the code. It is also difficult to integrate deep learning algorithms (mainly developed in Python) into these software.

We provide a unified python API for tracking that can be easily extended with new (and old) algorithms. We also provide implementations of well-known algorithms following our API.

Overview:
* Video
    * Able to read classical format (supported by opencv) + tiff
* Particle Tracking
    * MultiStepTracker (Detect / Link / Refine)
* Particle Detections
    * Spot Detector [2] (Similar as the one in Icy [1] but coded in pytorch)
    * Stardist [3] (In coming...)
* Particle Linking
    * EMHT [4] (Wrapper to the one implemented in Icy [1], requires Icy to be installed)
* Tracks Refining
    * Cleaning (In coming...)
    * EMC2 [5]: Track stitching (gap closing) (In coming...)


## Install

```bash
$ pip install byotrack
```

Some tracker implementations require additional dependencies that are not installed with the library, to use them you need to install their dependencies on your own.
Here is the complete list:

- IcyEMHTLinker
    - Icy: [Download Icy](https://icy.bioimageanalysis.org/download/)


## Getting started

```python
import byotrack
```

Please have a look at our examples notebook and our [documentation](https://byotrack.readthedocs.io/en/latest/index.html).

## Contribute

In coming...

## References


* [1] F. De Chaumont, S. Dallongeville, N. Chenouard, et al., “Icy:
      an open bioimage informatics platform for extended reproducible
      research”, Nature methods, vol. 9, no. 7, pp. 690–696, 2012.
* [2] J.-C. Olivo-Marin, “Extraction of spots in biological images
      using multiscale products”, Pattern Recognition, vol. 35, no. 9,
      pp. 1989–1996, 2002.
* [3] U. Schmidt, M. Weigert, C. Broaddus, and G. Myers, “Cell de-
      tection with star-convex polygons,” in Medical Image Computing
      and Computer Assisted Intervention–MICCAI 2018: 21st
      International Conference, Granada, Spain, September 16-20,
      2018, Proceedings, Part II 11. Springer, 2018, pp. 265–273.
* [4] N. Chenouard, I. Bloch, and J.-C. Olivo-Marin, “Multiple hypothesis
      tracking for cluttered biological image sequences”,
      IEEE transactions on pattern analysis and machine intelligence,
      vol. 35, no. 11, pp. 2736–3750, 2013.
* [5] T. Lagache, A. Hanson, J. Perez-Ortega, et al., “Tracking calcium
      dynamics from individual neurons in behaving animals”,
      PLoS computational biology, vol. 17, pp. e1009432, 10 2021.
