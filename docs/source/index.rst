Welcome to ByoTrack's documentation!
====================================

.. image:: images/tracking.svg
    :alt: Tracking pipeline

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

.. note::
   ByoTrack has been primarily developed for scenarios involving up to a few thousand targets in 2D or 3D microscopy data.
   Some components assume that individual frames fit in memory, which may limit scalability to very large 3D volumes.
   If you encounter limitations with your use case, feel free to open an issue or contribute a pull request.

🏆 **ByoTrack (PAST-FR)** won the `Cell Linking Benchmark <https://celltrackingchallenge.net/latest-clb-results/>`_ of
the *Cell Tracking Challenge* with its **SKT/KOFT** implementation
(see our `paper <https://ieeexplore.ieee.org/abstract/document/10635656/>`_ for details).

.. warning::

   This project and documentation is under active development.


Cite Us
-------

.. code-block:: bibtex

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


References
----------

* **[1]** F. De Chaumont, S. Dallongeville, N. Chenouard, et al., "Icy: an open bioimage informatics platform for extended reproducible research", Nature methods, 2012.
* **[2]** J.-C. Olivo-Marin, "Extraction of spots in biological images using multiscale products", Pattern Recognition, 2002.
* **[3]** U. Schmidt, M. Weigert, C. Broaddus, and G. Myers, "Cell detection with star-convex polygons", MICCAI, 2018.
* **[4]** N. Chenouard, I. Bloch, and J.-C. Olivo-Marin, "Multiple hypothesis tracking for cluttered biological image sequences", IEEE TPAMI, 2013.
* **[5]** T. Lagache, A. Hanson, J. Perez-Ortega, et al., "Tracking calcium dynamics from individual neurons in behaving animals", PLoS Computational Biology, 2021.
* **[6]** J. Schindelin, I. Arganda-Carreras, E. Frise, et al., "Fiji: an open-source platform for biological-image analysis", Nature Methods, 2012.
* **[7]** K. Jaqaman, D. Loerke, M. Mettlen, et al., "Robust single-particle tracking in live-cell time-lapse sequences.", Nature Methods, 2008.
* **[8]** J.-Y. Tinevez, N. Perry, J. Schindelin, et al., "TrackMate: An open and extensible platform for single-particle tracking.", Methods, 2017.
* **[9]** R. Reme, A. Newson, E. Angelini, J.-C. Olivo-Marin and T. Lagache, "Particle tracking in biological images with optical-flow enhanced kalman filtering", IEEE ISBI, 2024.
* **[10]** M. Maška, V. Ulman, D. Svoboda, P. Matula, et al., "A benchmark for comparison of cell tracking algorithms", in Bioinformatics, 2014.
* **[11]** R. Reme, A. Newson, E. Angelini, J.-C. Olivo-Marin and T. Lagache, "SINETRA: a Versatile Framework for Evaluating Single Neuron Tracking in Behaving Animals", IEEE ISBI, 2025.
* **[12]** A. Genovesio, Z. Belhassine, and J.-C. Olivo-Marin, "Adaptive gating in Gaussian Bayesian multi-target tracking", IEEE ICIP, 2004.

Contents
--------

.. toctree::
   :maxdepth: 2
   :glob:
   :caption: Install

   install

.. toctree::
   :maxdepth: 2
   :glob:
   :caption: Getting started

   run_examples/ByoTrack fundamental.ipynb
   run_examples/Video.ipynb
   run_examples/Detectors.ipynb
   run_examples/Linkers.ipynb

.. toctree::
   :glob:
   :caption: API

   api/video
   api/detections
   api/tracks
   api/tracker
   api/detector
   api/linker
   api/refiner
   api/optical_flow
   api/parameters


.. toctree::
   :glob:
   :caption: API implementations

   implementation/detectors/detectors
   implementation/linkers/linkers
   implementation/refiners/refiners
   implementation/optical_flows/optical_flows

.. toctree::
   :glob:
   :caption: Datasets

   datasets/ctc
   datasets/sinetra

.. toctree::
   :glob:
   :caption: Metrics

   metrics/ctc

.. toctree::
   :glob:
   :caption: Icy support

   icy

.. toctree::
   :glob:
   :caption: Fiji support

   fiji

.. toctree::
   :glob:
   :caption: Visualization

   visualize
