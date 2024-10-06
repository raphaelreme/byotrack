Welcome to ByoTrack's documentation!
====================================

.. image:: images/tracking.svg
    :alt: Tracking pipeline

**ByoTrack** is a Python library that enables tracking of biological object in videos (2D or 3D). Many bioimage informatics tools already implement their
own tracking tools (Icy, ImageJ, TrackMate...) but most of them are implemented in Java which makes it difficult for non-Java developers to
experiment with the code. It is also difficult to integrate deep learning algorithms (mainly developed in Python) into these software.

We provide a unified python API for tracking that can be easily extended with new (and old) algorithms. We also provide implementations
of well-known algorithms following our API. ByoTrack is based on numpy, pytorch and numba allowing fast computations with the access to the full python ecosystem.

.. note::

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

* **[1]** F. De Chaumont, S. Dallongeville, N. Chenouard, et al., "Icy: an open bioimage informatics platform for extended reproducible research", Nature methods, vol. 9, no. 7, pp. 690-696, 2012.
* **[2]** J.-C. Olivo-Marin, "Extraction of spots in biological images using multiscale products", Pattern Recognition, vol. 35, no. 9, pp. 1989-1996, 2002.
* **[3]** U. Schmidt, M. Weigert, C. Broaddus, and G. Myers, "Cell detection with star-convex polygons", in Medical Image Computing and Computer Assisted Intervention-MICCAI 2018: 21st International Conference, Granada, Spain, September 16-20, 2018, Proceedings, Part II 11. Springer, 2018, pp. 265-273.
* **[4]** N. Chenouard, I. Bloch, and J.-C. Olivo-Marin, "Multiple hypothesis tracking for cluttered biological image sequences", IEEE transactions on pattern analysis and machine intelligence, vol. 35, no. 11, pp. 2736-3750, 2013.
* **[5]** T. Lagache, A. Hanson, J. Perez-Ortega, et al., "Tracking calcium dynamics from individual neurons in behaving animals", PLoS computational biology, vol. 17, pp. e1009432, 10 2021.
* **[6]** J. Schindelin, I. Arganda-Carreras, E. Frise, et al., "Fiji: an open-source platform for biological-image analysis", Nature Methods, 9(7), 676-682, 2012.
* **[7]** K. Jaqaman, D. Loerke, M. Mettlen, et al., "Robust single-particle tracking in live-cell time-lapse sequences.", Nature Methods, 5(8), 695-702, 2008.
* **[8]** J.-Y. Tinevez, N. Perry, J. Schindelin, et al., "TrackMate: An open and extensible platform for single-particle tracking.", Methods, 115, 80-90, 2017.
* **[9]** R. Reme, A. Newson, E. Angelini, J.-C. Olivo-Marin and T. Lagache, "Particle tracking in biological images with optical-flow enhanced kalman filtering", in International Symposium on Biomedical Imaging (ISBI2024).

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
