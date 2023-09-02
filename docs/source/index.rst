Welcome to ByoTrack's documentation!
====================================

.. image:: images/tracking.svg
    :alt: Tracking pipeline

**ByoTrack** is a Python library that enables tracking of biological object in videos. Many bioimage informatics tools already implement their
own tracking tools (Icy, ImageJ, TrackMate...) but most of them are implemented in Java which makes it difficult for non-Java developers to
experiment with the code. It is also difficult to integrate deep learning algorithms (mainly developed in Python) into these software.

We provide a unified python API for tracking that can be easily extended with new (and old) algorithms. We also provide implementations
of well-known algorithms following our API.

.. note::

   This project and documentation is under active development.


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

   run_examples/*

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
   api/parameters


.. toctree::
   :glob:
   :caption: API implementations

   implementation/detectors/detectors
   implementation/linkers/linkers
   implementation/refiners/refiners

.. toctree::
   :glob:
   :caption: Icy support

   icy

.. toctree::
   :glob:
   :caption: Visualization

   visualize
