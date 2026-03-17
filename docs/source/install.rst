Installation
============

Recommended
-----------

We recommend to install the library from `Pypy <https://pypi.org/project/byotrack/>`_

.. code-block:: bash

    $ pip install byotrack


From source
-----------

You can also install the latest commit from source

.. code-block:: bash

    $ git clone https://github.com/raphaelreme/byotrack
    $ cd byotrack
    $ pip install .

Development
-----------

We advise to use UV for development. Please follow the guideline detailed in the Contributing Guide.

For pip--only users, you may install all the dependencies with:

.. code-block:: bash

    $ pip install .[full] --group dev --group doc

Additional requirements
-----------------------


Some components require additional dependencies that are not installed with the library by default.
For these components, you need to install their specific dependencies. Here is the complete list:

* StarDistDetector
    * StarDist (+ Tensorflow): `Install StarDist <https://github.com/stardist/stardist#installation>`_
* IcyEMHTLinker
    * Icy: `Download Icy <https://icy.bioimageanalysis.org/download/>`_
    * Spot Tracking Blocks plugin: `Install an Icy plugin <https://icy.bioimageanalysis.org/tutorial/how-to-install-an-icy-plugin/>`_
* TrackMateLinker
    * Fiji: `Download Fiji <https://imagej.net/downloads>`_
* TrackOnStraLinker
    * TrackAstra: `Install trackastra <https://github.com/weigertlab/trackastra#installation>`_

For visualization with `byotrack.visualize` module, you need to `install matplotlib <https://matplotlib.org/stable/install/index.html>`_
