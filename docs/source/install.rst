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
    $ python setup.py install

Development
-----------

.. code-block:: bash

    $ pip install -r requirements-dev.txt
    # If you need to build the documentation locally
    $ pip install -r docs/requirements.txt

Additional requirements
-----------------------

Some tracker implementations require additional dependencies that are not installed with the library, to use them you need to install their dependencies on your own.
Here is the complete list:

* StarDistDetector
    * stardist (+ tensorflow): `Install stardist <https://github.com/stardist/stardist#installation>`_
* KalmanLinker & KOFTLinker
    * torch_kf: `Install torch-kf <https://github.com/raphaelreme/torch-kf#install>`_
* IcyEMHTLinker
    * Icy: `Download Icy <https://icy.bioimageanalysis.org/download/>`_
    * Spot Tracking Blocks plugin: `Install an Icy plugin <https://icy.bioimageanalysis.org/tutorial/how-to-install-an-icy-plugin/>`_
* TrackMateLinker
    * Fiji: `Download Fiji <https://imagej.net/downloads>`_
    * tifffile: `Install tifffile <https://github.com/cgohlke/tifffile#quickstart>`_
* SkimageOpticalFlow
    * scikit-image: `Install scikit-image <https://scikit-image.org/docs/stable/user_guide/install.html>`_


For visualization with `byotrack.visualize` module, you need to `install matplotlib <https://matplotlib.org/stable/install/index.html>`_
