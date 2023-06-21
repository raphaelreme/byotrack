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
    * stardist (+ tensorflow): `Install stardist <https://github.com/stardist/stardist#installation>_`
* IcyEMHTLinker
    * Icy: `Download Icy <https://icy.bioimageanalysis.org/download/>`_
* DistStitcher and children (EMC2Stitcher)
    * pylapy: `Install pylapy <https://github.com/raphaelreme/pylapy#install>`_
* EMC2Stitcher
    * torch-tps: `Install torch-tps <https://github.com/raphaelreme/torch-tps#install>`_
