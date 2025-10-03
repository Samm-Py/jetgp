#####
JetGP
#####

A Gaussian Process library with support for arbitrary-order derivative-enhanced training data.

Installation
============

Anaconda
--------

Ensure that Anaconda distribution is installed on your system. `Click here <https://www.anaconda.com/docs/getting-started/anaconda/install>`_ for installation steps

Cloning the repository
----------------------

.. code-block:: bash

   $ git clone git@github.com:Samm-Py/oti_gp.git

Conda environment
-----------------

Set up the dependencies of this repository using the ``environment.yml`` file.

1. Go to root of the cloned repository. Create and activate the conda environment with the supplied ``environment.yml`` file at root:

.. code-block:: bash

   $ cd <path-to-oti_gp>
   $ conda env create -f environment.yml
   $ conda activate otigp

In the event where dependencies are added, the ``optigp`` environment can be updated

.. code-block:: bash

    $ conda env update --file environment.yml --prune

Local documentation build
=========================

The documentation of the library can be built locally.

1. Ensure that conda environment is activated

.. code-block:: bash

   $ conda activate otigp

2. Change directory to ``docs`` directory and make a ``build`` directory.

.. code-block:: bash

   $ cd docs
   $ mkdir build

3. Build and open the html documentation (i.e. using Firefox browser)

.. code-block:: bash

   $ sphinx-build -M html source build
   $ cd build/html
   $ firefox index.html