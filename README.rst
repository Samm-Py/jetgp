#####
JetGP
#####

A Gaussian Process library with support for arbitrary-order derivative-enhanced training data.

Installation
============

Anaconda
--------

Ensure that the Anaconda distribution is installed on your system. `Click here <https://www.anaconda.com/docs/getting-started/anaconda/install>`_ for installation steps.

Cloning the repository
----------------------

.. code-block:: bash

   $ git clone git@github.com:Samm-Py/jetgp.git

Conda environment
-----------------

Set up the dependencies of this repository using the ``environment.yml`` file.

1. Go to the root of the cloned repository. Create and activate the conda environment with the supplied ``environment.yml`` file at root:

.. code-block:: bash

   $ cd <path-to-oti_gp>
   $ conda env create -f environment.yml
   $ conda activate jetgp

In the event where dependencies are added, the ``jetgp`` environment can be updated:

.. code-block:: bash

   $ conda env update --file environment.yml --prune

Add JetGP to Python Path
------------------------

To make the ``jetgp`` library importable from anywhere, it must be added to your Python path.  
There are two recommended ways to do this:

**Option 1: Temporary addition using ``PYTHONPATH``**

.. code-block:: bash

   # From  ``.\jetgp-main\jetgp``
   $ export PYTHONPATH=$PYTHONPATH:$(pwd)

   # Optional: verify that JetGP is accessible
   $ python -c "import jetgp; print('JetGP successfully added to PYTHONPATH')"

To make this change permanent, add the export line to your shell configuration file (e.g., ``~/.bashrc`` or ``~/.zshrc``).

**Option 2: Persistent addition using Conda (recommended for Anaconda users)**

If using Anaconda, you can register the repository path with your environment using ``conda develop``  
(from the ``conda-build`` package):

.. code-block:: bash

   $ conda install conda-build
   $ cd <path-to-oti_gp> (e.g., ``.\jetgp-main\jetgp``).
   $ conda develop .

This method automatically makes ``jetgp`` importable whenever the ``jetgp`` environment is active.

Local documentation build
=========================

The documentation of the library can be built locally.

1. Ensure that the conda environment is activated:

.. code-block:: bash

   $ conda activate jetgp

2. Change directory to the ``docs`` directory and make a ``build`` directory:

.. code-block:: bash

   $ cd docs
   $ mkdir build

3. Build and open the HTML documentation (e.g., using Firefox browser):

.. code-block:: bash

   $ sphinx-build -M html source build
   $ cd build/html
   $ firefox index.html
