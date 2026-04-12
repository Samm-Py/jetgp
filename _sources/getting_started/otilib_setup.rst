otilib Setup
============

JetGP uses `otilib <https://github.com/mauriaristi/otilib>`_ as its OTI (Order Truncated Imaginary)
arithmetic backend. A small set of patches are required before building the library. The
``setup_otilib`` utility automates this entire process.

Prerequisites
-------------

- The ``jetgp`` conda environment must be active (see :ref:`installation <conda-environment>`).
- A local clone of otilib:

.. code-block:: bash

   $ git clone https://github.com/mauriaristi/otilib.git otilib-master

What the setup does
-------------------

The setup script always performs steps 1–2, then optionally step 3:

1. **Copies patched source files** from ``otilib_mods/`` into otilib-master. The patches add:

   - OpenMP support to the static module compile flags (``src/CMakeLists.txt``)
   - ``python3`` Cython build command (``src/python/pyoti/CMakeLists.txt``)
   - ``oti.empty()`` uninitialized array allocator (``creators.pxi``)
   - OpenMP thread query imports (``include.pxi``)
   - ``{arr_get_all_derivs}`` expansion to the array base template (``array_base.pxi``)
   - JetGP's ``cmod_writer.py`` to both otilib locations (``build/pyoti/`` and ``src/python/pyoti/python/``)
   - The ``regenerate_all_c.py`` and ``build_static.py`` build scripts (``build/``)

2. **Rewrites hardcoded absolute paths** in the otilib build scripts
   (``regenerate_all_c.py``, ``build_static.py``, ``rebuild_all_static.py``,
   ``rebuild_all_static.sh``) to match your machine's otilib location and active Python
   executable. Also saves the otilib path to ``~/.config/jetgp/otilib_path`` for
   runtime auto-detection.

3. **Full build** (only when ``--build`` is passed):

   - Regenerates all C/Cython sources from templates
   - Runs ``cmake ..``, ``make -j<workers>``, and ``make gendata``
   - Writes a ``.pth`` file into site-packages so that ``pyoti`` is importable
     directly from ``otilib-master/build/`` (equivalent to ``conda develop .``)
   - Compiles all Cython static modules in parallel — ``.so`` files land in
     ``otilib-master/build/pyoti/static/`` and are immediately importable

Usage
-----

With the ``jetgp`` environment active and from the ``jetgp`` directory, run:

.. code-block:: bash

   $ python -m jetgp.setup_otilib --otilib ../otilib-master --build --workers 8

This patches otilib with JetGP's required modifications and runs the full build.
To patch without building, omit ``--build``.

File map
--------

The following files are copied from ``otilib_mods/`` during setup:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - File in ``otilib_mods/``
     - Destination in otilib-master
   * - ``src_CMakeLists.txt``
     - ``src/CMakeLists.txt``
   * - ``src_python_pyoti_CMakeLists.txt``
     - ``src/python/pyoti/CMakeLists.txt``
   * - ``regenerate_all_c.py``
     - ``build/regenerate_all_c.py``
   * - ``build_static.py``
     - ``build/build_static.py``
   * - ``cmod_writer.py``
     - ``build/pyoti/cmod_writer.py``
   * - ``cmod_writer.py``
     - ``src/python/pyoti/python/cmod_writer.py``
   * - ``creators.pxi``
     - ``src/python/pyoti/python/source_conv/src/python/pyoti/cython/static/number/creators.pxi``
   * - ``include.pxi``
     - ``src/python/pyoti/python/source_conv/src/python/pyoti/cython/static/number/include.pxi``
   * - ``array_base.pxi``
     - ``src/python/pyoti/python/source_conv/src/python/pyoti/cython/static/number/array/base.pxi``
   * - ``rebuild_all_static.py``
     - ``build/rebuild_all_static.py``
   * - ``rebuild_all_static.sh``
     - ``build/rebuild_all_static.sh``
