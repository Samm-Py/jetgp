#!/usr/bin/env python3
"""
Fast rebuild script for any static onumm module.

Usage as script:
    python build_static.py m10n2
    python build_static.py m5n4

Usage as module:
    from jetgp.build_static import build_module
    build_module('m10n2', otilib_path='/path/to/otilib-master')
"""
import sys
import os
import re


def build_module(module_target, otilib_path=None, build_dir=None):
    """
    Build and install a PyOTI static module.

    Parameters
    ----------
    module_target : str
        Module target name (e.g., 'm1n8', 'm4n2').
    otilib_path : str, optional
        Path to otilib-master directory. If None, attempts auto-detection.
    build_dir : str, optional
        Path to the build directory. If None, uses otilib_path/build.
    """
    # Parse m and n values
    match = re.match(r'm(\d+)n(\d+)', module_target)
    if not match:
        raise ValueError(f"Invalid format '{module_target}'. Expected mXnY (e.g., m10n2)")

    m_val, n_val = match.group(1), match.group(2)
    module_name = f"onumm{m_val}n{n_val}"  # e.g., "onumm10n2"

    # Get otilib path
    if otilib_path is None:
        otilib_path = _get_otilib_path()

    if build_dir is None:
        build_dir = os.path.join(otilib_path, 'build')

    # Set environment variables
    os.environ['CFLAGS'] = os.environ.get('CFLAGS', '') + f" -I{otilib_path}/include"
    os.environ['LDFLAGS'] = os.environ.get('LDFLAGS', '') + f" -L{build_dir}/lib"

    # Import build dependencies (done here to allow environment setup first)
    from distutils.core import setup
    from distutils.extension import Extension
    from Cython.Build import cythonize
    import numpy

    # Libraries
    libraries = ['oti', 'oticwrap', 'otistatic', 'gfortran', module_target]

    print(f"Building module: {module_name}")
    print(f"  otilib_path: {otilib_path}")
    print(f"  build_dir: {build_dir}")
    print(f"  Libraries: {libraries}")

    include_dirs = [numpy.get_include(), f"{otilib_path}/include", f"{build_dir}/lib/"]
    extra_compile_args = ["-O3", "-fopenmp"]
    extra_link_args = ["-O3", "-fopenmp", f"-L{build_dir}/lib"]
    macros = [("CYTHON_TRACE_NOGIL", "1")]

    pyx_path = f"{otilib_path}/src/python/pyoti/cython/static/{module_name}.pyx"
    if not os.path.exists(pyx_path):
        raise FileNotFoundError(f"Cython source not found: {pyx_path}")

    extensions = [
        Extension(
            f"pyoti.static.{module_name}",
            sources=[pyx_path],
            include_dirs=include_dirs,
            define_macros=macros,
            libraries=libraries,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ]

    # Save current directory and change to build_dir
    original_dir = os.getcwd()
    os.chdir(build_dir)

    try:
        setup(
            name=f"pyoti_{module_target}",
            ext_modules=cythonize(
                extensions,
                include_path=[f"{otilib_path}/include"],
                nthreads=4,
            ),
            script_args=['build_ext', f'-b{build_dir}', '-j4'],
        )

        print(f"\nDone! {module_name} built to {build_dir}/pyoti/static/")

    finally:
        # Restore original directory
        os.chdir(original_dir)


def _get_otilib_path():
    """
    Auto-detect otilib path from environment or installed pyoti.

    Returns
    -------
    otilib_path : str
        Path to otilib-master directory.
    """
    # Check environment variable first
    otilib_path = os.environ.get('OTILIB_PATH')

    if otilib_path is not None:
        return otilib_path

    # Try to auto-detect from installed pyoti
    try:
        import pyoti

        # Get the pyoti package location
        if hasattr(pyoti, '__path__'):
            pyoti_install_path = pyoti.__path__[0]
        elif hasattr(pyoti, '__file__'):
            pyoti_install_path = os.path.dirname(pyoti.__file__)
        else:
            raise AttributeError("Cannot determine pyoti installation path")

        # Navigate up from pyoti to find otilib root
        current = pyoti_install_path
        for _ in range(6):
            parent = os.path.dirname(current)

            # Check if this looks like otilib root
            potential_build = os.path.join(parent, 'build')
            potential_cmake = os.path.join(parent, 'CMakeLists.txt')

            if os.path.isdir(potential_build) or os.path.isfile(potential_cmake):
                return parent

            current = parent

    except ImportError:
        pass

    raise RuntimeError(
        "Could not auto-detect otilib path. Please either:\n"
        "  1. Set the OTILIB_PATH environment variable\n"
        "  2. Pass otilib_path explicitly to build_module()"
    )


# Allow running as a script
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python build_static.py <mXnY>")
        print("Example: python build_static.py m10n2")
        sys.exit(1)

    module_target = sys.argv[1]

    # Optional: allow passing otilib_path as second argument
    otilib_path = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        build_module(module_target, otilib_path=otilib_path)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
