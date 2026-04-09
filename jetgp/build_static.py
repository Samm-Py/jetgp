#!/usr/bin/env python3
"""
Build a static pyoti onumm module and install it into the active pyoti package.

As a module:
    from jetgp.build_static import build_module
    build_module('m10n4', '/path/to/otilib-master')

As a script:
    python build_static.py m10n4 /path/to/otilib-master
"""
import os
import re
import sys


def build_module(module_target, otilib_path, build_dir=None):
    """
    Compile a Cython static onumm module and install it into pyoti.static.

    The .so is placed directly into the pyoti package's static/ directory
    (i.e. wherever pyoti is installed in the active Python environment),
    so it is importable as pyoti.static.<module_name> immediately.

    Parameters
    ----------
    module_target : str
        Module target string, e.g. 'm10n4'.
    otilib_path : str
        Path to the otilib-master root directory.
    build_dir : str, optional
        Intermediate build directory. Defaults to <otilib_path>/build.
    """
    import numpy
    from distutils.core import setup
    from distutils.extension import Extension
    from Cython.Build import cythonize
    import pyoti

    match = re.match(r'm(\d+)n(\d+)', module_target)
    if not match:
        raise ValueError(f"Invalid module target '{module_target}', expected mXnY")
    m_val, n_val = match.group(1), match.group(2)
    module_name = f"onumm{m_val}n{n_val}"

    if build_dir is None:
        build_dir = os.path.join(otilib_path, 'build')

    # -b for build_ext is the base output directory; distutils appends the
    # package path, so pyoti.static.onumm* lands at:
    #   <output_base>/pyoti/static/onumm*.so
    # Setting output_base to site-packages puts it alongside the installed pyoti.
    pyoti_dir = pyoti.__path__[0]           # .../site-packages/pyoti
    output_base = os.path.dirname(pyoti_dir) # .../site-packages

    os.environ['CFLAGS'] = (
        os.environ.get('CFLAGS', '') + f" -I{otilib_path}/include"
    )
    os.environ['LDFLAGS'] = (
        os.environ.get('LDFLAGS', '') + f" -L{build_dir}/lib"
    )

    pyx_path = os.path.join(
        otilib_path, 'src', 'python', 'pyoti', 'cython', 'static',
        f'{module_name}.pyx'
    )
    if not os.path.exists(pyx_path):
        raise RuntimeError(f"Cython source not found: {pyx_path}")

    extensions = [
        Extension(
            f"pyoti.static.{module_name}",
            sources=[pyx_path],
            include_dirs=[
                numpy.get_include(),
                os.path.join(otilib_path, 'include'),
                os.path.join(build_dir, 'lib'),
            ],
            define_macros=[("CYTHON_TRACE_NOGIL", "1")],
            libraries=['oti', 'oticwrap', 'otistatic', 'gfortran', module_target],
            extra_compile_args=["-O3", "-fopenmp"],
            extra_link_args=["-O3", "-fopenmp", f"-L{build_dir}/lib"],
        )
    ]

    orig_dir = os.getcwd()
    os.chdir(build_dir)
    try:
        setup(
            name=f"pyoti_{module_target}",
            ext_modules=cythonize(
                extensions,
                include_path=[os.path.join(otilib_path, 'include')],
                nthreads=4,
            ),
            script_args=['build_ext', f'-b{output_base}', '-j4'],
        )
    finally:
        os.chdir(orig_dir)

    print(f"\nDone! {module_name} installed to {pyoti_dir}/static/")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python build_static.py <mXnY> <otilib_path>")
        print("Example: python build_static.py m10n4 /path/to/otilib-master")
        sys.exit(1)
    build_module(sys.argv[1], sys.argv[2])
