#!/usr/bin/env python3
"""
Fast rebuild script for any static onumm module.
Usage: python build_static.py m10n2
       python build_static.py m5n4
"""
import sys
import os
import re

if len(sys.argv) < 2:
    print("Usage: python build_static.py <mXnY>")
    print("Example: python build_static.py m10n2")
    sys.exit(1)

mn = sys.argv[1]  # e.g., "m10n2"

# Parse m and n values
match = re.match(r'm(\d+)n(\d+)', mn)
if not match:
    print(f"Error: Invalid format '{mn}'. Expected mXnY (e.g., m10n2)")
    sys.exit(1)

m_val, n_val = match.group(1), match.group(2)
module_name = f"onumm{m_val}n{n_val}"  # e.g., "onumm10n2"

PROJECT_ROOT = "/home/sam/research_head/otilib-master"
BUILD_DIR = f"{PROJECT_ROOT}/build"

os.environ['CFLAGS'] = os.environ.get('CFLAGS', '') + f" -I{PROJECT_ROOT}/include"
os.environ['LDFLAGS'] = os.environ.get('LDFLAGS', '') + f" -L{BUILD_DIR}/lib"

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

# Libraries
libraries = ['oti', 'oticwrap', 'otistatic', 'gfortran', mn]

print(f"Libraries: {libraries}")

include_dirs = [numpy.get_include(), f"{PROJECT_ROOT}/include", f"{BUILD_DIR}/lib/"]
extra_compile_args = ["-O3", "-fopenmp"]
extra_link_args = ["-O3", "-fopenmp", f"-L{BUILD_DIR}/lib"]
macros = [("CYTHON_TRACE_NOGIL", "1")]

pyx_path = f"{PROJECT_ROOT}/src/python/pyoti/cython/static/{module_name}.pyx"
if not os.path.exists(pyx_path):
    print(f"Error: {pyx_path} not found")
    sys.exit(1)

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

os.chdir(BUILD_DIR)

setup(
    name=f"pyoti_{mn}",
    ext_modules=cythonize(
        extensions,
        include_path=[f"{PROJECT_ROOT}/include"],
        nthreads=4,
    ),
    script_args=['build_ext', f'-b{BUILD_DIR}', '-j4'],
)

print(f"\nDone! {module_name} built to {BUILD_DIR}/pyoti/static/")
